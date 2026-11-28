"""Isolated research enrollment is never production registration or attendance."""
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

import numpy as np

from facial_attendance.recognition.contracts import FaceInferenceContract
from facial_attendance.recognition.evaluation import (
    CONDITIONS, EvaluationResult, RecognitionEvaluation, SandboxInference, SandboxReports, frame_fingerprint,
)
from facial_attendance.recognition.inference import FaceInference
from facial_attendance.ui.webcam_prototype import Completion, Envelope, PrototypeController

BOX=(40,40,180,180)


def unit(index):
    values=[0.]*512;values[index]=1.;return tuple(values)


class FakeInference:
    contract=FaceInferenceContract()
    provenance={'model_id':contract.model_id,'model_version':contract.model_version,
                'threshold':contract.cosine_threshold,'checkpoint_sha256':'a'*64,'preprocessing_version':'fixture'}
    def __init__(self):self.vector=unit(0)
    def embed(self,frame,box):return self.vector


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.session=RecognitionEvaluation(source_kind='generated');self.session.set_consent(True)
        self.inference=FakeInference();self.worker=SandboxInference(self.inference,self.session);self.sequence=0

    def job(self,kind,*,truth='unspecified',value=None,conditions=None):
        self.sequence+=1
        frame=np.full((240,320,3),self.sequence if value is None else value,np.uint8)
        return self.session.job(kind,(1,self.sequence),frame,[BOX],10.,now=10.,truth=truth,conditions=conditions)

    def complete(self,kind,**kw):
        job=self.job(kind,**kw);return self.session.accept(self.worker(job),current_source=job.source,now=10.05)

    def enroll(self):
        self.session.set_consent(True)
        for _ in range(3):self.complete('reference')
        test_id=self.session.commit_reference();self.session.set_consent(True)
        return test_id

    def test_empty_gallery_is_new_isolated_and_production_gates_stay_false(self):
        other=RecognitionEvaluation();self.enroll()
        self.assertEqual(other.test_ids,());self.assertFalse(self.session.can_save)
        self.assertFalse(Envelope().can_save);self.assertFalse(PrototypeController().state.can_save)
        with self.assertRaises(PermissionError):self.session.matcher.register_live('someone','Name')

    def test_consent_required_before_reference_evaluation_or_preview(self):
        self.session.set_consent(False)
        for kind in ('reference','evaluation','preview'):
            with self.assertRaises(PermissionError):self.job(kind)

    def test_enrollment_has_generated_alias_and_minimum_independent_frames(self):
        self.complete('reference')
        with self.assertRaises(ValueError):self.session.commit_reference()
        self.complete('reference');self.complete('reference');self.session.commit_reference()
        person=self.session.matcher._gallery.people[0]
        self.assertEqual(person.person_id,'test-001');self.assertEqual(person.name,'Test subject 001')
        self.assertEqual(person.n_images,3);self.assertFalse(self.session.consent)

    def test_named_test_profile_matches_and_stays_out_of_reports(self):
        for _ in range(3):
            self.complete('reference')
        with self.assertRaisesRegex(ValueError, "single-line"):
            self.session.commit_reference(display_name="Test\nName")
        test_id = self.session.commit_reference(display_name="Demo V")
        self.session.set_consent(True)
        decision = self.complete('evaluation', truth=test_id)
        self.assertEqual((decision.test_id, decision.test_name), ("test-001", "Demo V"))
        self.assertNotIn("Demo V", json.dumps(self.session.report()))
        self.assertFalse(decision.can_save)

    def test_reference_and_evaluation_same_exact_frame_blocked_in_both_directions(self):
        self.complete('reference',value=10)
        with self.assertRaisesRegex(ValueError,'Reference frame'):self.complete('evaluation',value=10)
        self.complete('evaluation',value=20,truth='unknown')
        with self.assertRaisesRegex(ValueError,'already used'):self.complete('reference',value=20)

    def test_preview_cannot_present_reference_frame_as_a_success(self):
        self.complete('reference',value=10)
        with self.assertRaisesRegex(ValueError,'Reference frame'):self.complete('preview',value=10)
        self.assertEqual(self.session.records,[])

    def test_duplicate_reference_and_duplicate_evaluation_frames_are_rejected(self):
        self.complete('reference',value=10)
        with self.assertRaises(ValueError):self.complete('reference',value=10)
        self.complete('evaluation',value=30)
        with self.assertRaisesRegex(ValueError,'already recorded'):self.complete('evaluation',value=30)
        self.assertEqual(len(self.session.records),1)

    def test_known_accept_and_unknown_reject_use_unchanged_adapter_threshold(self):
        test_id=self.enroll();decision=self.complete('evaluation',truth=test_id)
        self.assertTrue(decision.accepted);self.assertAlmostEqual(decision.score,1.)
        self.assertEqual(decision.threshold,0.4619967254593038);self.assertFalse(decision.can_save)
        self.inference.vector=unit(1);decision=self.complete('evaluation',truth='unknown')
        self.assertFalse(decision.accepted);self.assertEqual(decision.score,0.)
        self.assertEqual(self.session.report()['counts']['correct_accept'],1)
        self.assertEqual(self.session.report()['counts']['correct_reject'],1)

    def test_false_accept_false_reject_and_unscored_counts(self):
        test_id=self.enroll();self.complete('evaluation',truth='unknown')
        self.inference.vector=unit(1);self.complete('evaluation',truth=test_id)
        self.complete('evaluation')
        counts=self.session.report()['counts']
        self.assertEqual(counts['false_accept'],1);self.assertEqual(counts['false_reject'],1);self.assertEqual(counts['unscored'],1)
        self.assertNotIn('accuracy',self.session.report())

    def test_wrong_known_identity_is_reported_as_misidentification(self):
        first=self.enroll();self.inference.vector=unit(1);self.enroll()
        self.complete('evaluation',truth=first)
        self.assertEqual(self.session.records[-1]['outcome'],'false_accept')
        self.assertTrue(self.session.records[-1]['wrong_identity'])

    def test_photograph_accepts_are_separate_and_not_liveness_success(self):
        test_id=self.enroll();conditions={k:v[0] for k,v in CONDITIONS.items()};conditions['presentation']='photograph'
        self.complete('evaluation',truth=test_id,conditions=conditions)
        report=self.session.report();self.assertEqual(report['photo_accepts'],1)
        self.assertEqual(report['live_person_counts']['correct_accept'],0)
        self.assertFalse(report['liveness_evaluated']);self.assertFalse(report['can_save'])

    def test_no_face_and_multiple_faces_cannot_generate_jobs(self):
        frame=np.zeros((240,320,3),np.uint8)
        for boxes in ([],[BOX,BOX],[(-1,0,20,20)]):
            with self.assertRaises(ValueError):self.session.job('reference',(1,1),frame,boxes,10.,now=10.)
        self.assertIsNone(self.session.pending)

    def test_expired_frame_and_expired_completion_are_rejected(self):
        with self.assertRaises(ValueError):self.session.job('evaluation',(1,1),np.zeros((240,320,3),np.uint8),[BOX],10.,now=11.)
        job=self.job('reference')
        with self.assertRaises(ValueError):self.session.accept(self.worker(job),current_source=job.source,now=10.81)
        self.assertEqual(self.session.references,[])

    def test_new_face_source_no_face_consent_withdrawal_reject_late_sample(self):
        for change in ('source','none','consent'):
            self.setUp();job=self.job('reference');result=self.worker(job)
            if change=='consent':self.session.set_consent(False)
            with self.assertRaises(ValueError):self.session.accept(result,current_source=None if change=='none' else (2,9),now=10.1)
            self.assertEqual(self.session.references,[])

    def test_reset_and_discard_reject_inflight_results(self):
        for reset in (lambda:self.session.reset(),lambda:self.session.discard_draft()):
            job=self.job('reference');result=self.worker(job);reset()
            with self.assertRaises(ValueError):self.session.accept(result,current_source=job.source,now=10.1)
            self.session.set_consent(True)

    def test_duplicate_enrollment_does_not_overwrite_test_gallery(self):
        self.enroll()
        for _ in range(3):self.complete('reference')
        with self.assertRaises(ValueError):self.session.commit_reference()
        self.assertEqual(self.session.test_ids,('test-001',))

    def test_inconsistent_reference_vectors_do_not_form_a_fake_template(self):
        for i in (0,1,2):self.inference.vector=unit(i);self.complete('reference')
        # Orthogonal mean passes .462; opposing vectors must fail instead.
        self.session.references[1]=tuple(-v for v in unit(0))
        with self.assertRaises(ValueError):self.session.commit_reference()
        self.assertEqual(self.session.test_ids,())

    def test_gallery_roundtrip_keeps_metadata_and_matching_without_files(self):
        test_id=self.enroll();before=self.session.matcher.match(unit(0))
        self.session.reload_in_memory();after=self.session.matcher.match(unit(0))
        self.assertEqual(before,after);self.assertEqual(after.person_id,test_id)
        self.assertEqual(self.session.reload_checks,1)

    def test_full_reset_releases_all_session_references_records_and_consent(self):
        self.enroll();self.complete('evaluation');old=self.session.run_id;self.session.reset()
        self.assertNotEqual(old,self.session.run_id);self.assertEqual(self.session.test_ids,())
        self.assertEqual(self.session.records,[]);self.assertEqual(self.session.references,[])
        self.assertEqual(self.session.reference_hashes,set());self.assertEqual(self.session.evaluation_hashes,set())
        self.assertFalse(self.session.consent)

    def test_model_version_and_threshold_must_match_selected_contract(self):
        bad=FakeInference();bad.provenance={**bad.provenance,'model_version':'different'}
        with self.assertRaises(ValueError):SandboxInference(bad,self.session)
        bad.provenance={**FakeInference.provenance,'threshold':.1}
        with self.assertRaises(ValueError):SandboxInference(bad,self.session)

    def test_wrong_context_or_model_result_is_not_relabelled(self):
        for mismatch in ('context','model'):
            job=self.job('preview');result=self.worker(job)
            completion=result.completion
            completion=replace(completion,context=replace(completion.context,face_id='old')) if mismatch=='context' else replace(completion,result=replace(completion.result,model_version='bad'))
            with self.assertRaises(ValueError):self.session.accept(replace(result,completion=completion),current_source=job.source,now=10.1)

    def test_personal_free_text_conditions_and_truth_are_not_allowed(self):
        with self.assertRaises(ValueError):self.job('evaluation',truth='Private Real Name')
        with self.assertRaises(ValueError):self.job('evaluation',conditions={'name':'Private'})

    def test_score_export_has_no_frames_embeddings_hashes_or_local_paths(self):
        self.enroll();self.complete('evaluation',truth='test-001')
        with TemporaryDirectory() as directory:
            store=SandboxReports(directory);path=store.export(self.session);text=path.read_text();data=json.loads(text)
            for word in ('embeddings','fingerprint','reference_hashes','/Users/','frame'):
                self.assertNotIn(word,text)
            self.assertFalse(data['physical_samples_recorded']);self.assertEqual(data['source'],'generated')
            store.delete(self.session.run_id);self.assertFalse(path.exists())

    def test_export_directory_cannot_be_redirected_to_a_production_path(self):
        with TemporaryDirectory() as directory,TemporaryDirectory() as target:
            root=Path(directory);(root/'artifacts').mkdir();(root/'artifacts'/'recognition-evaluation').symlink_to(target,target_is_directory=True)
            with self.assertRaises(ValueError):SandboxReports(root).export(self.session)
            self.assertEqual(list(Path(target).iterdir()),[])

    def test_cleanup_refuses_traversal_and_preserves_unexpected_files(self):
        with TemporaryDirectory() as directory:
            store=SandboxReports(directory)
            with self.assertRaises(ValueError):store.delete('../../production')
            path=store.export(self.session);extra=path.parent/'do-not-delete.txt';extra.write_text('unrelated')
            with self.assertRaises(OSError):store.delete(self.session.run_id)
            self.assertTrue(extra.exists())

    def test_export_replaces_hardlink_without_overwriting_its_external_target(self):
        with TemporaryDirectory() as directory:
            store=SandboxReports(directory);path=store.export(self.session);path.unlink()
            protected=Path(directory)/'production.json';protected.write_text('protected')
            os.link(protected,path);store.export(self.session)
            self.assertEqual(protected.read_text(),'protected')

    def test_delete_all_removes_only_owned_score_runs(self):
        with TemporaryDirectory() as directory:
            store=SandboxReports(directory);first=store.export(self.session)
            self.session.reset();second=store.export(self.session)
            store.delete_all();self.assertFalse(first.exists());self.assertFalse(second.exists())

    def test_delete_all_refuses_unexpected_content_without_deleting_it(self):
        with TemporaryDirectory() as directory:
            store=SandboxReports(directory);score=store.export(self.session)
            unrelated=score.parent/'unrelated-gallery.json';unrelated.write_text('unrelated')
            with self.assertRaises(ValueError):store.delete_all()
            self.assertTrue(unrelated.exists());self.assertTrue(score.exists())

    def test_fingerprint_uses_actual_pixels_and_shape_not_capture_id(self):
        frame=np.zeros((240,320,3),np.uint8)
        self.assertEqual(frame_fingerprint(frame),frame_fingerprint(frame.copy()))
        self.assertNotEqual(frame_fingerprint(frame),frame_fingerprint(frame+1))


@unittest.skipUnless(os.environ.get('FA_FACE_CHECKPOINT'),'Set FA_FACE_CHECKPOINT for actual sandbox model check')
class ActualSandboxModelTests(unittest.TestCase):
    def test_selected_checkpoint_reference_and_distinct_sample_use_same_adapter(self):
        from facial_attendance.recognition.gallery_matching import controlled_frame,CONTROLLED_BOX
        inference=FaceInference(os.environ['FA_FACE_CHECKPOINT']).load()
        session=RecognitionEvaluation(source_kind='generated');session.set_consent(True);service=SandboxInference(inference,session)
        for i in range(3):
            job=session.job('reference',(1,i),controlled_frame(i),[CONTROLLED_BOX],10.,now=10.)
            session.accept(service(job),current_source=job.source,now=10.05)
        session.commit_reference();session.set_consent(True)
        job=session.job('evaluation',(1,5),controlled_frame(5),[CONTROLLED_BOX],11.,now=11.,truth='test-001')
        decision=session.accept(service(job),current_source=job.source,now=11.05)
        self.assertTrue(decision.accepted);self.assertFalse(decision.can_save)
        self.assertEqual(inference.provenance['model_version'],'cb38ea47ceff')
        self.assertEqual(len(inference.embed(controlled_frame(6),CONTROLLED_BOX)),512)


class SandboxWorkerCleanupTests(unittest.TestCase):
    def test_reset_cleanup_does_not_spawn_replacements_while_old_native_work_runs(self):
        from facial_attendance.ui.recognition_sandbox import RecognitionSandboxApp
        app=RecognitionSandboxApp.__new__(RecognitionSandboxApp);app._cleanup_pending=True
        app.worker=Mock();app.detector=Mock();app.worker.thread.is_alive.return_value=True
        with patch('facial_attendance.ui.recognition_sandbox.SingleWorker') as factory:
            self.assertFalse(app._ensure_workers());factory.assert_not_called()

    def test_reset_reuses_model_service_after_old_workers_exit(self):
        from facial_attendance.ui.recognition_sandbox import RecognitionSandboxApp
        app=RecognitionSandboxApp.__new__(RecognitionSandboxApp);app._cleanup_pending=True
        app.worker=Mock();app.detector=Mock();app.service=object();app._detector_fn=object()
        app.worker.thread.is_alive.return_value=False;app.detector.thread.is_alive.return_value=False
        with patch('facial_attendance.ui.recognition_sandbox.SingleWorker') as factory:
            self.assertTrue(app._ensure_workers());self.assertEqual(factory.call_count,2)
            self.assertIs(factory.call_args.args[0],app.service)
            app._ensure_workers();self.assertEqual(factory.call_count,2)


if __name__=='__main__':unittest.main()
