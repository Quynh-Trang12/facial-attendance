"""UI-03 trained inference, isolated enrollment and unchanged authorization tests."""

from copy import deepcopy
from contextlib import redirect_stderr
from dataclasses import replace
from hashlib import sha256
import math
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
import unittest
from unittest.mock import Mock, patch

import numpy as np

from facial_attendance.recognition.contracts import FaceInferenceContract, SELECTED_COSINE_THRESHOLD
from facial_attendance.recognition.inference import (
    CHECKPOINT_SHA256, FaceCheckpointMismatch, FaceInference, FaceInferenceError,
    FaceUnavailable, cosine, crop_box, normalized_vector,
)
from facial_attendance.recognition.gallery_matching import (
    CONTROLLED_BOX, ControlledTestEnrollment, GalleryMatcher, PreparedTestEnrollment,
    controlled_frame, empty_gallery,
)
from facial_attendance.ui.gallery import FaceGallery, GalleryFormatError, IncompatibleGalleryError
from facial_attendance.ui.recognition_adapter import TrainedPreviewAdapters, TrainedRecognitionAdapter
from facial_attendance.ui.live_preview import LiveController, SingleWorker, WorkResult, evaluate_adapters
from facial_attendance.ui.webcam_app import LiveWebcamApp, main
from facial_attendance.ui.webcam_prototype import Completion, Liveness, PrototypeController


def unit(axis=0):
    return tuple(1.0 if i == axis else 0.0 for i in range(512))


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.gallery = empty_gallery()
        self.matcher = GalleryMatcher(self.gallery)

    def test_empty_gallery_is_valid_unknown(self):
        result = self.matcher.match(unit())
        self.assertFalse(result.accepted)
        self.assertEqual((result.person_id, result.name, result.score), (None, None, None))

    def test_known_name_and_unknown_rejection(self):
        self.gallery.register("test-001", "Test Person", unit(), n_images=5)
        self.assertEqual(self.matcher.match(unit()).name, "Test Person")
        result = self.matcher.match(unit(1))
        self.assertEqual((result.accepted, result.person_id, result.name, result.score), (False, None, None, 0.0))

    def test_actual_cosine_scoring_and_threshold_boundary(self):
        self.assertEqual(cosine(unit(), unit()), 1)
        self.assertEqual(cosine(unit(), unit(1)), 0)
        self.assertEqual(cosine(unit(), tuple(-v for v in unit())), -1)
        contract = FaceInferenceContract()
        self.assertTrue(contract.accepts_cosine(SELECTED_COSINE_THRESHOLD))
        self.assertFalse(contract.accepts_cosine(math.nextafter(SELECTED_COSINE_THRESHOLD, -math.inf)))
        t = SELECTED_COSINE_THRESHOLD
        near = (t, math.sqrt(1-t*t)) + (0.,)*510
        self.assertAlmostEqual(cosine(unit(), near), t, places=15)

    def test_compatible_gallery_and_deterministic_casefold_id_tie(self):
        self.gallery.register("test-B", "Second", unit(), n_images=3)
        self.gallery.register("test-A", "First", unit(), n_images=3)
        # The duplicate check keeps the deterministic tie-break; check-in calls a tie unknown.
        self.assertEqual(self.matcher.match_threshold_only(unit()).person_id, "test-A")
        tie = self.matcher.match(unit())
        self.assertEqual((tie.accepted, tie.person_id, tie.name, tie.score), (False, None, None, 1.0))
        self.gallery.model_version = "different"
        with self.assertRaises(IncompatibleGalleryError): self.matcher.match(unit())

    def test_invalid_embedding_dimensions_values_and_norm(self):
        for vector in ([0.]*512, [1.]*511, [float("nan")]*512, [True]*512, [[1.]*512]):
            with self.subTest(vector=type(vector).__name__), self.assertRaises(ValueError):
                self.matcher.match(vector)
        for vector in ([0.]*512, [float("inf")]*512, [1.]*511, [True]*512):
            with self.assertRaises(ValueError): normalized_vector(vector)

    def test_corrupt_and_incompatible_gallery_loading_fail_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"test-gallery.json"
            path.write_text("{bad json")
            with self.assertRaises(GalleryFormatError):
                FaceGallery.load(path, model_id=self.gallery.model_id, model_version=self.gallery.model_version, embedding_dim=512)
            self.gallery.save(path)
            with self.assertRaises(IncompatibleGalleryError):
                FaceGallery.load(path, model_id=self.gallery.model_id, model_version="other", embedding_dim=512)

    def test_no_live_registration_api_can_use_stub_authorization(self):
        for evidence in (None, Liveness("live", "demo", .08, source="stub"), Liveness("uncertain", "demo", .4)):
            with self.assertRaises(PermissionError): self.matcher.register_live("x", "Name", unit(), liveness=evidence)
        self.assertEqual(self.gallery.people, ())


class ControlledEnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.inference = Mock()
        self.inference.embed.return_value = unit()
        self.enrollment = ControlledTestEnrollment(self.inference)

    def test_five_generated_samples_are_aggregated_then_subsequently_matched(self):
        prepared = self.enrollment.prepare(("test-001", "Generated Person"))
        self.assertEqual(self.inference.embed.call_count, 5)
        for call in self.inference.embed.call_args_list:
            frame, bbox = call.args
            self.assertEqual(bbox, CONTROLLED_BOX)
            self.assertFalse(frame.flags.writeable)
        person = self.enrollment.commit(prepared)
        self.assertEqual(person.n_images, 5)
        self.assertEqual(len(person.embeddings), 1)
        self.assertEqual(self.enrollment.match(unit()).person_id, "test-001")

    def test_invalid_id_and_name_do_not_run_model_or_modify_gallery(self):
        for identity in (("", "Name"), (" id", "Name"), ("id", "x"*65), ("id", "")):
            with self.assertRaises(ValueError): self.enrollment.prepare(identity)
        self.inference.embed.assert_not_called()
        self.assertEqual(self.enrollment._gallery.people, ())

    def test_duplicate_id_and_duplicate_face_are_rejected(self):
        self.enrollment.commit(self.enrollment.prepare(("test-001", "Name")))
        for identity in (("TEST-001", "Other"), ("test-002", "Other")):
            with self.assertRaises(ValueError): self.enrollment.commit(self.enrollment.prepare(identity))
        self.assertEqual(len(self.enrollment._gallery.people), 1)

    def test_count_consistency_and_zero_mean_guards(self):
        minus = tuple(-v for v in unit())
        for vectors in ((unit(),)*2, (unit(),)*11, (unit(),unit(),minus), (unit(),minus,unit(1),tuple(-v for v in unit(1)))):
            with self.subTest(count=len(vectors)), self.assertRaises(ValueError):
                self.enrollment.commit(PreparedTestEnrollment("test-001", "Name", vectors))
        self.assertEqual(self.enrollment._gallery.people, ())

    def test_prepare_does_not_commit_and_reload_matches_in_temporary_gallery(self):
        draft = self.enrollment.prepare(("test-001", "Name"))
        self.assertEqual(self.enrollment._gallery.people, ())
        self.enrollment.commit(draft)
        with TemporaryDirectory(prefix="ui03-controlled-") as directory:
            path = Path(directory)/"generated-test-gallery.json"
            self.enrollment._gallery.save(path)
            gallery = FaceGallery.load(path, model_id=self.enrollment.contract.model_id,
                                       model_version=self.enrollment.contract.model_version, embedding_dim=512)
            self.assertEqual(GalleryMatcher(gallery).match(unit()).name, "Name")


class AdapterAndSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.controller = PrototypeController()
        self.request = self.controller.begin_capture(controlled_frame(), [CONTROLLED_BOX])
        self.inference = Mock(provenance={"model_id": FaceInferenceContract().model_id,
            "model_version": FaceInferenceContract().model_version, "checkpoint_sha256": CHECKPOINT_SHA256,
            "preprocessing_version": "test"})
        self.inference.embed.return_value = unit()
        self.adapter = TrainedRecognitionAdapter(self.inference, GalleryMatcher(empty_gallery()))

    def test_completion_preserves_full_context_and_trained_provenance(self):
        result = self.adapter(self.request)
        self.assertEqual(result.context, self.request.context)
        self.assertEqual(result.result.source, "model")
        self.assertEqual(result.result.model_version, "cb38ea47ceff")
        self.assertTrue(self.controller.receive(result))
        self.assertFalse(self.controller.state.can_save)

    def test_model_and_input_errors_are_typed_without_stub_fallback(self):
        for error, code in ((FaceUnavailable("missing"), "model_unavailable"),
                            (FaceCheckpointMismatch("wrong"), "version_mismatch"),
                            (ValueError("bad crop"), "invalid_input"), (RuntimeError("failed"), "inference_error")):
            self.inference.embed.side_effect = error
            completion = self.adapter(self.request)
            self.assertEqual((completion.result.module, completion.result.code), ("recognition", code))
            self.assertTrue(completion.result.blocks_persistence)

    def test_wrong_version_and_invalid_provenance_are_not_accepted(self):
        bad = replace(self.request, context=replace(self.request.context, contract_version="wrong"))
        self.assertEqual(self.adapter(bad).result.code, "version_mismatch")
        completion = self.adapter(self.request)
        self.controller.receive(replace(completion, result=replace(completion.result, checkpoint_sha256=None)))
        self.assertIsNone(self.controller.state.recognition)

    def test_retry_face_loss_multiple_and_camera_failure_reject_late_face_result(self):
        for invalidate in (lambda c:c.reset(), lambda c:c.capture_error("no_face", "No face"),
                           lambda c:c.capture_error("multiple_faces", "Multiple"),
                           lambda c:c.capture_error("camera_unavailable", "Camera lost")):
            c = PrototypeController(); request = c.begin_capture(controlled_frame(), [CONTROLLED_BOX])
            completion = self.adapter(request); invalidate(c)
            self.assertFalse(c.receive(completion))
            self.assertIsNone(c.state.recognition)
            self.assertFalse(c.state.can_save)

    def test_registration_worker_only_prepares_and_reset_discards_completion(self):
        app = LiveWebcamApp.__new__(LiveWebcamApp)
        app.controller = self.controller; app._generation = 1
        app._enrollment_token = (1, self.controller.attempt_id)
        old = app._enrollment_token
        self.controller.reset()
        app._registration_worker = Mock()
        app._registration_worker.take.return_value = WorkResult(old, "uncommitted draft")
        app.registration_message = Mock(); app.adapters = Mock()
        app._poll_test_enrollment()
        app.adapters.matcher.commit.assert_not_called()

    def test_registration_attempt_spans_captures_but_old_face_completion_is_rejected(self):
        live = LiveController()
        live.controller.reset("registration")
        attempt = live.controller.attempt_id
        live.begin_sample((1,1),10,action="registration")
        first = live.detected(WorkResult((1,1),[CONTROLLED_BOX]),controlled_frame())
        late = self.adapter(first)
        live.begin_sample((1,2),10.3,action="registration")
        second = live.detected(WorkResult((1,2),[CONTROLLED_BOX]),controlled_frame())
        self.assertEqual(second.context.attempt_id,attempt)
        self.assertEqual(second.context.action,"registration")
        self.assertNotEqual(second.context.capture_id,first.context.capture_id)
        self.assertFalse(live.controller.receive(late))
        live.fail("no_face","Face lost")
        self.assertNotEqual(live.controller.attempt_id,attempt)

    def test_visible_test_form_after_retry_cannot_enroll_from_attendance_action(self):
        self.controller.receive(Completion(self.request.context,Liveness("live","demo",.08)))
        self.controller.receive(self.adapter(self.request))
        app=LiveWebcamApp.__new__(LiveWebcamApp)
        app.controller=self.controller;app._controlled_registration=True
        app._registration_worker=Mock(busy=False);app.registration_message=Mock()
        app._start_test_enrollment()
        app._registration_worker.submit.assert_not_called()

    def test_blocked_state_cancels_draft_even_if_worker_finishes_after_state_changes(self):
        for blocked in ("spoof","uncertain","error"):
            app=LiveWebcamApp.__new__(LiveWebcamApp)
            app.controller=self.controller;app._generation=1
            old=(1,self.controller.attempt_id);app._enrollment_token=old
            app._registration_worker=Mock();app._registration_worker.take.return_value=None
            app.registration_message=Mock();app.adapters=Mock()
            self.controller.state=replace(self.controller.state,ui_state=blocked)
            app._poll_test_enrollment()
            self.assertIsNone(app._enrollment_token)
            self.controller.state=replace(self.controller.state,ui_state="checking")
            app._registration_worker.take.return_value=WorkResult(old,"late draft")
            app._poll_test_enrollment()
            app.adapters.matcher.commit.assert_not_called()

    def test_slow_model_work_is_bounded_off_main_thread_and_shutdown_discards(self):
        entered, release = Event(), Event(); thread_ids = []
        def slow(value):
            thread_ids.append(get_ident()); entered.set(); release.wait(3)
            return self.adapter(value)
        worker = SingleWorker(slow, "face-test")
        try:
            self.assertTrue(worker.submit((1,1), self.request)); self.assertTrue(entered.wait(2))
            for i in range(20): self.assertFalse(worker.submit((1,i+2), self.request))
            worker.close(); release.set(); worker.thread.join(2)
            self.assertFalse(worker.thread.is_alive()); self.assertIsNone(worker.take())
            self.assertEqual(len(thread_ids),1); self.assertNotEqual(thread_ids[0],get_ident())
        finally:
            release.set(); worker.close()

    def test_cli_refuses_controlled_enrollment_with_camera_or_external_gallery(self):
        for args in (("--controlled-registration-demo", "--face-checkpoint", "file"),
                     ("--controlled-registration-demo", "--demo-camera", "--face-checkpoint", "file", "--gallery", "real.json"),
                     ("--dev-manual", "--face-checkpoint", "file")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error: main(list(args))
            self.assertEqual(error.exception.code, 2)


@unittest.skipUnless(os.environ.get("FA_FACE_CHECKPOINT"), "Set FA_FACE_CHECKPOINT for actual supplied Face weights")
class SuppliedFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.path = Path(os.environ["FA_FACE_CHECKPOINT"])
        cls.engine = FaceInference(cls.path).load()

    def test_actual_file_hash_weights_version_architecture_and_head(self):
        from facial_attendance.recognition.model import model_version
        self.assertEqual(sha256(self.path.read_bytes()).hexdigest(), CHECKPOINT_SHA256)
        self.assertEqual(model_version(self.engine.model.state_dict()), "cb38ea47ceff")
        self.assertIsNone(self.engine.model.head)
        self.assertFalse(self.engine.model.training)
        self.assertIsNone(self.engine.provenance["checkpoint_epoch"])

    def test_cpu_weights_only_loader_is_cached(self):
        with patch("torch.load", wraps=self.torch.load) as load:
            engine = FaceInference(self.path).load(); engine.load()
        load.assert_called_once()
        self.assertTrue(load.call_args.kwargs["weights_only"])
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

    def test_hash_failure_happens_before_deserialization_and_missing_is_explicit(self):
        with patch("torch.load") as load, self.assertRaises(FaceCheckpointMismatch):
            FaceInference(self.path, expected_sha256="0"*64).load()
        load.assert_not_called()
        with self.assertRaises(FaceUnavailable): FaceInference(self.path.parent/"missing-face.pt").load()

    def test_changed_metadata_or_weight_tensors_are_rejected(self):
        payload = self.torch.load(self.path,map_location="cpu",weights_only=True)
        for field, value in (("model_id","wrong"),("model_version","0"*12),("architecture","wrong"),("approach","classification"),("threshold",.5)):
            bad = {"state_dict":payload["state_dict"], "meta":{**payload["meta"],field:value}}
            with patch("torch.load", return_value=bad), self.assertRaises(FaceCheckpointMismatch): FaceInference(self.path).load()
        bad = deepcopy(payload); bad["state_dict"]["embedding.weight"].add_(.1)
        with patch("torch.load", return_value=bad), self.assertRaises(FaceCheckpointMismatch): FaceInference(self.path).load()

    def test_bgr_to_rgb_tensor_matches_authoritative_eval_transform_without_mutation(self):
        from PIL import Image
        from facial_attendance.recognition.model import eval_transform
        frame = np.full((160,180,3),(11,22,33),dtype=np.uint8); original=frame.copy();frame.flags.writeable=False
        tensor = self.engine.preprocess(frame,(40,40,120,120))
        expected = eval_transform()(Image.new("RGB",(96,96),(33,22,11))).unsqueeze(0)
        self.assertTrue(self.torch.equal(tensor,expected))
        self.assertEqual((tuple(tensor.shape),tensor.dtype), ((1,3,64,64),self.torch.float32))
        np.testing.assert_array_equal(frame,original)

    def test_source_coordinate_square_clipping_and_invalid_boxes(self):
        frame = np.zeros((100,120,3),dtype=np.uint8)
        self.assertEqual(crop_box(frame,(0,0,80,80)),(0,0,80,80))
        self.assertEqual(crop_box(frame,(10,5,90,95)),(10,5,90,95))
        for bbox in ((-1,0,80,80),(0,0,63,80),(1,1,1,90),(0,0,121,100),(True,0,80,80)):
            with self.assertRaises(ValueError): self.engine.embed(frame,bbox)

    def test_real_512d_embeddings_are_finite_normalized_and_deterministic(self):
        a=self.engine.embed(controlled_frame(),CONTROLLED_BOX)
        b=self.engine.embed(controlled_frame(),CONTROLLED_BOX)
        self.assertEqual(len(a),512); self.assertEqual(a,b)
        FaceInferenceContract().validate_normalized_embedding(a)
        self.assertAlmostEqual(cosine(a,b),1)

    def test_invalid_raw_model_output_is_never_a_match(self):
        for raw in (self.torch.zeros(1,511),self.torch.zeros(1,512),self.torch.full((1,512),float("nan"))):
            with patch.object(self.engine.model,"embed",return_value=raw),self.assertRaises(FaceInferenceError):
                self.engine.embed(controlled_frame(),CONTROLLED_BOX)

    def test_real_controlled_enrollment_subsequent_recognition_and_reload(self):
        service=ControlledTestEnrollment(self.engine)
        vector=self.engine.embed(controlled_frame(),CONTROLLED_BOX)
        self.assertFalse(service.match(vector).accepted)
        service.commit(service.prepare(("test-actual-001","Generated Actual Model Test")))
        match=service.match(vector)
        self.assertTrue(match.accepted); self.assertEqual(match.person_id,"test-actual-001")
        with TemporaryDirectory(prefix="ui03-actual-generated-") as directory:
            path=Path(directory)/"generated-gallery.json";service._gallery.save(path)
            gallery=FaceGallery.load(path,model_id=service.contract.model_id,model_version=service.contract.model_version,embedding_dim=512)
            self.assertEqual(GalleryMatcher(gallery).match(vector),match)

    @unittest.skipUnless(os.environ.get("FA_EMOTION_CHECKPOINT"), "Set actual renamed Emotion checkpoint")
    def test_both_trained_outputs_preserve_context_but_stub_liveness_never_authorizes(self):
        adapters=TrainedPreviewAdapters(self.path,emotion_checkpoint=os.environ["FA_EMOTION_CHECKPOINT"])
        for scenario in ("Live match","Spoof","Uncertain"):
            adapters.scenario=scenario
            c=PrototypeController();request=c.begin_capture(controlled_frame(),[CONTROLLED_BOX])
            for completion in evaluate_adapters((request,adapters)):
                self.assertEqual(completion.context,request.context);c.receive(completion)
            self.assertFalse(c.state.can_save);self.assertEqual(c.state.liveness.source,"stub")
            self.assertEqual(c.state.recognition is None,scenario!="Live match")
            if scenario=="Live match": self.assertEqual(c.state.emotion.source,"model")


if __name__ == "__main__":
    unittest.main()
