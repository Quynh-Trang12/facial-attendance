"""Local profile persistence and recognition without simulated authorization."""

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from facial_attendance.recognition.evaluation import RecognitionEvaluation, SandboxInference
from facial_attendance.recognition.gallery_matching import GalleryMatcher, empty_gallery
from facial_attendance.ui.diagnostic_snapshot import privacy_safe_snapshot
from facial_attendance.ui.live_preview import LiveController, WorkResult, evaluate_adapters
from facial_attendance.ui.liveness_preview import THRESHOLD
from facial_attendance.ui.local_profile_store import LocalProfileStore
from facial_attendance.ui.recognition_adapter import TrainedRecognitionAdapter
from facial_attendance.ui.registration_pose import FacePose
from facial_attendance.ui.trial_profiles import REGISTRATION_RELEASE, TrialProfiles
from facial_attendance.ui.webcam_app import LiveWebcamApp
from facial_attendance.ui.webcam_prototype import Completion, Liveness, PrototypeController, StubAdapters
from test_recognition_sandbox import BOX, FakeInference, unit


class LocalProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(TemporaryDirectory())
        self.path = Path(self.temp) / "profiles.sqlite3"

    def test_named_profile_can_be_reloaded_and_cleared_without_attendance(self):
        store = LocalProfileStore(self.path)
        session = RecognitionEvaluation()
        session.set_consent(True)
        infer = SandboxInference(FakeInference(), session)
        for sequence in range(3):
            frame = np.full((240, 320, 3), sequence, np.uint8)
            job = session.job("reference", (1, sequence), frame, [BOX], 10., now=10.)
            session.accept(infer(job), current_source=job.source, now=10.05)
        session.commit_reference(display_name="Local name", save_gallery=store.save)
        reloaded = LocalProfileStore(self.path)
        match = GalleryMatcher(reloaded.load()).match(unit(0))
        self.assertEqual((match.person_id, match.name), ("test-001", "Local name"))
        self.assertFalse(session.can_save)
        reloaded.save(empty_gallery())
        self.assertFalse(LocalProfileStore(self.path).load().people)

    def test_parallel_session_cannot_overwrite_another_workspace_change(self):
        first, stale = LocalProfileStore(self.path), LocalProfileStore(self.path)
        gallery = first.load()
        self.assertFalse(stale.load().people)
        gallery.register("person", "Example", unit(0), n_images=3)
        first.save(gallery)
        with self.assertRaisesRegex(ValueError, "changed in another window"):
            stale.save(empty_gallery())
        self.assertEqual(LocalProfileStore(self.path).load().people[0].name, "Example")

    def test_profile_databases_reject_the_other_crop_without_overwriting_profiles(self):
        for tight in (False, True):
            with self.subTest(crop_to_detector_box=tight):
                path = self.path.with_name("box.sqlite3") if tight else self.path
                store = LocalProfileStore(path, crop_to_detector_box=tight)
                gallery = store.load()
                gallery.register("person", "Example", unit(0), n_images=5)
                store.save(gallery)
                incompatible = LocalProfileStore(path, crop_to_detector_box=not tight)
                with self.assertRaisesRegex(ValueError, "different face crop"):
                    incompatible.load()
                with self.assertRaisesRegex(ValueError, "different face crop"):
                    incompatible.save(empty_gallery())
                reopened = LocalProfileStore(path, crop_to_detector_box=tight)
                self.assertEqual(reopened.load().people[0].name, "Example")

    def test_failed_save_keeps_draft_and_does_not_activate_the_profile(self):
        session = RecognitionEvaluation()
        session.set_consent(True)
        session.references.extend([unit(0)] * 3)
        save = Mock(side_effect=ValueError("write failed"))
        with self.assertRaisesRegex(ValueError, "write failed"):
            session.commit_reference(display_name="Example", save_gallery=save)
        self.assertEqual(session.test_ids, ())
        self.assertEqual(len(session.references), 3)
        self.assertTrue(session.consent)


class LocalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.live = LiveController(unverified_preview=True)
        self.frame = np.ones((240, 320, 3), np.uint8)
        self.live.begin_sample((1, 1), 10.)
        self.request = self.live.detected(WorkResult((1, 1), [BOX]), self.frame)

    def test_trained_recognition_runs_without_a_simulated_liveness_result(self):
        gallery = empty_gallery()
        gallery.register("person", "Named profile", unit(0), n_images=3)
        adapters = StubAdapters("Live match")
        adapters.liveness = Mock(side_effect=AssertionError("should not run"))
        adapters.recognition = TrainedRecognitionAdapter(FakeInference(), GalleryMatcher(gallery))
        batch = evaluate_adapters((self.request, adapters), unverified_preview=True)
        adapters.liveness.assert_not_called()
        self.assertTrue(self.live.completed(WorkResult((1, 1), batch)))
        state = self.live.controller.state
        self.assertEqual((state.ui_state, state.recognition.name), ("preview", "Named profile"))
        self.assertIsNone(state.liveness)
        self.assertFalse(state.can_save)

    def test_even_a_live_completion_cannot_authorize_the_local_workspace(self):
        fake = Completion(self.request.context, Liveness("live", "demo", .01))
        self.assertFalse(self.live.completed(WorkResult((1, 1), (fake,))))
        self.assertIsNone(self.live.controller.state.liveness)
        self.assertFalse(self.live.controller.state.can_save)

    def test_diagnostics_can_distinguish_unavailable_from_simulated_liveness(self):
        report = privacy_safe_snapshot({"ui_state": "checking", "can_save": False}, {"liveness_is_simulated": False})
        self.assertFalse(report["authorization"]["liveness_is_simulated"])
        self.assertFalse(report["authorization"]["attendance_available"])


class WidgetValue:
    def __init__(self, master=None, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ReferenceWorker:
    def __init__(self, infer, name):
        self.infer = infer
        self.result = None

    @property
    def busy(self):
        return self.result is not None

    def submit(self, token, job):
        self.result = WorkResult(token, self.infer(job))
        return True

    def take(self):
        result, self.result = self.result, None
        return result


class GuidedRegistrationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.store = LocalProfileStore(Path(directory) / "profiles.sqlite3")
        self.now = 10.
        self.sequence = 0
        self.inference = FakeInference()
        self.preview = SimpleNamespace(last_check=None, last_check_window=None, SCAN_SECONDS=3.0)
        poses = (FacePose(0, .50), FacePose(.20, .50), FacePose(-.20, .50),
                 FacePose(0, .35), FacePose(0, .65))
        self.pose_estimator = Mock(side_effect=lambda request: poses[len(self.profiles.session.references)])
        tk = SimpleNamespace(StringVar=WidgetValue, BooleanVar=WidgetValue, ttk=Mock())
        self.enterContext(patch("facial_attendance.ui.trial_profiles.monotonic", side_effect=lambda: self.now))
        with patch.dict(sys.modules, {"tkinter": tk}), patch(
                "facial_attendance.ui.trial_profiles.SingleWorker", ReferenceWorker):
            self.profiles = TrialProfiles(
                Mock(), SimpleNamespace(face_inference=self.inference), Mock(),
                store=self.store, liveness_provenance=dict(REGISTRATION_RELEASE), liveness_preview=self.preview,
                pose_estimator=self.pose_estimator,
            )
        self.profiles.show()
        self.profiles.consent.set(True)
        self.profiles.change_consent()
        self.profiles.name.set("Vy")

    def observe(self, score, *, uncertain=False, respond=True, readiness=True, **metadata):
        if readiness and self.profiles._attempt_id is not None and self.profiles._last_face is None:
            self.now = max(self.now, self.profiles._attempt_started_at + 3.0)
        self.now += .4
        if readiness:
            self.preview.last_check = SimpleNamespace(decision="live")
            self.preview.last_check_window = (self.now - 3.0, self.now)
        self.sequence += 1
        token = (1, self.sequence)
        frame = np.full((240, 320, 3), self.sequence, np.uint8)
        request = PrototypeController().begin_capture(frame, [BOX])
        self.profiles.observe(token, request, self.now)
        live = score is not None and score < THRESHOLD
        prediction = None if score is None and not uncertain else {
            **REGISTRATION_RELEASE, "status": "uncertain" if uncertain else "live" if live else "spoof",
            "spoof_score": score, "allow_attendance": live, **metadata,
        }
        if respond:
            self.profiles.observe_liveness(token, request, prediction)
        return token, request, prediction

    def capture(self, score):
        self.observe(score)
        self.profiles.capture()
        self.profiles.poll()

    def assert_no_saved_profile(self):
        self.profiles.save()
        self.assertFalse(self.store.path.exists())
        self.assertFalse(self.profiles.session.matcher._gallery.people)
        self.assertIsNone(self.profiles.save_notice)

    def test_five_individually_live_references_survive_reload_and_do_not_record_attendance(self):
        directions = (None, (-1, 0), (1, 0), (0, -1), (0, 1))
        for count, score in enumerate((.2, .21, .44, .39)):
            view = self.profiles.guidance()
            self.assertEqual((view.captured, view.total, view.pose.direction), (count, 5, directions[count]))
            self.capture(score)
            self.assertEqual(self.profiles.guidance().feedback, "accepted")
        self.assertTrue(self.profiles.status.get().startswith("4/5"))
        self.assertEqual(self.profiles.guidance().pose.direction, directions[-1])
        self.assert_no_saved_profile()
        self.capture(.2)
        self.assertTrue(self.profiles.status.get().startswith("5/5 · Registration ready"))
        self.assertEqual(self.profiles.guidance().captured, 5)
        self.assertIsNone(self.profiles.guidance().pose)
        self.profiles.save_button.configure.assert_called_with(state="normal")
        self.assertEqual(len({context.capture_id for context, _ in self.profiles._accepted_liveness}), 5)
        self.assert_no_attendance_file()
        self.profiles.save()
        self.assertFalse(self.profiles.active)
        self.assertIsNone(self.profiles.guidance())
        self.assertIn("Registration saved", self.profiles.save_notice)
        reloaded = LocalProfileStore(self.store.path).load()
        self.assertEqual(reloaded.people[0].n_images, 5)
        self.assertEqual(GalleryMatcher(reloaded).match(unit(0)).name, "Vy")
        self.assert_no_attendance_file()

    def assert_no_attendance_file(self):
        self.assertEqual(list(self.store.path.parent.glob("*attendance*")), [])

    def test_wrong_pose_and_missing_landmarks_keep_progress_and_require_a_fresh_capture(self):
        self.capture(.2)
        attempt = self.profiles._attempt_id
        initial_evidence = tuple(self.profiles._accepted_liveness)
        self.pose_estimator.side_effect = [FacePose(-.20, .50), None, FacePose(.20, .50)]
        for message in ("Face direction does not match", "Could not check face direction"):
            with self.subTest(message=message):
                _, request, _ = self.observe(.2)
                self.profiles.capture()
                self.profiles.poll()
                measured = self.pose_estimator.call_args.args[0]
                self.assertIs(measured.frame, request.frame)
                self.assertEqual(measured.bbox, request.bbox)
                self.assertEqual(measured.context.capture_id, request.context.capture_id)
                self.assertEqual(measured.context.face_id, request.context.face_id)
                self.assertEqual(self.profiles._attempt_id, attempt)
                self.assertEqual(tuple(self.profiles._accepted_liveness), initial_evidence)
                self.assertEqual(self.profiles.guidance().captured, 1)
                self.assertEqual(self.profiles.guidance().pose, TrialProfiles.POSES[1])
                self.assertEqual(self.profiles.guidance().feedback, "retry")
                self.assertIn(message, self.profiles.status.get())
                self.assertFalse(self.profiles._can_capture(self.now))
        self.capture(.2)
        self.assertEqual(self.profiles.guidance().captured, 2)

    def test_wrong_first_pose_does_not_calibrate_and_spoof_aborts_before_pose_check(self):
        self.pose_estimator.side_effect = [FacePose(.20, .50), FacePose(0, .62)]
        self.capture(.2)
        self.assertEqual(self.profiles.guidance().captured, 0)
        self.assertIsNone(self.profiles._neutral_pose)
        self.capture(.2)
        self.assertEqual(self.profiles._neutral_pose, FacePose(0, .62))

        self.capture(.8)
        self.assertEqual(self.pose_estimator.call_count, 2)
        self.assertEqual(self.profiles.guidance().feedback, "spoof")
        self.assertEqual(self.profiles.guidance().captured, 0)
        self.assertIsNone(self.profiles._neutral_pose)

    def test_spoof_as_first_or_fifth_sample_aborts_and_discards_all_references(self):
        for accepted in (0, 4):
            with self.subTest(accepted=accepted):
                for _ in range(accepted):
                    self.capture(.2)
                aborted = self.profiles._attempt_id
                self.capture(.8)
                self.assertFalse(self.profiles.session.references)
                self.assertTrue(self.profiles.status.get().startswith("0/5 · Selected photo was flagged as spoof. Registration aborted."))
                self.profiles.save_button.configure.assert_called_with(state="disabled")
                self.assert_no_saved_profile()
                self.capture(.2)
                self.assertFalse(self.profiles.session.references)
                self.profiles.discard()
                self.assertNotEqual(self.profiles._attempt_id, aborted)
                self.capture(.2)
                self.assertEqual(len(self.profiles.session.references), 1)
                self.profiles.discard()

    def test_capture_requires_fresh_live_scan_from_current_attempt(self):
        self.observe(.2, readiness=False)
        self.profiles.capture()
        self.assertIsNone(self.profiles._capture_request)
        self.assertFalse(self.profiles.worker.busy)
        self.preview.last_check = SimpleNamespace(decision="live")
        self.preview.last_check_window = (self.now - 3.0, self.now)
        self.assertFalse(self.profiles._can_capture(self.now))
        for _ in range(8):
            self.observe(.2, readiness=False)
        self.preview.last_check_window = (self.now - 3.0, self.now)
        self.preview.last_check = SimpleNamespace(decision="spoof")
        self.assertFalse(self.profiles._can_capture(self.now))
        self.preview.last_check = SimpleNamespace(decision="live")
        self.assertTrue(self.profiles._can_capture(self.now))
        self.profiles.discard()
        self.observe(.2, readiness=False)
        self.assertFalse(self.profiles._can_capture(self.now))

    def test_spoof_before_capture_or_during_another_photos_embedding_preserves_attempt(self):
        self.capture(.2)
        attempt = self.profiles._attempt_id
        self.observe(.8)
        self.assertEqual(self.profiles._attempt_id, attempt)
        self.assertTrue(self.profiles._can_capture(self.now))
        self.assertEqual(len(self.profiles.session.references), 1)

        self.observe(.2)
        self.profiles.capture()
        self.observe(.8)
        self.profiles.poll()
        self.assertEqual(self.profiles._attempt_id, attempt)
        self.assertEqual(len(self.profiles.session.references), 2)
        self.assert_no_saved_profile()

    def test_expired_unscored_selected_photo_does_not_submit_another_frame(self):
        old = self.observe(.2, respond=False)
        self.profiles.capture()
        self.now += self.profiles.session.MAX_SAMPLE_AGE + .01
        self.profiles.poll()
        self.assertIn("Selected photo expired", self.profiles.status.get())
        self.profiles.observe_liveness(*old)
        self.assertFalse(self.profiles.worker.busy)
        self.assertFalse(self.profiles.session.references)

    def test_selected_spoof_aborts_pending_capture_and_stale_live_cannot_restore_it(self):
        old = self.observe(.8, respond=False)
        self.profiles.capture()
        self.assertIn("Checking liveness", self.profiles.status.get())
        self.assertFalse(self.profiles.worker.busy)
        self.profiles.observe_liveness(*old)
        self.assertIsNone(self.profiles._attempt_id)
        guidance = self.profiles.guidance()
        self.assertEqual((guidance.feedback, guidance.captured, guidance.pose), ("spoof", 0, None))
        self.profiles.change_consent()
        self.assertIsNone(self.profiles._attempt_id)
        self.profiles.poll()
        self.profiles.observe_liveness(*old)
        self.assertTrue(self.profiles.status.get().startswith("0/5 · Selected photo was flagged as spoof. Registration aborted."))
        self.profiles.discard()
        self.assertEqual(self.profiles.guidance().pose, TrialProfiles.POSES[0])
        self.assertEqual(self.profiles.guidance().feedback, "guidance")
        self.profiles.observe_liveness(*old)
        self.assert_no_saved_profile()
        self.assertFalse(self.profiles.session.references)
        self.profiles.cancel()
        self.assertIsNone(self.profiles.guidance())

    def test_uncertain_requires_fresh_capture_preserving_live_references(self):
        self.capture(.2)
        pose = self.profiles.guidance().pose
        token, request, prediction = self.observe(None, uncertain=True)
        self.profiles.capture()
        self.assertTrue(self.profiles.status.get().startswith("1/5 · Uncertain"))
        self.assertEqual(self.profiles.guidance().pose, pose)
        self.assertEqual(self.profiles.guidance().feedback, "retry")
        self.profiles.observe_liveness(token, request, {**prediction, "status": "live",
                                                        "spoof_score": .2, "allow_attendance": True})
        self.profiles.capture()
        self.profiles.poll()
        self.assertEqual(len(self.profiles.session.references), 1)
        self.capture(.2)
        self.assertEqual(len(self.profiles.session.references), 2)
        self.assert_no_saved_profile()

    def test_exact_capture_and_face_match_and_repeated_result_cannot_be_counted_twice(self):
        token, request, prediction = self.observe(.2, respond=False)
        for field in ("capture_id", "face_id"):
            with self.subTest(field=field):
                mismatched = replace(request, context=replace(request.context, **{field: "other"}))
                self.profiles.observe_liveness(token, mismatched, prediction)
                self.profiles.capture()
                self.assertFalse(self.profiles.worker.busy)
        self.profiles.observe_liveness(token, request, prediction)
        self.profiles.capture()
        job = self.profiles.session.pending
        self.assertEqual(job.request.context.capture_id, request.context.capture_id)
        self.assertEqual(job.request.context.face_id, request.context.face_id)
        self.assertEqual(job.request.context.action, "registration")
        self.assertEqual(job.request.context.attempt_id, self.profiles._attempt_id)
        result = self.profiles.worker.result
        self.profiles.poll()
        self.profiles.observe_liveness(token, request, prediction)
        self.profiles.capture()
        self.profiles.worker.result = result
        self.profiles.poll()
        self.assertEqual(len(self.profiles.session.references), 1)

    def test_invalid_model_metadata_stub_and_errors_block_registration(self):
        for metadata in ({"source": "stub"}, {"model_id": None}, {"model_version": None}, {"precision": None}):
            with self.subTest(metadata=metadata):
                self.capture(.2)
                self.observe(.2, **metadata)
                self.assertFalse(self.profiles.session.references)
                self.assertIsNone(self.profiles._attempt_id)
                self.assert_no_saved_profile()
                self.profiles.discard()
        self.capture(.2)
        self.observe(None)
        self.assertTrue(self.profiles.status.get().startswith("0/5 · Liveness verification failed"))
        self.assert_no_saved_profile()
        self.profiles.discard()
        self.observe(.2)
        self.profiles.capture()
        self.profiles.worker.result = WorkResult(self.profiles.session.pending.source, error=RuntimeError("test"))
        self.profiles.poll()
        self.assertIn("Reference inference failed", self.profiles.status.get())
        self.assert_no_saved_profile()

    def test_final_confirmation_rechecks_policy_and_all_evidence(self):
        for _ in range(5):
            self.capture(.2)
        self.assertTrue(self.profiles._can_save(self.now))
        context, accepted = self.profiles._accepted_liveness[-1]
        self.profiles._accepted_liveness[-1] = (
            context, replace(accepted, result=replace(accepted.result, decision_policy_version=None)))
        self.assert_no_saved_profile()
        self.profiles._accepted_liveness[-1] = (
            context, replace(accepted, context=replace(context, face_id="different face")))
        self.assert_no_saved_profile()
        self.profiles._accepted_liveness[-1] = (context, accepted)
        self.profiles._liveness_provenance.pop("checkpoint_sha256")
        self.assert_no_saved_profile()

    def test_threshold_equality_is_spoof(self):
        self.capture(THRESHOLD)
        self.assertTrue(self.profiles.status.get().startswith("0/5 · Selected photo was flagged as spoof. Registration aborted."))
        self.assert_no_saved_profile()

    def test_failed_profile_write_does_not_publish_identity_or_success(self):
        for _ in range(5):
            self.capture(.2)
        self.store.path.mkdir()
        self.profiles.save()
        self.assertIn("Save failed", self.profiles.status.get())
        self.assertFalse(self.profiles.session.matcher._gallery.people)
        self.assertIsNone(self.profiles.save_notice)
        self.profiles.on_gallery_change.assert_not_called()

    def test_camera_restart_and_multiple_faces_invalidate_ready_registration(self):
        for failure in ("restart", "multiple_faces"):
            with self.subTest(failure=failure):
                for _ in range(5):
                    self.capture(.2)
                previous = self.profiles._attempt_id
                if failure == "restart":
                    self.profiles.invalidate_source()
                else:
                    self.profiles.observe((1, self.sequence + 1), None, self.now + .01)
                self.assertNotEqual(self.profiles._attempt_id, previous)
                self.assertFalse(self.profiles.session.references)
                self.assert_no_saved_profile()

    def test_face_loss_and_replacement_discard_photos_and_late_results(self):
        self.capture(.2)
        self.observe(.2)
        self.profiles.capture()
        self.profiles.observe((1, self.sequence + 1), None, self.now + .01)
        self.profiles.poll()
        self.assertFalse(self.profiles.session.references)
        self.capture(.2)
        self.inference.vector = unit(1)
        self.capture(.2)
        self.assertFalse(self.profiles.session.references)
        self.assertTrue(self.profiles.status.get().startswith("0/5"))

    def test_duplicate_matches_are_named_without_replacing_the_profile(self):
        gallery = empty_gallery()
        gallery.register("existing", "Existing person", unit(0), n_images=3)
        self.store.save(gallery)
        self.profiles.session.matcher._gallery = gallery
        for _ in range(5):
            self.capture(.2)
        self.profiles.save()
        self.assertIn("Existing person", self.profiles.status.get())
        self.assertEqual(len(LocalProfileStore(self.store.path).load().people), 1)

    def test_registration_opens_only_for_fresh_unknown_model_result(self):
        app = LiveWebcamApp.__new__(LiveWebcamApp)
        app.live = LiveController(unverified_preview=True)
        app.controller = app.live.controller
        app._local_workspace = True
        app._developer_mode = True
        app._trial_profiles = self.profiles
        self.profiles.cancel()
        token = (1, 1)
        app.live.begin_sample(token, self.now)
        request = app.live.detected(WorkResult(token, [BOX]), np.ones((240, 320, 3), np.uint8))
        self.assertFalse(app._can_start_local_registration(self.now))
        adapter = TrainedRecognitionAdapter(self.inference, GalleryMatcher(empty_gallery()))
        app.live.completed(WorkResult(token, (adapter(request),)))
        self.assertTrue(app._can_start_local_registration(self.now))
        app._developer_mode = False
        self.assertFalse(app._can_start_local_registration(self.now))
        app._developer_mode = True
        self.assertFalse(app._can_start_local_registration(self.now + 1))
        app.live.boxes = [BOX, BOX]
        self.assertFalse(app._can_start_local_registration(self.now))

    def test_registration_button_can_request_or_cancel_while_waiting_for_a_face(self):
        app = LiveWebcamApp.__new__(LiveWebcamApp)
        app._local_workspace = True
        app._trial_profiles = self.profiles
        app._can_start_local_registration = Mock(return_value=False)
        app._render = Mock()
        self.profiles.cancel()

        app._developer_mode = False
        app._show_registration()
        self.assertFalse(getattr(app, "_registration_requested", False))
        app._developer_mode = True
        app._show_registration()
        self.assertTrue(app._registration_requested)
        self.assertFalse(self.profiles.active)
        app._show_registration()
        self.assertFalse(app._registration_requested)
