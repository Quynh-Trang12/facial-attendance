"""Candidate-only temporal verification through the existing authorization and SQLite service."""

from contextlib import closing, redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from facial_attendance.attendance.store import AttendanceStore
from facial_attendance.attendance.verification import AttendanceVerification
from facial_attendance.ui.liveness_preview import THRESHOLD
from facial_attendance.ui.trial_profiles import REGISTRATION_RELEASE
from facial_attendance.ui.live_preview import evaluate_adapters
from facial_attendance.ui.webcam_app import main
from facial_attendance.ui.webcam_prototype import Completion, OperationError, PrototypeController
from test_attendance import BOX, FRAME, PERSON
import test_webcam_live as webcam


def prediction(score=.05, **overrides):
    live = score < THRESHOLD
    return {**REGISTRATION_RELEASE, "spoof_score": score, "status": "live" if live else "spoof",
            "allow_attendance": live, **overrides}


class CandidateVerificationTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(TemporaryDirectory())) / "test-attendance.sqlite3"
        self.people = {PERSON.person_id: PERSON.name, "second": "Second fixture"}
        self.now = 10.0
        self.token = (1, 0)
        self.verifier = AttendanceVerification(
            AttendanceStore(self.path), REGISTRATION_RELEASE,
            {"checkpoint_sha256": PERSON.checkpoint_sha256, "preprocessing_version": PERSON.preprocessing_version},
            self.people.get, max_age_seconds=1., clock=lambda: self.now,
        )
        self.service = self.verifier.service
        self.session = self.service.start_session()
        self.verifier.sync_session()

    def capture(self, *, step=.30, box=BOX):
        self.now += step
        self.token = (self.token[0], self.token[1] + 1)
        request = PrototypeController().begin_capture(FRAME, [box])
        self.verifier.expect(self.token, request, self.now)
        return self.token, request

    def sample(self, *, identity=PERSON, result=None, step=.30, box=BOX):
        token, request = self.capture(step=step, box=box)
        return self.verifier.accept(token, request, (Completion(request.context, identity),),
                                    prediction() if result is None else result)

    def five(self, *, identity=PERSON):
        for _ in range(4):
            self.assertFalse(self.sample(identity=identity))
            self.assertFalse(self.service.state.can_record)
        self.assertTrue(self.sample(identity=identity))

    def records(self):
        if not self.path.is_file():
            return []
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT person_id, session_id, status FROM attendance ORDER BY recorded_at").fetchall()

    def test_gate_passing_capture_is_bound_to_the_manual_record_and_a_second_session(self):
        for _ in range(4):
            self.assertFalse(self.sample())
        token, request = self.capture()
        with patch.object(self.service, "complete", wraps=self.service.complete) as complete:
            self.assertTrue(self.verifier.accept(token, request, (Completion(request.context, PERSON),), prediction()))
        recognition, liveness = complete.call_args.args
        self.assertEqual(recognition.context, liveness.context)
        self.assertEqual(recognition.context.capture_id, request.context.capture_id)
        self.assertEqual(recognition.context.face_id, request.context.face_id)
        self.assertEqual(recognition.context.action, "attendance")
        self.assertNotEqual(recognition.context.attempt_id, request.context.attempt_id)
        self.assertEqual(self.service.state.verified_name, PERSON.name)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.service.record().code, "recorded")
        self.assertTrue(self.sample())
        self.assertEqual(self.service.record().code, "already_recorded")
        self.assertEqual(self.records(), [(PERSON.person_id, self.session, "present")])

        self.service.end_session()
        second_session = self.service.start_session()
        self.verifier.sync_session()
        self.five()
        self.assertEqual(self.service.record().code, "recorded")
        self.assertEqual(self.records(), [(PERSON.person_id, self.session, "present"),
                                         (PERSON.person_id, second_session, "present")])

    def test_one_second_and_maximum_gap_are_both_enforced(self):
        for _ in range(5):
            self.assertFalse(self.sample(step=.10))
        self.assertFalse(self.service.state.can_record)
        self.assertFalse(self.sample(step=.51))
        for _ in range(3):
            self.assertFalse(self.sample(step=.30))
        self.assertTrue(self.sample(step=.30))

    def test_spoof_aborts_and_uncertain_resets(self):
        self.five()
        ambiguous = prediction(spoof_score=None, status="uncertain", allow_attendance=False)
        self.assertFalse(self.sample(result=ambiguous))
        self.assertEqual(self.service.record().code, "uncertain")
        for _ in range(4):
            self.assertFalse(self.sample())
        self.assertFalse(self.sample(result=prediction(THRESHOLD)))
        self.assertEqual(self.service.record().code, "spoof")
        self.verifier.invalidate("no_face")
        for _ in range(6):
            self.assertFalse(self.sample())
        self.assertEqual(self.service.record().code, "spoof")
        self.assertEqual(self.records(), [])
        self.verifier.reset()
        self.five()

    def test_unknown_face_loss_and_person_switch_reset_existing_authority(self):
        self.five()
        unknown = replace(PERSON, person_id=None, name=None, score=.1, accepted=False)
        self.assertFalse(self.sample(identity=unknown))
        self.assertEqual(self.service.record().code, "unknown")
        self.five()
        self.verifier.invalidate("no_face")
        self.assertEqual(self.service.record().code, "no_face")
        self.five()
        self.assertFalse(self.sample(box=(80, 80, 100, 100)))
        self.assertFalse(self.service.state.can_record)
        self.verifier.invalidate("multiple_faces")
        self.assertEqual(self.service.record().code, "multiple_faces")
        self.five()
        switched = replace(PERSON, person_id="second", name="Second fixture")
        self.assertFalse(self.sample(identity=switched))
        self.assertFalse(self.service.state.can_record)
        for _ in range(3):
            self.assertFalse(self.sample(identity=switched))
        self.assertTrue(self.sample(identity=switched))
        self.assertEqual(self.service.state.verified_name, "Second fixture")
        self.assertEqual(self.records(), [])

    def test_stale_or_mismatched_callbacks_never_authorize(self):
        for _ in range(4):
            self.sample()
        old_token, old_request = self.capture()
        token, request = self.capture(step=.10)
        self.assertFalse(self.verifier.accept(old_token, old_request,
                                             (Completion(old_request.context, PERSON),), prediction()))
        self.assertFalse(self.service.state.can_record)
        for field in ("capture_id", "face_id", "attempt_id", "action"):
            with self.subTest(field=field):
                wrong = replace(request.context, **{field: "other"})
                self.assertFalse(self.verifier.accept(token, request, (Completion(wrong, PERSON),), prediction()))
                self.assertFalse(self.service.state.can_record)
                token, request = self.capture()
        self.now += 1.1
        self.assertFalse(self.verifier.accept(token, request, (Completion(request.context, PERSON),), prediction()))
        self.assertEqual(self.service.record().code, "stale")
        self.assertEqual(self.records(), [])

    def test_old_pending_callback_cannot_survive_new_session_or_retry(self):
        token, request = self.capture()
        self.service.end_session()
        self.service.start_session()
        self.assertFalse(self.verifier.accept(token, request, (Completion(request.context, PERSON),), prediction()))
        newer_token, newer_request = self.capture()
        self.verifier.reset()
        self.assertFalse(self.verifier.accept(newer_token, newer_request,
                                             (Completion(newer_request.context, PERSON),), prediction()))
        self.assertFalse(self.service.state.can_record)

    def test_new_capture_revokes_previous_auth_while_preserving_gate_streak(self):
        self.five()
        self.verifier.pending_capture()
        self.assertEqual(self.service.record().code, "verification_pending")
        self.assertEqual(self.records(), [])
        self.assertTrue(self.sample())
        self.assertEqual(self.service.record().code, "recorded")

    def test_failed_storage_and_invalid_provenance_never_report_success(self):
        for invalid in (prediction(source="stub"), prediction(model_version="other"),
                        prediction(spoof_score=float("nan"))):
            self.assertFalse(self.sample(result=invalid))
            self.assertEqual(self.service.record().code, "invalid_verification")
        self.assertFalse(self.sample(identity=replace(PERSON, model_version="other")))
        self.assertEqual(self.service.record().code, "invalid_verification")
        self.five()
        self.path.mkdir()
        failed = self.service.record()
        self.assertEqual(failed.code, "save_failed")
        self.assertIsNone(failed.record)
        self.assertFalse(self.service.state.can_record)

    def test_emotion_failure_has_no_vote_and_expired_recording_is_rejected(self):
        for _ in range(4):
            self.sample()
        token, request = self.capture()
        error = OperationError("emotion", "inference_error", "fixture", False)
        self.assertTrue(self.verifier.accept(token, request,
                                            (Completion(request.context, PERSON), Completion(request.context, error)),
                                            prediction()))
        self.now += 1.01
        self.assertEqual(self.service.record().code, "stale")
        self.assertEqual(self.records(), [])


class CandidateAppTests(unittest.TestCase):
    setUp = webcam.LiveAppTests.setUp
    packet = webcam.LiveAppTests.packet

    def test_gui_passes_current_capture_pair_and_invalidates_when_face_lost(self):
        app = self.app
        verifier = Mock()
        app._attendance_verification = verifier
        app._registration_requested = False
        app.liveness_preview = Mock()
        app.liveness_preview.display_text.return_value = ""
        app._tick()
        verifier.pending_capture.assert_called_once_with()
        app.detector.finish([webcam.BOX])
        app._tick()
        token, (request, adapters, emotion) = app.inference.job
        verifier.expect.assert_called_once_with(token, request, self.now)
        batch = evaluate_adapters((request, adapters, emotion), unverified_preview=True)
        raw = prediction()
        app.inference.finish((batch, raw))
        app._tick()
        verifier.accept.assert_called_once_with(token, request, batch, raw, error=None)

        self.now += .45
        self.packet(2)
        app._tick()
        app.detector.finish([])
        app._tick()
        verifier.invalidate.assert_called_with("no_face")

    def test_cli_candidate_requires_explicit_isolated_databases(self):
        flags = ["--local-workspace", "--face-checkpoint", "face.pt", "--emotion-checkpoint", "emotion.pt",
                 "--attendance", "--attendance-verification-test"]
        for arguments, message in (
            ([], "requires --attendance, --local-profile-db and --attendance-db"),
            (["--local-profile-db", "artifacts/local-recognition/profiles.sqlite3",
              "--attendance-db", "artifacts/attendance/attendance.sqlite3"], "must be separate files"),
        ):
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    main([*flags, *arguments])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_gui_delivers_current_worker_failure_to_attendance(self):
        app = self.app
        verifier = Mock()
        app._attendance_verification = verifier
        app._registration_requested = False
        app.liveness_preview = Mock()
        app.liveness_preview.display_text.return_value = ""
        app._tick()
        app.detector.finish([webcam.BOX])
        app._tick()
        token, (request, _, _) = app.inference.job
        error = RuntimeError("fixture worker failure")
        app.inference.finish(error=error)
        app._tick()
        verifier.accept.assert_called_once_with(token, request, None, None, error=error)
