"""Deterministic contract and SQLite tests. No physical camera or genuine evidence."""

import os
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from facial_attendance.attendance.service import AttendanceService, VerificationPolicy
from facial_attendance.attendance.store import AttendanceStore
from facial_attendance.recognition.contracts import (
    SELECTED_MODEL_ID,
    SELECTED_MODEL_VERSION,
)
from facial_attendance.recognition.gallery_matching import empty_gallery
from facial_attendance.ui.local_profile_store import LocalProfileStore
from facial_attendance.ui.webcam_app import main
from facial_attendance.ui.webcam_prototype import (
    Completion,
    Emotion,
    Liveness,
    OperationError,
    Recognition,
)

POLICY = VerificationPolicy(
    "fixture-live",
    "fixture-rgb",
    "fixture-capture-v1",
    "a" * 64,
    "fixture-face-rgb",
    1.0,
)
PERSON = Recognition(
    "fixture-person",
    "Attendance Test",
    0.82,
    True,
    SELECTED_MODEL_ID,
    SELECTED_MODEL_VERSION,
    "model",
    POLICY.recognition_checkpoint,
    POLICY.recognition_preprocessing,
)
LIVE = Liveness(
    "live",
    "accepted_live",
    0.07,
    POLICY.liveness_model,
    POLICY.liveness_preprocessing,
    POLICY.liveness_decision,
    "model",
)
BOX = (0, 0, 80, 80)
FRAME = np.zeros((100, 100, 3), np.uint8)


class AttendanceTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "attendance.sqlite3"
        self.people = {PERSON.person_id: PERSON.name}
        self.now = 10.0
        self.service = AttendanceService(
            AttendanceStore(self.path),
            policy=POLICY,
            registered_name=self.people.get,
            clock=lambda: self.now,
        )
        self.session = self.service.start_session()

    def request(self):
        return self.service.begin_capture(FRAME, [BOX], self.now)

    def complete(self, request=None, *, identity=PERSON, liveness=LIVE):
        request = request or self.request()
        return self.service.complete(
            Completion(request.context, identity), Completion(request.context, liveness)
        )

    def rows(self):
        if not self.path.is_file():
            return []
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute(
                "SELECT * FROM attendance ORDER BY recorded_at"
            ).fetchall()

    def assert_blocked(self, code):
        self.assertFalse(self.service.state.can_record)
        self.assertIsNone(self.service.state.verified_name)
        self.assertEqual(self.service.record().code, code)
        self.assertEqual(self.rows(), [])

    def test_known_live_records_only_after_confirmation(self):
        self.assertTrue(self.complete())
        self.assertEqual(self.service.state.verified_name, PERSON.name)
        self.assertEqual(self.rows(), [])
        first = self.service.record()
        self.assertEqual(first.code, "recorded")
        record = first.record
        self.assertEqual(
            self.rows(),
            [
                (
                    record.attendance_id,
                    PERSON.person_id,
                    PERSON.name,
                    self.session,
                    record.recorded_at,
                    "present",
                )
            ],
        )
        self.assertEqual(
            datetime.fromisoformat(record.recorded_at).utcoffset(), timedelta(0)
        )
        self.assertIsNone(self.service.state.verified_name)

    def test_spoof_aborts_until_explicit_retry(self):
        request = self.request()
        self.assertFalse(
            self.complete(
                request, liveness=replace(LIVE, status="spoof", reason="detected_spoof")
            )
        )
        self.assertFalse(self.complete(request))
        self.assertIsNone(self.request())
        self.assert_blocked("spoof")
        self.service.reset()
        self.assertTrue(self.complete())

    def test_unknown_with_live_and_stale_gallery_match(self):
        unknown = replace(PERSON, person_id=None, name=None, score=0.1, accepted=False)
        self.assertFalse(self.complete(identity=unknown))
        self.assert_blocked("unknown")
        self.assertTrue(self.complete())
        self.people.clear()
        self.assertEqual(self.service.record().code, "unknown")
        self.assert_blocked("unknown")

    def test_uncertain_and_stub_liveness(self):
        self.assertFalse(
            self.complete(liveness=replace(LIVE, status="uncertain", p_spoof=None))
        )
        self.assert_blocked("uncertain")
        self.assertFalse(self.complete(liveness=replace(LIVE, source="stub")))
        self.assert_blocked("invalid_verification")

    def test_empty_multiple_or_invalid_regions_revoke_ready_state(self):
        for boxes, code in (
            ([], "no_face"),
            ([BOX, BOX], "multiple_faces"),
            ([(0, 0, 200, 200)], "capture_error"),
        ):
            with self.subTest(code=code):
                self.assertTrue(self.complete())
                self.assertIsNone(self.service.begin_capture(FRAME, boxes, self.now))
                self.assert_blocked(code)

    def test_camera_and_inference_errors_revoke_authority(self):
        for error, code in (
            (
                OperationError("camera", "camera_unavailable", "failed", True),
                "capture_error",
            ),
            (
                OperationError("liveness", "inference_error", "failed", True),
                "inference_error",
            ),
            (
                OperationError("liveness", "model_unavailable", "failed", True),
                "model_unavailable",
            ),
        ):
            with self.subTest(code=code):
                self.assertFalse(self.complete(liveness=error))
                self.assert_blocked(code)
        self.assertFalse(
            self.complete(
                identity=OperationError(
                    "recognition", "inference_error", "failed", True
                )
            )
        )
        self.assert_blocked("inference_error")

    def test_mismatched_capture_face_attempt_action_and_contract_are_rejected(self):
        request = self.request()
        for field, value in (
            ("capture_id", "other"),
            ("face_id", "other"),
            ("attempt_id", "other"),
            ("action", "registration"),
            ("contract_version", "other"),
        ):
            with self.subTest(field=field):
                other = replace(request.context, **{field: value})
                self.assertFalse(
                    self.service.complete(
                        Completion(request.context, PERSON), Completion(other, LIVE)
                    )
                )
                self.assertFalse(
                    self.service.complete(
                        Completion(other, PERSON), Completion(request.context, LIVE)
                    )
                )
                self.assert_blocked("verification_pending")

    def test_expired_replaced_and_aborted_completions(self):
        old = self.request()
        current = self.request()
        self.assertFalse(self.complete(old))
        self.assertTrue(self.complete(current))
        self.now += 1.01
        self.assert_blocked("stale")
        next_request = self.request()
        self.service.reset()
        self.assertFalse(self.complete(next_request))
        self.assert_blocked("verification_pending")
        expired = self.request()
        self.now += 1.01
        self.assertFalse(self.complete(expired))
        self.assert_blocked("stale")

    def test_invalid_scores_metadata_and_malformed_responses(self):
        for value in (None, float("nan"), float("inf"), -1, 1.01, True):
            with self.subTest(score=value):
                self.assertFalse(self.complete(liveness=replace(LIVE, p_spoof=value)))
                self.assert_blocked("invalid_verification")
        for invalid in (
            replace(LIVE, model_version=None),
            replace(LIVE, preprocessing_version="other"),
            replace(LIVE, decision_policy_version="other"),
            replace(LIVE, reason="inference_error"),
            {},
        ):
            self.assertFalse(self.complete(liveness=invalid))
            self.assert_blocked("invalid_verification")
        for invalid in (
            replace(PERSON, model_version="unknown"),
            replace(PERSON, source="stub"),
            replace(PERSON, checkpoint_sha256=None),
            replace(PERSON, preprocessing_version=None),
            replace(PERSON, score=0.02),
            replace(PERSON, score=float("nan")),
        ):
            self.assertFalse(self.complete(identity=invalid))
            self.assert_blocked("invalid_verification")
        request = self.request()
        self.assertFalse(
            self.service.complete(Completion(request.context, PERSON), None)
        )
        self.assert_blocked("invalid_verification")

    def test_duplicate_confirmation_and_database_uniqueness(self):
        self.assertTrue(self.complete())
        self.assertEqual(self.service.record().code, "recorded")
        self.assertEqual(self.service.record().code, "already_recorded")
        self.assertTrue(self.complete())
        self.assertEqual(self.service.record().code, "already_recorded")
        self.assertEqual(len(self.rows()), 1)
        row = self.rows()[0]
        with closing(sqlite3.connect(self.path)) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO attendance VALUES (?, ?, ?, ?, ?, ?)",
                    ("duplicate", *row[1:]),
                )

    def test_session_exit_invalidates_pending_and_same_person_can_record_in_new_session(
        self,
    ):
        self.assertTrue(self.complete())
        self.assertEqual(self.service.record().code, "recorded")
        request = self.request()
        self.assertEqual(self.service.start_session(), self.session)
        self.service.end_session()
        self.assertEqual(self.service.record().code, "session_inactive")
        self.assertIsNone(self.request())
        new_session = self.service.start_session()
        self.assertNotEqual(new_session, self.session)
        self.assertFalse(self.complete(request))
        self.assertTrue(self.complete())
        self.assertEqual(self.service.record().code, "recorded")
        self.assertEqual([row[3] for row in self.rows()], [self.session, new_session])

    def test_failed_database_write_clears_ready_without_success(self):
        self.assertTrue(self.complete())
        self.path.mkdir()
        result = self.service.record()
        self.assertIsNone(result.record)
        self.assert_blocked("save_failed")

    def test_does_not_use_a_biometric_database_for_attendance(self):
        profiles = LocalProfileStore(self.path.with_name("profiles.sqlite3"))
        gallery = empty_gallery()
        gallery.register(PERSON.person_id, PERSON.name, [1.0] + [0.0] * 511, n_images=5)
        profiles.save(gallery)
        original = profiles.path.read_bytes()
        service = AttendanceService(
            AttendanceStore(profiles.path),
            policy=POLICY,
            registered_name=self.people.get,
            clock=lambda: self.now,
        )
        service.start_session()
        request = service.begin_capture(FRAME, [BOX], self.now)
        self.assertTrue(
            service.complete(
                Completion(request.context, PERSON), Completion(request.context, LIVE)
            )
        )
        self.assertEqual(service.record().code, "save_failed")
        self.assertEqual(profiles.path.read_bytes(), original)
        self.assertEqual(profiles.load().people[0].person_id, PERSON.person_id)
        self.assertEqual(self.rows(), [])

    def test_profile_save_is_separate_from_attendance(self):
        profiles = LocalProfileStore(self.path.with_name("profiles.sqlite3"))
        gallery = empty_gallery()
        gallery.register(PERSON.person_id, PERSON.name, [1.0] + [0.0] * 511, n_images=5)
        profiles.save(gallery)
        self.assertEqual(profiles.load().people[0].name, PERSON.name)
        self.assertFalse(self.path.exists())
        self.assert_blocked("verification_pending")

    def test_emotion_completion_cannot_change_attendance(self):
        request = self.request()
        expression = Completion(request.context, Emotion(1, "Neutral", 0.8))
        self.assertFalse(
            self.service.complete(expression, Completion(request.context, LIVE))
        )
        self.assert_blocked("invalid_verification")
        self.assertTrue(self.complete())  # No emotion result is needed.
        self.assertEqual(self.service.record().code, "recorded")

    def test_unconfigured_verification_cannot_be_promoted_by_real_looking_completion(
        self,
    ):
        service = AttendanceService(AttendanceStore(self.path))
        service.start_session()
        external = self.request()
        self.assertIsNone(service.begin_capture(FRAME, [BOX], self.now))
        self.assertFalse(
            service.complete(
                Completion(external.context, PERSON), Completion(external.context, LIVE)
            )
        )
        self.assertEqual(service.record().code, "model_unavailable")
        self.assertFalse(self.path.exists())

    def test_inactive_session_cannot_use_old_authorization(self):
        self.assertTrue(self.complete())
        self.service.end_session()
        self.assert_blocked("session_inactive")


class AttendanceCliTests(unittest.TestCase):
    def test_cli_rejects_attendance_outside_local_workspace(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
            main(["--attendance"])
        self.assertEqual(failure.exception.code, 2)

    def test_cli_rejects_attendance_database_without_opt_in(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
            main(["--attendance-db", "fixture.sqlite3"])
        self.assertEqual(failure.exception.code, 2)


@unittest.skipUnless(
    os.getenv("FA_NATIVE_ATTENDANCE_TESTS") == "1",
    "Requires a native Tk desktop session",
)
class AttendanceWidgetTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from tkinter import ttk

        from facial_attendance.ui.dashboard import Dashboard, apply_theme

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Native Tk unavailable: {error}")
        self.addCleanup(root.destroy)
        root.withdraw()
        apply_theme(root)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "attendance.sqlite3"
        self.root = root
        self.dashboard_type = Dashboard
        self.parent = ttk.Frame(root)

    def dashboard(
        self,
        service=None,
        *,
        retry_attendance=None,
        retrying_after_spoof=None,
        automatic_record=False,
    ):
        return self.dashboard_type(
            self.parent,
            restart_camera=lambda: None,
            switch_mode=lambda: None,
            trained_liveness=True,
            attendance=service,
            retry_attendance=retry_attendance,
            retrying_after_spoof=retrying_after_spoof,
            automatic_record=automatic_record,
        )

    def show_inspector(self):
        self.root.geometry("820x390")
        self.parent.pack(fill="both", expand=True)
        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, minsize=270)
        self.parent.rowconfigure(2, weight=1)
        self.root.deiconify()
        self.root.update()

    def test_inspector_scrolls_in_small_steps_and_keeps_position_on_content_resize(
        self,
    ):
        from tkinter import ttk
        from types import SimpleNamespace

        dashboard = self.dashboard()
        filler = ttk.Frame(dashboard.results, height=300)
        filler.pack(fill="x")
        self.show_inspector()
        area = dashboard._scroll_area
        self.assertTrue(dashboard._scrollbar.winfo_ismapped())
        aqua = self.root.tk.call("tk", "windowingsystem") == "aqua"
        event = SimpleNamespace(widget=dashboard.results, delta=-1 if aqua else -120)
        self.assertEqual(dashboard._scroll_inspector(event), "break")
        top = area.canvasy(0)
        self.assertEqual(top, 8 if aqua else 36)

        filler.configure(height=420)
        self.root.update()
        self.assertEqual(area.canvasy(0), top)
        area.yview_moveto(1)
        filler.configure(height=300)
        self.root.update()
        self.assertEqual(area.yview()[1], 1.0)

    def test_inspector_keeps_registration_button_in_place_as_messages_wrap(self):
        from tkinter import ttk

        service = AttendanceService(
            AttendanceStore(self.path),
            policy=POLICY,
            registered_name={PERSON.person_id: PERSON.name}.get,
        )
        dashboard = self.dashboard(service, automatic_record=True)
        panel = dashboard.attendance_view
        prompt_slot, prompt = dashboard.message_slot(
            dashboard.results, text="Registration saved."
        )
        dashboard.reserve_message_lines(prompt_slot, 3)
        prompt_slot.pack(fill="x")
        button = ttk.Button(dashboard.results, text="Register new person")
        button.pack(anchor="w")
        self.show_inspector()
        panel.session.invoke()
        self.root.update()
        button_y = button.winfo_y()
        prompt_height = prompt.winfo_reqheight()

        service.invalidate_capture("spoof")
        panel.refresh()
        prompt.configure(
            text="Waiting for one unregistered face. Ask the new person to face the camera."
        )
        self.root.update()
        self.assertIn("Spoof detected", panel.feedback.cget("text"))
        self.assertGreater(prompt.winfo_reqheight(), prompt_height)
        self.assertEqual(button.winfo_y(), button_y)

    def test_opt_in_panel_preserves_preview_and_cannot_write_without_adapter(self):
        preview = self.dashboard()
        labels = tuple(
            child.cget("text")
            for child in preview.results.winfo_children()
            if child.winfo_class() == "TLabel"
        )
        self.assertIn("Attendance saving disabled", labels)
        service = AttendanceService(AttendanceStore(self.path))
        panel = self.dashboard(service).attendance_view
        self.assertIsNone(panel.retry_button)
        self.assertTrue(panel.confirm.instate(("disabled",)))
        panel.session.invoke()
        self.assertIsNotNone(service.session_id)
        self.assertIn("Model unavailable", panel.feedback.cget("text"))
        self.assertTrue(panel.confirm.instate(("disabled",)))
        panel.confirm.invoke()
        self.assertFalse(self.path.exists())
        panel.session.invoke()
        self.assertIsNone(service.session_id)
        self.assertIn("inactive", panel.session_status.cget("text"))

    def test_retry_verification_starts_fresh_in_same_session_despite_live_preview(self):
        from facial_attendance.attendance.verification import AttendanceVerification
        from facial_attendance.ui.liveness_preview import LastLivenessCheck
        from facial_attendance.ui.trial_profiles import REGISTRATION_RELEASE
        from facial_attendance.ui.webcam_prototype import Envelope, PrototypeController

        now = 10.0
        verifier = AttendanceVerification(
            AttendanceStore(self.path),
            REGISTRATION_RELEASE,
            {
                "checkpoint_sha256": PERSON.checkpoint_sha256,
                "preprocessing_version": PERSON.preprocessing_version,
            },
            {PERSON.person_id: PERSON.name}.get,
            max_age_seconds=1.0,
            clock=lambda: now,
        )
        dashboard = self.dashboard(verifier.service, retry_attendance=verifier.reset)
        panel = dashboard.attendance_view
        panel.session.invoke()
        verifier.sync_session()
        session_id = verifier.service.session_id
        verifier.invalidate("spoof")
        dashboard.update_preview(
            Envelope(ui_state="preview"),
            None,
            None,
            detecting=False,
            scan=LastLivenessCheck("live", 0.1, 7, "12:00:00"),
            trained_liveness=True,
            trained_emotion=False,
            score_error=False,
        )
        self.assertEqual(panel.retry_button.winfo_manager(), "pack")
        self.assertIn("Live preview", panel.feedback.cget("text"))
        self.assertTrue(panel.confirm.instate(("disabled",)))

        panel.retry_button.invoke()
        self.assertEqual(verifier.service.session_id, session_id)
        self.assertEqual(verifier.service.state.code, "verification_pending")
        self.assertEqual(panel.retry_button.winfo_manager(), "")
        self.assertTrue(panel.confirm.instate(("disabled",)))
        self.assertFalse(self.path.exists())

        prediction = {
            **REGISTRATION_RELEASE,
            "status": "live",
            "spoof_score": 0.1,
            "allow_attendance": True,
        }
        for sequence in range(1, 6):
            now += 0.3
            request = PrototypeController().begin_capture(FRAME, [BOX])
            token = (1, sequence)
            verifier.expect(token, request, now)
            verifier.accept(
                token, request, (Completion(request.context, PERSON),), prediction
            )
            panel.refresh()
            self.assertEqual(panel.confirm.instate(("!disabled",)), sequence == 5)
        self.assertFalse(self.path.exists())

    def test_widget_shows_spoof_during_auto_retry_and_manual_button_after_second_spoof(
        self,
    ):
        from facial_attendance.attendance.verification import AttendanceVerification
        from facial_attendance.ui.trial_profiles import REGISTRATION_RELEASE
        from facial_attendance.ui.webcam_prototype import PrototypeController

        now = 10.0
        verifier = AttendanceVerification(
            AttendanceStore(self.path),
            REGISTRATION_RELEASE,
            {
                "checkpoint_sha256": PERSON.checkpoint_sha256,
                "preprocessing_version": PERSON.preprocessing_version,
            },
            {PERSON.person_id: PERSON.name}.get,
            max_age_seconds=1.0,
            automatic_retry=True,
            clock=lambda: now,
        )
        panel = self.dashboard(
            verifier.service,
            retry_attendance=verifier.reset,
            retrying_after_spoof=verifier.retrying_after_spoof,
        ).attendance_view
        panel.session.invoke()
        verifier.sync_session()
        spoof = {
            **REGISTRATION_RELEASE,
            "status": "spoof",
            "spoof_score": 0.8,
            "allow_attendance": False,
        }
        for sequence in range(1, 3):
            now += 0.3
            request = PrototypeController().begin_capture(FRAME, [BOX])
            token = (1, sequence)
            verifier.expect(token, request, now)
            verifier.accept(
                token, request, (Completion(request.context, PERSON),), spoof
            )
            panel.refresh(preview_live=True)
            self.assertIn("spoof", panel.feedback.cget("text").lower())
            self.assertTrue(panel.confirm.instate(("disabled",)))
            self.assertEqual(
                panel.retry_button.winfo_manager(), "" if sequence == 1 else "pack"
            )

        panel.retry_button.invoke()
        self.assertEqual(verifier.service.state.code, "verification_pending")
        self.assertFalse(self.path.exists())

    def test_auto_record_panel_stays_stable_between_current_samples(self):
        service = AttendanceService(
            AttendanceStore(self.path),
            policy=POLICY,
            registered_name={PERSON.person_id: PERSON.name}.get,
            clock=lambda: 10.0,
        )
        panel = self.dashboard(service, automatic_record=True).attendance_view
        panel.session.invoke()
        pending = panel.feedback.cget("text")
        request = service.begin_capture(FRAME, [BOX], 10.0)
        service.complete(
            Completion(request.context, PERSON), Completion(request.context, LIVE)
        )
        panel.refresh()
        self.assertEqual(panel.feedback.cget("text"), pending)
        self.assertEqual(panel.confirm.winfo_manager(), "")
        panel.confirm.invoke()
        self.assertFalse(self.path.exists())
        panel.show_record_result(service.record())
        self.assertEqual(panel.feedback.cget("text"), "Attendance recorded")
        self.assertIn(PERSON.name, panel.history.cget("text"))
        service.invalidate_capture("verification_pending")
        panel.refresh()
        self.assertEqual(panel.feedback.cget("text"), "Attendance recorded")
        service.invalidate_capture("spoof")
        panel.refresh()
        self.assertIn("Spoof", panel.feedback.cget("text"))

    def test_widget_reports_persisted_success_and_displays_failure_without_success(
        self,
    ):
        service = AttendanceService(
            AttendanceStore(self.path),
            policy=POLICY,
            registered_name={PERSON.person_id: PERSON.name}.get,
            clock=lambda: 10.0,
        )
        panel = self.dashboard(service).attendance_view

        def verify():
            request = service.begin_capture(FRAME, [BOX], 10.0)
            service.complete(
                Completion(request.context, PERSON), Completion(request.context, LIVE)
            )
            panel.refresh()
            self.assertTrue(panel.confirm.instate(("!disabled",)))

        panel.session.invoke()
        self.path.mkdir()
        verify()
        panel.confirm.invoke()
        self.assertIn("Save failed", panel.feedback.cget("text"))
        self.assertNotIn("Attendance recorded", panel.feedback.cget("text"))
        self.assertTrue(panel.confirm.instate(("disabled",)))

        self.path.rmdir()
        verify()
        panel.confirm.invoke()
        self.assertEqual(panel.feedback.cget("text"), "Attendance recorded")
        self.assertIn(PERSON.name, panel.history.cget("text"))
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(
                db.execute("SELECT person_id, status FROM attendance").fetchall(),
                [(PERSON.person_id, "present")],
            )
        verify()
        panel.confirm.invoke()
        self.assertEqual(
            panel.feedback.cget("text"), "Already recorded in this session"
        )
