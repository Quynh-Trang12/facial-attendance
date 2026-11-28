"""Check that released liveness cannot reuse a different or expired capture."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from facial_attendance.ui.liveness_preview import (
    THRESHOLD,
    LivenessPreview,
    ReleasedLiveness,
)
from facial_attendance.ui.webcam_app import main
from facial_attendance.ui.webcam_prototype import (
    Envelope,
    PrototypeController,
    Recognition,
)

BOX = (10, 12, 70, 72)
LIVE = {
    "status": "live",
    "spoof_score": 0.2,
    "threshold": THRESHOLD,
    "allow_attendance": True,
}
SPOOF = {
    "status": "spoof",
    "spoof_score": 0.8,
    "threshold": THRESHOLD,
    "allow_attendance": False,
}


class ReleasedPreviewTests(unittest.TestCase):
    def setUp(self):
        self.preview = LivenessPreview(THRESHOLD)
        self.preview.advance(0)

    def sample(
        self, sequence, timestamp, person="test-001", prediction=LIVE, *, box=BOX
    ):
        recognition = Recognition(
            person, "Test name", 0.9, person is not None, source="model"
        )
        state = Envelope(recognition=recognition, ui_state="preview")
        token = (1, sequence)
        self.preview.accept(token, box, timestamp, state, prediction)
        self.assertFalse(state.can_save)
        return SimpleNamespace(token=token, sample_time=timestamp, boxes=[box])

    def test_uses_mean_of_valid_scores_instead_of_per_frame_labels(self):
        self.sample(0, 0.2, prediction=LIVE)
        self.sample(1, 0.6, prediction=SPOOF)
        self.sample(2, 1.0, prediction=None)
        self.sample(3, 1.4, prediction={"status": "uncertain", "spoof_score": None})
        self.preview.advance(2.9)
        self.assertIsNone(self.preview.last_check)
        self.assertIn("Collecting", self.preview.display_text())
        self.preview.advance(3.0)
        self.assertEqual(self.preview.last_check.decision, "spoof")
        self.assertEqual(self.preview.last_check.valid_frames, 2)
        self.assertEqual(self.preview.last_check_window, (0.0, 3.0))
        self.assertEqual(self.preview.last_check.mean_spoof_score, 0.5)
        self.assertEqual(self.preview.display_text().splitlines()[-1], "spoof")

    def test_last_scan_stays_visible_and_every_new_scan_has_its_own_scores(self):
        self.sample(0, 0.1, prediction=SPOOF)
        self.preview.advance(3.0)
        previous = self.preview.display_text()
        self.sample(1, 3.1)
        self.sample(2, 3.5, prediction=SPOOF)
        self.sample(3, 3.9)
        self.preview.advance(5.9)
        self.assertEqual(self.preview.display_text(), previous)
        self.preview.advance(6.0)
        self.assertEqual(self.preview.last_check.decision, "live")
        self.assertEqual(self.preview.last_check.valid_frames, 3)
        self.assertAlmostEqual(self.preview.last_check.mean_spoof_score, 0.4)

    def test_empty_scan_is_uncertain_and_threshold_equality_is_spoof(self):
        self.sample(0, 0.2, prediction=None)
        self.assertIn("Collecting", self.preview.display_text())
        self.preview.advance(3)
        self.assertEqual(self.preview.last_check.decision, "uncertain")
        self.assertIsNone(self.preview.last_check.mean_spoof_score)
        self.assertEqual(self.preview.last_check.valid_frames, 0)
        self.sample(0, 3.1, prediction={**SPOOF, "spoof_score": THRESHOLD})
        self.preview.advance(6)
        self.assertEqual(self.preview.last_check.decision, "spoof")

    def test_waits_for_a_pending_frame_and_rejects_duplicate_or_late_scores(self):
        self.sample(0, 0.1, prediction=LIVE)
        self.preview.advance(3.1, pending_at=2.9)
        self.assertIsNone(self.preview.last_check)
        self.sample(1, 2.9, prediction=SPOOF)
        self.sample(1, 2.9, prediction=SPOOF)
        self.preview.advance(3.2)
        self.assertEqual(self.preview.last_check.valid_frames, 2)
        self.assertEqual(self.preview.last_check.mean_spoof_score, 0.5)
        self.sample(2, 2.98, prediction=SPOOF)
        self.preview.advance(6.3)
        self.assertEqual(self.preview.last_check.decision, "uncertain")

    def test_unmatched_frame_does_not_split_scan_but_changed_identity_does(self):
        self.sample(0, 0.1, prediction=SPOOF)
        self.sample(1, 0.5, person=None, prediction=SPOOF)
        self.sample(2, 0.9, person="test-002")
        self.preview.advance(3.0)
        self.assertIsNone(self.preview.last_check)
        self.preview.advance(4.0)
        self.assertEqual(self.preview.last_check.valid_frames, 1)
        self.assertEqual(self.preview.last_check.decision, "live")

    def test_disjoint_face_clears_old_result_and_restarts_scan(self):
        self.sample(0, 0.1, person=None, prediction=SPOOF)
        self.preview.advance(3.0)
        self.sample(1, 3.1, person=None, prediction=SPOOF)
        self.sample(2, 3.5, person=None, box=(80, 12, 140, 72))
        self.assertIn("Collecting", self.preview.display_text())
        self.assertIsNone(self.preview.last_check_window)
        self.preview.advance(6.5)
        self.assertEqual(self.preview.last_check.valid_frames, 1)
        self.assertEqual(self.preview.last_check.decision, "live")

    def test_raw_score_is_diagnostic_and_reset_discards_a_partial_scan(self):
        current = self.sample(0, 0.1)
        self.assertEqual(self.preview.current(current, 0.2, 1).spoof_score, 0.2)
        self.assertIsNone(self.preview.current(current, 1.2, 1))
        self.assertIsNone(self.preview.last_check)
        self.preview.clear()
        self.preview.advance(0.3)
        self.preview.advance(3.4)
        self.assertEqual(self.preview.last_check.decision, "uncertain")


class ReleasedLivenessStartupTests(unittest.TestCase):
    def test_local_workspace_uses_the_installed_release_by_default(self):
        checkpoint = Mock()
        checkpoint.is_file.return_value = True
        tk = SimpleNamespace(Tk=Mock(), TclError=RuntimeError)
        with (
            patch.dict(
                sys.modules,
                {"cv2": SimpleNamespace(), "tkinter": tk, "PIL.ImageTk": Mock()},
            ),
            patch(
                "facial_attendance.ui.recognition_adapter.TrainedPreviewAdapters"
            ) as trained,
            patch(
                "facial_attendance.ui.webcam_app.LOCAL_LIVENESS_CHECKPOINT", checkpoint
            ),
            patch("facial_attendance.ui.webcam_app.LiveWebcamApp") as app,
            patch("facial_attendance.ui.liveness_preview.ReleasedLiveness") as released,
        ):
            self.assertEqual(
                main(
                    [
                        "--local-workspace",
                        "--face-checkpoint",
                        "face.pt",
                        "--emotion-checkpoint",
                        "emotion.pt",
                        "--yunet-model",
                        "yunet.onnx",
                    ]
                ),
                0,
            )
        released.assert_called_once_with(checkpoint, "yunet.onnx")
        self.assertTrue(trained.call_args.kwargs["crop_to_detector_box"])
        profile_store = app.call_args.kwargs["local_profile_store"]
        self.assertEqual(profile_store.path.name, "profiles-box.sqlite3")
        self.assertEqual(profile_store._profile_version, 1)
        self.assertIs(app.call_args.kwargs["released_liveness"], released.return_value)
        self.assertIs(app.call_args.kwargs["detector"], released.return_value.detect)


@unittest.skipUnless(
    os.getenv("FA_LIVENESS_RELEASE") and os.getenv("FA_LIVENESS_YUNET"),
    "Set the released checkpoint and YuNet paths to exercise the supplied weights",
)
class ReleasedWeightsTests(unittest.TestCase):
    def test_gui_crop_matches_released_full_frame_inference(self):
        import cv2
        import numpy as np
        import torch

        from facial_attendance.ui.diagnostics import Diagnostics
        from facial_attendance.ui.webcam_prototype import StubAdapters

        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        model = ReleasedLiveness(
            os.environ["FA_LIVENESS_RELEASE"], os.environ["FA_LIVENESS_YUNET"]
        )
        frame = np.random.default_rng(42).integers(
            0, 256, size=(200, 240, 3), dtype=np.uint8
        )
        self.assertEqual(model.detect(np.zeros_like(frame)), [])
        model.adapter.detector = model.detector
        # Control only the located region; both paths execute the real model.
        with patch.object(model.detector, "detect", return_value=[(10, 12, 60, 60)]):
            request = PrototypeController().begin_capture(frame, model.detect(frame))
            self.assertEqual(request.bbox, BOX)
            adapters = StubAdapters("Live match")
            adapters.diagnostics = Diagnostics()
            _, prediction = model.evaluate((request, adapters, False))
            expected = model.adapter.predict_frame(frame)
        self.assertEqual(prediction["spoof_score"], expected["spoof_score"])
        self.assertEqual(prediction["status"], expected["status"])
        self.assertEqual(prediction["threshold"], THRESHOLD)
