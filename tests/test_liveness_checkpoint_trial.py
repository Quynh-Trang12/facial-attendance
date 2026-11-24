"""Experimental scoring with real checkpoint opt-in and GUI safety regressions."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np

from facial_attendance.liveness.checkpoint_trial import LivenessCheckpointTrial, TrialScore
from facial_attendance.ui.diagnostic_snapshot import privacy_safe_snapshot
from facial_attendance.ui.live_preview import CameraPacket, evaluate_adapters
import test_emotion_presentation as fixtures
import test_webcam_live as webcam


class CheckpointTrialTests(unittest.TestCase):
    def test_threshold_boundary_is_an_observation(self):
        sample = TrialScore("Full", "ee6f1341785d22ad", .2, .2)
        self.assertTrue(sample.threshold_met)
        self.assertFalse(hasattr(sample, "can_save"))

    def test_unrecognized_checkpoint_is_rejected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional trained runtime is unavailable")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pt"
            path.write_bytes(b"not a recognized checkpoint")
            with self.assertRaisesRegex(ValueError, "supplied liveness checkpoints"):
                LivenessCheckpointTrial([path])

    @unittest.skipUnless(os.getenv("FA_TRIAL_FULL") and os.getenv("FA_TRIAL_BASELINE"),
                         "Set local paths for both supplied liveness checkpoints")
    def test_real_checkpoints_score_the_same_input_with_their_own_thresholds(self):
        import torch
        torch.set_num_threads(1)
        trial = LivenessCheckpointTrial([os.environ["FA_TRIAL_FULL"], os.environ["FA_TRIAL_BASELINE"]])
        crop = fixtures.camera_frame()[40:200, 60:220]
        full, baseline = trial(crop)
        self.assertEqual((full.model_version, baseline.model_version), ("ee6f1341785d22ad", "97208d8b15f8ac34"))
        self.assertEqual((full.threshold, baseline.threshold), (.20014987885951996, .09586922824382782))
        self.assertTrue(all(0 <= result.score <= 1 for result in (full, baseline)))
        with self.assertRaises(ValueError):
            trial(np.zeros((32, 32, 3), dtype=np.float32))

    def test_export_only_keeps_allowlisted_score_fields(self):
        source = {"liveness": {"experimental_comparison": {
            "method": "experimental-softmax-spoof-index-1", "can_authorize": True, "status": "ready",
            "scores": [{"variant": "Full", "model_version": "ee6f1341785d22ad",
                        "score": .3, "threshold": .2, "threshold_met": True, "crop": "private"}],
        }}}
        summary = privacy_safe_snapshot(source, {})["liveness"]["experimental_comparison"]
        self.assertFalse(summary["can_authorize"])
        self.assertTrue(summary["scores"][0]["threshold_met"])
        self.assertNotIn("crop", summary["scores"][0])


class TrialGuiTests(unittest.TestCase):
    setUp = fixtures.EmotionGuiTests.setUp
    packet = fixtures.EmotionGuiTests.packet

    def install(self):
        app = fixtures.EmotionGuiTests.install(self)
        app.metrics = Mock()
        app._liveness_trial_worker = webcam.ManualWorker()
        self.scores = (TrialScore("Full", "ee6f1341785d22ad", .7, .2),)
        return app

    def confirm_face(self):
        app = self.app
        app._tick()
        app.detector.finish([fixtures.BOX])
        app._tick()

    def finish_trial(self):
        app = self.app
        app._liveness_trial_worker.finish(self.scores)
        app._tick()

    def test_trial_score_is_display_only_and_clears_on_loss_or_multiple_faces(self):
        for boxes in ([], [fixtures.BOX, (230, 30, 300, 110)]):
            with self.subTest(boxes=boxes):
                app = self.install()
                self.confirm_face()
                app.inference.finish(evaluate_adapters(app.inference.job[1]))
                self.finish_trial()
                self.assertIn("Spoof threshold met", app.display_liveness.set.call_args.args[0])
                self.assertEqual(app.controller.state.liveness.source, "stub")
                self.assertFalse(app.controller.state.can_save)
                self.now += .42
                self.packet(2)
                app._tick()
                app.detector.finish(boxes)
                app._tick()
                self.assertIsNone(app.current_liveness_trial())
                self.assertIn("No current score", app.display_liveness.set.call_args.args[0])
                app.retry()

    def test_retry_rejects_a_late_score_even_after_a_new_face_is_detected(self):
        app = self.install()
        self.confirm_face()
        original = app._liveness_trial_worker.job
        app.retry()
        self.now += .42
        self.packet(2)
        self.confirm_face()
        self.assertIs(app._liveness_trial_worker.job, original)
        self.finish_trial()
        self.assertIsNone(app.current_liveness_trial())
        self.assertFalse(app.controller.state.can_save)

    def test_camera_error_clears_trial_and_keeps_enrollment_blocked(self):
        app = self.install()
        self.confirm_face()
        self.finish_trial()
        app.feed.latest.return_value = CameraPacket(app._generation, 2, self.now, error="unavailable")
        app._tick()
        self.assertIsNone(app.current_liveness_trial())
        self.assertFalse(app.controller.state.can_save)
