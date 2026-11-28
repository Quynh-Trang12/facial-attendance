"""Panel caching, diagnostic bounds and strict current-result regression tests."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from facial_attendance.ui.diagnostics import Diagnostics
from facial_attendance.ui.live_preview import evaluate_adapters
from facial_attendance.ui.webcam_prototype import (
    Completion,
    Emotion,
    Liveness,
    PrototypeController,
    StubAdapters,
)
from tests import test_webcam_live as legacy


class DiagnosticBoundsTests(unittest.TestCase):
    def test_recent_percentiles_are_bounded_and_existing_timing_fields_remain_compatible(
        self,
    ):
        metrics = Diagnostics(True, clock=lambda: 10)
        for _ in range(1000):
            metrics.duration("model", 0.01)
        self.assertEqual(len(metrics._samples["model"]), 256)
        metrics.clock = lambda: 15
        result = metrics.snapshot(force=True)
        self.assertEqual(result["timings_ms"]["model"]["n"], 1000)
        self.assertAlmostEqual(result["recent_p95_ms"]["model"], 10)
        self.assertEqual(len(metrics._samples), 0)

    def test_disabled_diagnostics_allocate_no_samples(self):
        metrics = Diagnostics(False)
        for _ in range(100):
            metrics.duration("model", 0.1)
        self.assertFalse(metrics._samples)

    def test_module_timings_respect_liveness_gating_and_original_context(self):
        for scenario, modules in (
            ("Live match", {"liveness", "recognition", "emotion"}),
            ("Spoof", {"liveness"}),
            ("Uncertain", {"liveness", "emotion"}),
        ):
            controller = PrototypeController()
            request = controller.begin_capture(legacy.Frame(), [legacy.BOX])
            adapters = StubAdapters(scenario)
            adapters.diagnostics = Diagnostics(True)
            completions = evaluate_adapters((request, adapters))
            self.assertTrue(
                all(item.context == request.context for item in completions)
            )
            report = adapters.diagnostics.snapshot(force=True)
            self.assertEqual(set(report["timings_ms"]), modules)
            self.assertFalse(controller.state.can_save)


class PanelRenderTests(unittest.TestCase):
    setUp = legacy.LiveAppTests.setUp
    packet = legacy.LiveAppTests.packet
    run_one_face = legacy.LiveAppTests.run_one_face

    def test_two_processing_phases_do_not_rewrite_unchanged_visible_text(self):
        app = self.app
        app.live.status = "Processing — detecting a face"
        app._render()
        app.summary.set.reset_mock()
        app.live.status = "Processing — simulated adapters"
        app._render()
        app.summary.set.assert_not_called()

    def test_selected_emotion_metadata_is_written_once_and_not_replaced_on_every_sample(
        self,
    ):
        self.run_one_face()
        app = self.app
        app._trained_emotion_enabled = True
        app.model_info = Mock()
        trained = Emotion(
            2,
            "Positive",
            0.8,
            source="model",
            model_id="test-model",
            model_version="a" * 64,
            checkpoint_sha256="b" * 64,
            preprocessing_version="test",
            selected_epoch=9,
        )
        app.controller.state = replace(app.controller.state, emotion=trained)
        app._render()
        self.assertIn("Selected EmotionCNN", app.model_info.set.call_args.args[0])
        app.controller.state = replace(
            app.controller.state,
            context=replace(app.controller.state.context, capture_id="next"),
        )
        app._render()
        self.assertEqual(app.model_info.set.call_count, 1)
        app.live.begin_sample((1, 2), 10.3)
        app._render()
        self.assertEqual(app.model_info.set.call_count, 1)
        self.assertIsNone(app.controller.state.emotion)

    def test_registration_button_is_never_repacked_by_routine_result_updates(self):
        self.run_one_face()
        app = self.app
        app._trained_recognition_enabled = True
        app.registration_prompt = Mock()
        app.registration_button = Mock()
        app._last_render = None
        app._render()
        app.controller.state = replace(
            app.controller.state,
            context=replace(app.controller.state.context, capture_id="next"),
        )
        app._render()
        self.assertEqual(app.registration_button.configure.call_count, 1)
        app.registration_button.pack.assert_not_called()
        app.registration_button.pack_forget.assert_not_called()

    def test_waiting_registration_reports_existing_profile_match(self):
        self.run_one_face()
        app = self.app
        app._local_workspace = True
        app._developer_mode = True
        app._trained_recognition_enabled = True
        app._registration_requested = True
        app.registration_prompt = Mock()
        app.registration_button = Mock()
        app.liveness_preview = SimpleNamespace(
            display_text=lambda: "Liveness · Collecting frames…"
        )
        app._trial_profiles = SimpleNamespace(
            active=False,
            liveness_threshold=0.46,
            session=SimpleNamespace(MAX_SAMPLE_AGE=0.8),
        )
        state = app.controller.state
        app.controller.state = replace(
            state,
            ui_state="preview",
            recognition=replace(state.recognition, source="model", name="Vy"),
        )

        app._render()

        app.registration_prompt.set.assert_called_with(
            "Face matches the saved profile “Vy”. Show a new person to register."
        )
        self.assertFalse(app._can_start_local_registration(self.now))
        self.assertTrue(app._registration_requested)
        app.registration_button.configure.assert_called_with(
            text="Cancel registration request", state="normal"
        )

    def test_repeated_same_context_json_updates_are_throttled_but_latest_is_eventually_rendered(
        self,
    ):
        self.run_one_face()
        app = self.app
        app._debug_visible = True
        app._last_render = None
        app._render()
        app._next_debug_update = 11
        app.payload.insert.reset_mock()
        app.controller.state = replace(
            app.controller.state, emotion=Emotion(0, "Negative")
        )
        app._render()
        app.payload.insert.assert_not_called()
        self.now = 11.01
        app._ui_time = self.now
        app._render()
        self.assertEqual(
            json.loads(app.payload.insert.call_args.args[1])["emotion"]["label"],
            "Negative",
        )

    def test_uncertain_and_spoof_bypass_json_throttle_and_never_retain_identity(self):
        for status, score in (("uncertain", 0.4), ("spoof", 0.96)):
            self.run_one_face()
            app = self.app
            app._debug_visible = True
            app._last_render = None
            app._render()
            app._next_debug_update = 100
            context = app.controller.state.context
            app.controller.receive(Completion(context, Liveness(status, "test", score)))
            app._render()
            payload = json.loads(app.payload.insert.call_args.args[1])
            self.assertEqual(payload["liveness"]["status"], status)
            self.assertIsNone(payload["recognition"])
            self.assertFalse(payload["can_save"])
            app.live.reset()
            app._next_detection = 0

    def test_face_loss_and_camera_error_clear_current_fields_even_with_hidden_diagnostics(
        self,
    ):
        for code in ("no_face", "camera_unavailable", "multiple_faces"):
            self.run_one_face()
            app = self.app
            app._debug_visible = False
            app.live.fail(code, "Controlled failure")
            app._render()
            self.assertTrue(
                all(
                    getattr(app.controller.state, key) is None
                    for key in ("recognition", "emotion", "liveness")
                )
            )
            self.assertFalse(app.controller.state.can_save)
            self.assertNotIn("Demo Person", app.summary.set.call_args.args[0])
            app.live.reset()
            app._next_detection = 0

    def test_slow_adapter_does_not_block_preview_or_accumulate_jobs(self):
        app = self.app
        app._tick()
        app.detector.finish([legacy.BOX])
        app._tick()
        original = app.inference.job
        draws = app._draw_frame.call_count
        for sequence in range(2, 20):
            self.now += 0.033
            self.packet(sequence)
            app._tick()
        self.assertIs(app.inference.job, original)
        self.assertGreaterEqual(app._draw_frame.call_count - draws, 18)
        self.assertFalse(app.controller.state.can_save)


if __name__ == "__main__":
    unittest.main()
