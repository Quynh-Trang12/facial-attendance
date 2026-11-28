"""Camera-like display workloads; informational history cannot authorize actions."""

import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import Mock

import cv2
import numpy as np

from facial_attendance.ui.emotion_presentation import EmotionPresentation, appearance
from facial_attendance.ui.live_preview import (
    CameraPacket,
    LiveController,
    WorkResult,
    evaluate_adapters,
)
from facial_attendance.ui.presentation import StablePresentation
from facial_attendance.ui.webcam_prototype import (
    Completion,
    Emotion,
    OperationError,
    StubAdapters,
)
from tests import test_webcam_live as legacy

BOX = (60, 40, 220, 200)


def camera_frame(box=BOX, *, noise=0, light=0, invert=False, seed=1):
    """Same moving textured subject, sensor noise and exposure changes, no person image."""
    y, x = np.indices((160, 160))
    patch = np.stack((70 + x // 2, 60 + y // 2, 70 + (x + y) // 4), axis=-1).astype(
        np.uint8
    )
    if invert:
        patch = 255 - patch
    frame = np.full((240, 320, 3), 70, dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = cv2.resize(patch, (x2 - x1, y2 - y1))
    values = frame.astype(np.float32) + light
    if noise:
        values += np.random.default_rng(seed).normal(0, noise, frame.shape)
    frame = np.clip(values, 0, 255).astype(np.uint8)
    frame.flags.writeable = False
    return frame


class EmotionPresentationTests(unittest.TestCase):
    def setUp(self):
        self.display = EmotionPresentation()
        self.live = LiveController()
        self.seq = 0

    def begin(self, ts=10.0, frame=None, boxes=None):
        frame = camera_frame() if frame is None else frame
        boxes = [BOX] if boxes is None else boxes
        self.seq += 1
        token = (1, self.seq)
        self.display.observe_preview(frame, ts)
        self.display.new_context()
        self.live.begin_sample(token, ts)
        req = self.live.detected(WorkResult(token, boxes), frame)
        self.display.detected(token, frame, boxes, ts)
        if req:
            self.assertTrue(self.display.register(token, req, ts))
        return token, req

    def complete(self, ts=10.0, frame=None, box=BOX, label=1, latency=0.04):
        token, req = self.begin(ts, frame, [box])
        adapters = StubAdapters("Live match")
        adapters.emotion = lambda req: Completion(
            req.context,
            Emotion(label, ("Negative", "Neutral", "Positive")[label], 0.711),
        )
        batch = evaluate_adapters((req, adapters))
        self.live.completed(WorkResult(token, batch))
        self.assertTrue(self.display.accept(token, req, batch, ts, ts + latency))
        return self.display.view(ts + latency)

    def test_routine_context_retains_emotion_but_clears_authority(self):
        self.complete()
        self.begin(10.3)
        raw = self.live.controller.state
        before = asdict(raw)
        self.assertEqual(self.display.view(10.4).label, "Neutral")
        self.assertIsNone(raw.recognition)
        self.assertIsNone(raw.liveness)
        self.assertIsNone(raw.emotion)
        self.assertFalse(raw.can_save)
        self.assertEqual(asdict(raw), before)
        self.assertFalse(hasattr(self.display.view(10.4), "can_save"))

    def test_stationary_sensor_noise_does_not_clear_latest_emotion(self):
        for i in range(20):
            ts = 10 + i * 0.3
            frame = camera_frame(noise=7, seed=i)
            shown = self.complete(ts, frame)
            self.assertEqual(shown.label, "Neutral")
            self.display.observe_preview(camera_frame(noise=7, seed=i + 50), ts + 0.1)
            self.assertEqual(self.display.view(ts + 0.1).label, "Neutral")
        self.assertNotIn("unavailable_ambiguous_replacement", self.display.counts)

    def test_head_movement_scale_and_exposure_do_not_blank_emotion(self):
        for i in range(20):
            dx = round(12 * np.sin(i / 2))
            grow = round(10 * np.sin(i / 3))
            box = (60 + dx - grow, 40 - grow, 220 + dx + grow, 200 + grow)
            ts = 10 + i * 0.3
            self.complete(
                ts, camera_frame(box, noise=7, light=round(18 * np.sin(i)), seed=i), box
            )
            self.assertEqual(self.display.view(ts + 0.2).label, "Neutral")
        self.assertGreater(min(self.display.values["appearance_correlation"]), 0.95)

    def test_new_prediction_updates_without_majority_overriding_raw_class(self):
        self.assertEqual(self.complete(label=1).label, "Neutral")
        self.assertEqual(self.complete(10.3, label=2).label, "Positive")
        self.assertEqual(self.live.controller.state.emotion.class_id, 2)

    def test_one_detector_miss_has_bounded_explicit_display_grace(self):
        self.complete()
        self.begin(10.3, boxes=[])
        shown = self.display.view(10.6)
        self.assertEqual(shown.label, "Neutral")
        self.assertEqual(shown.reason, "detector_gap_grace")
        self.assertEqual(self.live.controller.state.ui_state, "no_face")
        self.assertFalse(self.live.controller.state.can_save)
        self.assertEqual(self.display.view(10.651).label, "Unavailable")
        self.assertEqual(self.display.reason, "sustained_face_loss")

    def test_one_miss_then_reappearance_keeps_display_when_continuity_matches(self):
        self.complete()
        self.begin(10.3, boxes=[])
        self.complete(10.6, camera_frame(noise=7, light=12))
        self.assertEqual(self.display.view(10.7).label, "Neutral")

    def test_two_misses_clear_and_reentry_needs_new_result(self):
        self.complete()
        self.begin(10.3, boxes=[])
        self.begin(10.6, boxes=[])
        self.assertEqual(self.display.view(10.66).label, "Unavailable")
        self.begin(10.9)
        self.assertEqual(self.display.view(10.91).label, "Unavailable")

    def test_multiple_faces_clear_immediately_and_cancel_old_completion(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.begin(10.6, boxes=[BOX, (230, 30, 300, 110)])
        self.assertEqual(self.display.view(10.61).label, "Unavailable")
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.62))

    def test_same_box_replacement_clears_history_and_old_pending(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.begin(10.6, camera_frame(invert=True))
        self.assertEqual(self.display.view(10.61).label, "Unavailable")
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.62))
        self.assertEqual(self.display.counts["clear_ambiguous_replacement"], 1)

    def test_large_jump_cannot_inherit_another_face_emotion(self):
        self.complete()
        box = (180, 90, 300, 220)
        self.begin(10.3, camera_frame(box), [box])
        self.assertEqual(self.display.view(10.31).label, "Unavailable")
        self.assertEqual(self.display.reason, "ambiguous_box_jump")

    def test_uninformative_region_cannot_retain_history(self):
        self.complete()
        self.begin(10.3, np.full((240, 320, 3), 100, np.uint8))
        self.assertEqual(self.display.view(10.31).label, "Unavailable")
        self.assertIsNone(appearance(np.full((240, 320, 3), 100, np.uint8), BOX)[0])

    def test_expiry_is_from_capture_and_not_renewed_by_context_or_preview(self):
        self.complete()
        self.display.observe_preview(camera_frame(), 10.79)
        self.display.new_context()
        self.assertEqual(self.display.view(10.799).label, "Neutral")
        self.assertEqual(self.display.view(10.801).label, "Unavailable")
        self.assertEqual(self.display.reason, "freshness_expired")

    def test_old_history_expiry_does_not_cancel_a_new_still_fresh_pending_result(self):
        self.complete()
        token, req = self.begin(10.6)
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.assertEqual(self.display.view(10.81).label, "Unavailable")
        self.assertTrue(self.display.accept(token, req, batch, 10.6, 10.9))
        self.assertEqual(self.display.view(10.9).label, "Positive")

    def test_variable_latency_does_not_inherit_raw_authority_from_superseded_context(
        self,
    ):
        self.complete()
        token, req = self.begin(10.3)
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.seq += 1
        next_token = (1, self.seq)
        self.live.begin_sample(next_token, 10.6)
        frame = camera_frame(noise=7)
        self.display.observe_preview(frame, 10.6)
        self.display.detected(next_token, frame, [BOX], 10.6)
        self.assertFalse(self.live.completed(WorkResult(token, batch)))
        self.assertTrue(self.display.accept(token, req, batch, 10.3, 10.72))
        self.assertEqual(self.display.view(10.72).source_context, req.context)
        self.assertIsNone(self.live.controller.state.recognition)
        self.assertIsNone(self.live.controller.state.liveness)
        self.assertFalse(self.live.controller.state.can_save)

    def test_late_result_cannot_extend_800ms_limit(self):
        token, req = self.begin()
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.assertFalse(self.display.accept(token, req, batch, 10.0, 10.801))
        self.assertEqual(self.display.view(10.81).label, "Unavailable")

    def test_pending_prediction_cannot_repopulate_after_sustained_face_loss(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.begin(10.6, boxes=[])
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.96))
        self.assertEqual(self.display.view(10.96).label, "Unavailable")

    def test_malformed_prediction_is_cleared_without_changing_raw_policy(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = (Completion(req.context, Emotion(2, "Neutral", 0.8)),)
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.34))
        self.assertEqual(self.display.view(10.34).label, "Unavailable")
        self.assertEqual(self.display.reason, "invalid_prediction")

    def test_wrong_context_token_box_frame_or_timestamp_cannot_be_relabelled(self):
        for which in ("context", "token", "bbox", "frame", "time"):
            self.setUp()
            token, req = self.begin()
            batch = evaluate_adapters((req, StubAdapters("Live match")))
            if which == "context":
                batch = (
                    replace(batch[-1], context=replace(req.context, face_id="other")),
                )
            if which == "token":
                token = (1, 99)
            if which == "bbox":
                req = replace(req, bbox=(61, 40, 221, 200))
            if which == "frame":
                req = replace(req, frame=camera_frame())
            self.assertFalse(
                self.display.accept(
                    token, req, batch, 9.99 if which == "time" else 10.0, 10.04
                ),
                which,
            )

    def test_duplicate_completion_cannot_refresh_display(self):
        token, req = self.begin()
        batch = evaluate_adapters((req, StubAdapters("Live match")))
        self.assertTrue(self.display.accept(token, req, batch, 10.0, 10.04))
        self.assertFalse(self.display.accept(token, req, batch, 10.0, 10.05))

    def test_spoof_clears_even_when_authoritative_context_was_superseded(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = evaluate_adapters((req, StubAdapters("Spoof")))
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.34))
        self.assertEqual(self.display.view(10.34).label, "Unavailable")

    def test_emotion_model_failure_clears_but_recognition_error_does_not(self):
        self.complete()
        token, req = self.begin(10.3)
        batch = (
            Completion(
                req.context,
                OperationError("recognition", "inference_error", "failed", True),
            ),
        )
        self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.34))
        self.assertEqual(self.display.view(10.34).label, "Neutral")
        token, req = self.begin(10.6)
        batch = (
            Completion(
                req.context,
                OperationError("emotion", "inference_error", "failed", False),
            ),
        )
        self.assertFalse(self.display.accept(token, req, batch, 10.6, 10.64))
        self.assertEqual(self.display.view(10.64).label, "Unavailable")

    def test_retry_camera_failure_action_change_and_shutdown_cancel_history(self):
        for reason in ("retry", "camera_failure", "action_change", "shutdown"):
            self.setUp()
            self.complete()
            token, req = self.begin(10.3)
            batch = evaluate_adapters((req, StubAdapters("Live match")))
            self.display.reset(reason)
            self.assertFalse(self.display.accept(token, req, batch, 10.3, 10.34))
            self.assertEqual(self.display.view(10.34).label, "Unavailable")

    def test_generation_and_dimension_changes_clear_history(self):
        self.complete()
        frame = camera_frame()
        self.display.detected((2, 1), frame, [BOX], 10.3)
        self.assertEqual(self.display.view(10.31).label, "Unavailable")
        self.complete(10.6)
        self.display.observe_preview(np.zeros((480, 640, 3), np.uint8), 10.7)
        self.assertEqual(self.display.view(10.71).label, "Unavailable")

    def test_measurements_are_bounded_and_report_actual_clear_reasons(self):
        self.complete()
        self.display.view(10.04)
        self.display.clear("camera_failure")
        self.display.view(10.1)
        for i in range(1000):
            self.display.measure("test", i)
        audit = self.display.audit()
        self.assertEqual(audit["counts"]["unavailable_camera_failure"], 1)
        self.assertEqual(audit["recent_measurements"]["test"]["n"], 256)
        self.assertEqual(audit["freshness_limit_ms"], 800)

    def test_trained_emotion_can_finish_after_recognition_error_without_authorization(
        self,
    ):
        token, req = self.begin()
        adapters = StubAdapters("Live match")
        adapters.trained_recognition = adapters.trained_emotion = True
        adapters.recognition = Mock(side_effect=RuntimeError("failed"))
        original = adapters.emotion
        adapters.emotion = Mock(side_effect=original)
        batch = evaluate_adapters((req, adapters))
        self.live.completed(WorkResult(token, batch))
        self.assertEqual(self.live.controller.state.ui_state, "error")
        self.assertFalse(self.live.controller.state.can_save)
        self.assertIsNone(self.live.controller.state.emotion)
        self.assertEqual(adapters.emotion.call_count, 1)
        self.assertTrue(self.display.accept(token, req, batch, 10.0, 10.04))


class EmotionGuiTests(unittest.TestCase):
    setUp = legacy.LiveAppTests.setUp

    def install(self):
        app = self.app
        app.presentation = StablePresentation()
        app.emotion_presentation = EmotionPresentation()
        for name in ("status", "identity", "emotion", "liveness"):
            setattr(app, "display_" + name, Mock())
        self.packet()
        return app

    def packet(self, seq=1, frame=None):
        packet = CameraPacket(
            self.app._generation,
            seq,
            self.now,
            camera_frame() if frame is None else frame,
        )
        self.app.feed.latest.return_value = packet
        return packet

    def finish(self):
        app = self.app
        app.detector.finish([BOX])
        app._tick()
        app.inference.finish(evaluate_adapters(app.inference.job[1]))
        app._tick()

    def test_routine_updates_and_confidence_changes_do_not_rewrite_emotion(self):
        app = self.install()
        app._tick()
        self.finish()
        app.display_emotion.set.reset_mock()
        app.adapters.emotion = lambda req: Completion(
            req.context, Emotion(2, "Positive", 0.712)
        )
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2, camera_frame(noise=7, light=12))
        app._tick()
        self.assertIsNone(app.controller.state.liveness)
        self.assertFalse(app.controller.state.can_save)
        self.finish()
        app.display_emotion.set.assert_not_called()
        self.assertEqual(app.controller.state.emotion.confidence, 0.712)

    def test_no_face_raw_clears_immediately_but_emotion_only_has_grace(self):
        app = self.install()
        app._tick()
        self.finish()
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2)
        app._tick()
        app.detector.finish([])
        app._tick()
        self.assertIsNone(app.controller.state.emotion)
        self.assertIsNone(app.controller.state.liveness)
        self.assertIn("Positive", app.display_emotion.set.call_args.args[0])
        self.assertFalse(app.controller.state.can_save)
        self.now += 0.351
        self.packet(3)
        app._tick()
        self.assertIn("Unavailable", app.display_emotion.set.call_args.args[0])

    def test_superseded_inference_can_update_only_informational_display(self):
        app = self.install()
        app._tick()
        app.detector.finish([BOX])
        app._tick()
        source = app.inference.job
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2)
        app._tick()
        app.detector.finish([BOX])
        app._tick()
        app.inference.finish(evaluate_adapters(source[1]))
        app._tick()
        self.assertIn("Positive", app.display_emotion.set.call_args.args[0])
        self.assertIsNone(app.controller.state.emotion)
        self.assertIsNone(app.controller.state.liveness)
        self.assertFalse(app.controller.state.can_save)

    def test_raw_diagnostic_json_remains_current_and_not_historical(self):
        app = self.install()
        app._debug_visible = True
        app.emotion_display_info = Mock()
        app._tick()
        self.finish()
        old = app.controller.state.context
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2)
        app._tick()
        payload = json.loads(app.payload.insert.call_args.args[1])
        self.assertIsNone(payload["emotion"])
        self.assertIsNone(payload["recognition"])
        self.assertFalse(payload["can_save"])
        self.assertEqual(app.emotion_presentation.view(self.now).source_context, old)

    def test_retry_and_camera_failure_clear_independent_display(self):
        app = self.install()
        app._tick()
        self.finish()
        app.retry()
        self.assertIn("Unavailable", app.display_emotion.set.call_args.args[0])
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2)
        app._tick()
        self.finish()
        app.feed.latest.return_value = CameraPacket(
            app._generation, 3, self.now, error="unavailable"
        )
        app._tick()
        self.assertIn("Unavailable", app.display_emotion.set.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
