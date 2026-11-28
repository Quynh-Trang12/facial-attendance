"""Hardware-free automatic scheduling, invalidation and bounded-worker tests."""

import json
import sys
import unittest
from dataclasses import asdict, replace
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

from facial_attendance.ui.face_detection import FaceDetectorUnavailable
from facial_attendance.ui.live_preview import (
    CameraFeed,
    CameraPacket,
    LiveController,
    SingleWorker,
    WorkResult,
    evaluate_adapters,
)
from facial_attendance.ui.webcam_app import LiveWebcamApp, main
from facial_attendance.ui.webcam_prototype import SCENARIOS, Completion, StubAdapters


class Frame:
    dtype = "uint8"

    def __init__(self, width=1280, height=720):
        self.shape = height, width, 3
        self.flags = SimpleNamespace(writeable=True)

    def copy(self):
        return Frame(self.shape[1], self.shape[0])

    def __getitem__(self, selection):
        y, x = selection
        return Frame(x.stop - x.start, y.stop - y.start)


BOX = (100, 100, 300, 250)


class LiveControllerTests(unittest.TestCase):
    def setUp(self):
        self.live = LiveController()
        self.frame = Frame()

    def request(self, token=(1, 1), boxes=None):
        self.live.begin_sample(token, 10)
        return self.live.detected(
            WorkResult(token, [BOX] if boxes is None else boxes), self.frame
        )

    def finish(self, request, scenario="Live match"):
        batch = evaluate_adapters((request, StubAdapters(scenario)))
        return self.live.completed(WorkResult(self.live.token, batch))

    def assert_cleared(self):
        state = self.live.controller.state
        for name in ("recognition", "liveness", "emotion"):
            self.assertIsNone(getattr(state, name))
        self.assertFalse(state.can_save)

    def test_all_scenarios_use_original_bgr_context_and_never_authorize_save(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                request = self.request()
                self.assertIs(request.frame, self.frame)
                self.assertEqual(request.bbox, BOX)
                self.assertEqual(request.context.contract_version, "sys02-v1")
                self.assertEqual(request.context.action, "attendance")
                self.assertTrue(self.finish(request, scenario))
                state = self.live.controller.state
                self.assertEqual(state.context, request.context)
                self.assertEqual(state.liveness.source, "stub")
                self.assertFalse(asdict(state)["can_save"])
                if scenario in {"Spoof", "Uncertain"}:
                    self.assertIsNone(state.recognition)
                    self.assertEqual(self.live.controller.demo_identity, "Unavailable")
                if scenario == "Spoof":
                    self.assertIsNone(state.emotion)

    def test_face_loss_multiple_faces_and_invalid_box_clear_all_prior_results(self):
        for boxes, expected in (
            ([], "no_face"),
            ([BOX, (400, 50, 500, 150)], "multiple_faces"),
            ([(0, 0, 1400, 40)], "error"),
        ):
            with self.subTest(expected=expected):
                self.finish(self.request())
                self.assertIsNone(self.request((1, 2), boxes))
                self.assert_cleared()
                self.assertEqual(self.live.controller.state.ui_state, expected)

    def test_new_sample_clears_output_and_rejects_old_detection_and_adapter_batch(self):
        request = self.request()
        batch = evaluate_adapters((request, StubAdapters("Live match")))
        self.live.begin_sample((1, 2), 10.3)
        self.assert_cleared()
        self.assertIsNone(self.live.detected(WorkResult((1, 1), [BOX]), self.frame))
        self.assertFalse(self.live.completed(WorkResult((1, 1), batch)))
        self.assertIsNone(self.live.controller.state.context)

    def test_wrong_capture_face_attempt_action_and_version_cannot_be_relabelled(self):
        request = self.request()
        completion = StubAdapters("Live match").liveness(request)
        for field, value in (
            ("capture_id", "other"),
            ("face_id", "other"),
            ("attempt_id", "other"),
            ("action", "registration"),
            ("contract_version", "sys02-v2"),
        ):
            with self.subTest(field=field):
                wrong = Completion(
                    replace(request.context, **{field: value}), completion.result
                )
                self.assertFalse(self.live.completed(WorkResult((1, 1), (wrong,))))
                self.assertIsNone(self.live.controller.state.liveness)

    def test_reset_camera_failure_and_expiry_reject_late_results_and_clear_boxes(self):
        for invalidate in (
            self.live.reset,
            lambda: self.live.fail("camera_unavailable", "Camera unavailable"),
            lambda: self.live.expire(11.01),
        ):
            with self.subTest(invalidate=invalidate):
                request = self.request()
                batch = evaluate_adapters((request, StubAdapters("Live match")))
                self.finish(request)
                invalidate()
                self.assert_cleared()
                self.assertEqual(self.live.boxes, [])
                self.assertFalse(self.live.completed(WorkResult((1, 1), batch)))

    def test_detector_missing_exception_and_invalid_outputs_fail_closed_then_recover(
        self,
    ):
        for result, code in (
            (
                WorkResult((1, 1), error=FaceDetectorUnavailable("missing")),
                "model_unavailable",
            ),
            (WorkResult((1, 1), error=RuntimeError("bad")), "inference_error"),
            (WorkResult((1, 1), [(True, 0, 40, 40)]), "invalid_input"),
            (WorkResult((1, 1), None), "invalid_input"),
        ):
            with self.subTest(code=code):
                self.live.begin_sample((1, 1), 10)
                self.assertIsNone(self.live.detected(result, self.frame))
                self.assertEqual(
                    self.live.controller.state.errors[0].module, "detector"
                )
                self.assertEqual(self.live.controller.state.errors[0].code, code)
                self.assertEqual(self.live.boxes, [])
                self.assert_cleared()
                self.live.reset()
                self.assertTrue(self.finish(self.request((2, 1))))

    def test_adapter_failure_and_emotion_only_failure_remain_distinct(self):
        request = self.request()
        adapters = StubAdapters("Live match")
        adapters.emotion = Mock(side_effect=RuntimeError("no emotion"))
        self.live.completed(WorkResult((1, 1), evaluate_adapters((request, adapters))))
        self.assertEqual(self.live.controller.demo_identity, "Demo Person")
        self.assertIsNone(self.live.controller.state.emotion)
        self.assertFalse(self.live.controller.state.can_save)
        self.assertFalse(
            self.live.completed(WorkResult((1, 1), error=RuntimeError("worker")))
        )
        self.assert_cleared()


class ManualWorker:
    """A single-slot worker driven deterministically by tests, without threads."""

    def __init__(self):
        self.busy = False
        self.job = None
        self.result = None

    def submit(self, token, value):
        if self.busy:
            return False
        self.busy = True
        self.job = token, value
        return True

    def finish(self, value=None, error=None):
        self.result = WorkResult(self.job[0], value, error)

    def take(self):
        result, self.result = self.result, None
        if result is not None:
            self.busy = False
        return result

    def close(self):
        pass


class LiveAppTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.enterContext(
            patch(
                "facial_attendance.ui.webcam_app.monotonic",
                side_effect=lambda: self.now,
            )
        )
        app = self.app = LiveWebcamApp.__new__(LiveWebcamApp)
        app.root = Mock()
        app.live = LiveController()
        app.controller = app.live.controller
        app.adapters = StubAdapters("Live match")
        app.feed = Mock()
        app.feed.restart.return_value = 2
        app.detector = ManualWorker()
        app.inference = ManualWorker()
        app._closed = False
        app._generation = 1
        app._drawn_sequence = None
        app._next_detection = 0
        app._detection_frame = None
        app._detector_failed = False
        app._opened_at = 10
        app._last_render = None
        app._timer = "timer"
        app.preview_size = 640, 360
        app.photo = None
        for name in ("canvas", "summary", "payload", "hint", "_draw_frame"):
            setattr(app, name, Mock())
        self.packet()

    def packet(self, sequence=1, **kwargs):
        packet = CameraPacket(
            self.app._generation, sequence, self.now, Frame(), **kwargs
        )
        self.app.feed.latest.return_value = packet
        return packet

    def run_one_face(self):
        app = self.app
        app._tick()  # Automatic sample, without operator controls.
        app.detector.finish([BOX])
        app._tick()  # Detection dispatches the adapters automatically.
        app.inference.finish(evaluate_adapters(app.inference.job[1]))
        app._tick()

    def test_default_tick_automatically_detects_evaluates_labels_and_draws_without_controls(
        self,
    ):
        self.run_one_face()
        app = self.app
        self.assertEqual(app.controller.demo_identity, "Demo Person")
        text = app.summary.set.call_args.args[0]
        self.assertIn("SIMULATED", text)
        self.assertIn("Liveness (simulated): live", text)
        self.assertIn("Emotion (simulated): Positive", text)
        payload = json.loads(app.payload.insert.call_args.args[1])
        self.assertEqual(payload["context"]["action"], "attendance")
        self.assertEqual(payload["liveness"]["source"], "stub")
        self.assertFalse(payload["can_save"])
        app.canvas.create_rectangle.assert_called_with(
            50, 50, 150, 125, outline="#ffce65", width=2, tags="region"
        )

    def test_resizing_refits_the_current_frame_and_outline_without_starting_a_capture(
        self,
    ):
        import numpy as np

        app = self.app
        del app._draw_frame
        app._preview_viewport_size = (640, 360)
        app._preview_mirrored = True
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[100:250, 100:300, 2] = 255
        frame.flags.writeable = False
        app.feed.latest.return_value = replace(self.packet(), frame=frame)
        with patch("PIL.ImageTk.PhotoImage") as photo:
            self.run_one_face()
            self.assertIs(app.detector.job[1], frame)
            request = app.inference.job[1][0]
            self.assertIs(request.frame, frame)
            self.assertEqual(request.bbox, BOX)
            self.assertEqual(tuple(request.face_crop[0, 0]), (0, 0, 255))
            context = app.controller.state.context
            app._resize_preview(SimpleNamespace(width=960, height=680))
            app._tick()
            image = photo.call_args.args[0]
            self.assertEqual(image.size, (960, 540))
            self.assertEqual(image.getpixel((810, 100)), (255, 0, 0))
            self.assertEqual(image.getpixel((150, 100)), (0, 0, 0))
            app.canvas.place_configure.assert_called_with(width=960, height=540)
            app.canvas.create_rectangle.assert_called_with(
                735, 75, 885, 187.5, outline="#ffce65", width=2, tags="region"
            )
            self.assertEqual(app.controller.state.context, context)
            app._resize_preview(SimpleNamespace(width=640, height=420))
            app._tick()
            self.assertEqual(photo.call_args.args[0].size, (640, 360))
            app.canvas.create_image.assert_called_once()

    def test_registration_overlay_requires_developer_mode_and_a_visible_preview(self):
        app = self.app
        app._registration_overlay = Mock()
        app._trial_profiles = Mock()
        app._preview_item = "image"
        app._developer_mode = True
        app._render()
        app._registration_overlay.update.assert_called_with(
            app._trial_profiles.guidance.return_value, app.preview_size, self.now
        )

        app._developer_mode = False
        app._render()
        app._registration_overlay.update.assert_called_with(
            None, app.preview_size, self.now
        )
        app._developer_mode = True
        app._preview_item = None
        app._render()
        app._registration_overlay.update.assert_called_with(
            None, app.preview_size, self.now
        )

    def test_camera_preview_continues_and_work_slots_remain_bounded_while_detection_is_busy(
        self,
    ):
        app = self.app
        app._tick()
        job = app.detector.job
        for sequence in range(2, 30):
            self.now += 0.033
            self.packet(sequence)
            app._tick()
        self.assertIs(app.detector.job, job)
        self.assertEqual(app._draw_frame.call_count, 29)
        self.assertIsNone(app.inference.job)

    def test_analysis_waits_for_revised_cadence_and_samples_the_latest_packet(self):
        self.run_one_face()
        app = self.app
        self.now += 0.31
        skipped = self.packet(2)
        app._tick()
        self.assertFalse(app.detector.busy)
        app._draw_frame.assert_called_with(skipped.frame)

        self.now += 0.10
        latest = self.packet(3)
        app._tick()
        self.assertEqual(app.detector.job[0], (1, 3))
        self.assertIs(app.detector.job[1], latest.frame)
        self.assertIsNone(app.controller.state.recognition)
        self.assertIsNone(app.controller.state.liveness)
        self.assertFalse(app.controller.state.can_save)

    def test_next_sample_no_face_and_multiple_faces_clear_display(self):
        app = self.app
        for sequence, boxes, expected in (
            (2, [], "no_face"),
            (3, [BOX, (500, 100, 600, 200)], "multiple_faces"),
        ):
            with self.subTest(expected=expected):
                if sequence == 2:
                    self.run_one_face()
                self.now += app.ANALYSIS_INTERVAL + 0.01
                self.packet(sequence)
                app._tick()
                self.assertIsNone(app.controller.state.recognition)
                app.detector.finish(boxes)
                app._tick()
                self.assertEqual(app.controller.state.ui_state, expected)
                self.assertIsNone(app.controller.state.emotion)
                self.assertIsNone(app.controller.state.liveness)

    def test_slow_adapter_cannot_hide_face_loss_or_restore_old_identity(self):
        app = self.app
        app._tick()
        app.detector.finish([BOX])
        app._tick()
        old_job = app.inference.job
        self.assertTrue(app.inference.busy)
        self.now += app.ANALYSIS_INTERVAL + 0.01
        self.packet(2)
        app._tick()
        self.assertTrue(app.detector.busy)
        app.detector.finish([])
        app._tick()
        self.assertEqual(app.controller.state.ui_state, "no_face")
        app.inference.finish(evaluate_adapters(old_job[1]))
        app._tick()
        self.assertEqual(app.controller.state.ui_state, "no_face")
        self.assertIsNone(app.controller.state.recognition)
        self.assertIsNone(app.controller.state.liveness)
        self.assertIsNone(app.controller.state.emotion)
        self.assertEqual(app._draw_frame.call_count, 2)

    def test_retry_and_camera_error_do_not_accept_old_detection_or_adapter_completion(
        self,
    ):
        app = self.app
        app._tick()
        app.detector.finish([BOX])
        app._tick()
        old_job = app.inference.job
        app.retry()
        app.inference.finish(evaluate_adapters(old_job[1]))
        self.packet(1)
        app._tick()
        self.assertIsNone(app.controller.state.recognition)
        app.detector.finish([BOX])
        self.packet(2, error="disconnected")
        app._tick()
        self.assertEqual(app.controller.state.errors[0].code, "camera_unavailable")
        self.assertEqual(app.live.boxes, [])
        self.assertIsNone(app.controller.state.emotion)

    def test_expired_detection_and_stalled_camera_cannot_restore_results(self):
        app = self.app
        app._tick()
        app.detector.finish([BOX])
        self.now += 1.01
        # Fresh preview, but an obsolete analysis frame.
        self.packet(2)
        app._tick()
        self.assertIsNone(app.inference.job)
        self.assertIsNone(app.controller.state.recognition)
        self.now += 1.01  # No new camera packet.
        app._tick()
        self.assertEqual(app.controller.state.errors[0].code, "camera_unavailable")
        self.assertEqual(app.live.boxes, [])

    def test_detector_unavailable_stays_visible_until_retry_and_no_boxes_are_invented(
        self,
    ):
        app = self.app
        app._tick()
        app.detector.finish(error=FaceDetectorUnavailable("missing Haar"))
        app._tick()
        self.assertEqual(app.controller.state.errors[0].code, "model_unavailable")
        self.assertIn("Detector unavailable", app.summary.set.call_args.args[0])
        self.assertEqual(app.live.boxes, [])
        self.now += 0.5
        self.packet(2)
        app._tick()
        self.assertFalse(app.detector.busy)
        app.retry()
        self.packet(1)
        app._tick()
        self.assertTrue(app.detector.busy)

    def test_close_is_idempotent_and_late_tick_does_not_schedule(self):
        self.run_one_face()
        app = self.app
        app.close()
        app.close()
        app.root.after.reset_mock()
        app._tick()
        app.root.after.assert_not_called()
        app.feed.close.assert_called_once()
        app.root.destroy.assert_called_once()
        self.assertIsNone(app.controller.state.recognition)

    def test_default_constructor_contains_no_dropdowns_or_manual_evaluation_buttons(
        self,
    ):
        widgets = []

        def widget(*args, **kwargs):
            obj = Mock()
            widgets.append(kwargs)
            return obj

        ttk = SimpleNamespace(
            Frame=widget,
            Label=widget,
            Button=widget,
            Combobox=Mock(side_effect=AssertionError("No dropdowns")),
        )
        tk = SimpleNamespace(ttk=ttk, Canvas=widget, StringVar=Mock)
        with (
            patch.dict(
                sys.modules,
                {
                    "tkinter": tk,
                    "tkinter.ttk": ttk,
                    "tkinter.scrolledtext": SimpleNamespace(ScrolledText=widget),
                },
            ),
            patch("facial_attendance.ui.webcam_app.CameraFeed"),
            patch("facial_attendance.ui.webcam_app.SingleWorker"),
        ):
            app = LiveWebcamApp(Mock(), Mock(), source_name="Mock webcam")
        ttk.Combobox.assert_not_called()
        texts = [entry.get("text", "") for entry in widgets]
        self.assertIn("DEMO / SIMULATED OUTPUT", texts)
        self.assertFalse(any(text in {"Freeze frame", "Run stubs"} for text in texts))
        self.assertFalse(app._debug_visible)
        self.assertTrue(app._preview_mirrored)
        self.assertEqual(app.controller.action, "attendance")

    def test_cli_selects_automatic_app_by_default_and_manual_only_by_explicit_flag(
        self,
    ):
        cv2 = SimpleNamespace(VideoCapture=Mock())
        tk = SimpleNamespace(Tk=Mock(), TclError=RuntimeError)
        with (
            patch.dict(sys.modules, {"cv2": cv2, "tkinter": tk, "PIL.ImageTk": Mock()}),
            patch("facial_attendance.ui.webcam_app.LiveWebcamApp") as automatic,
            patch("facial_attendance.ui.webcam_app.WebcamApp") as manual,
        ):
            self.assertEqual(main([]), 0)
            automatic.assert_called_once()
            manual.assert_not_called()
            tk.Tk.return_value.after_idle.assert_called_with(
                automatic.return_value.start
            )
            self.assertEqual(
                main(
                    [
                        "--dev-manual",
                        "--dev-action",
                        "registration",
                        "--dev-scenario",
                        "Spoof",
                    ]
                ),
                0,
            )
            self.assertEqual(manual.call_args.kwargs["action"], "registration")
            self.assertEqual(manual.call_args.kwargs["scenario"], "Spoof")


class WorkerTests(unittest.TestCase):
    def test_single_worker_rejects_backlog_and_runs_off_the_caller_thread(self):
        entered, finish = Event(), Event()
        called_on = []

        def operation(value):
            called_on.append(get_ident())
            entered.set()
            finish.wait(2)
            return value * 2

        worker = SingleWorker(operation, "sys05-test")
        try:
            self.assertTrue(worker.submit((1, 1), 4))
            self.assertTrue(entered.wait(2))
            for i in range(100):
                self.assertFalse(worker.submit((1, i + 2), i))
            self.assertTrue(worker.busy)
            self.assertNotEqual(called_on, [get_ident()])
            finish.set()
            # Condition wait is event-driven and synchronizes publication.
            with worker._condition:
                self.assertTrue(
                    worker._condition.wait_for(
                        lambda: worker._result is not None, timeout=2
                    )
                )
            self.assertEqual(worker.take(), WorkResult((1, 1), 8))
            self.assertFalse(worker.busy)
        finally:
            finish.set()
            worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertFalse(worker.submit((2, 1), 1))

    def test_worker_exception_is_returned_and_shutdown_discards_late_output(self):
        entered, finish = Event(), Event()

        def operation(value):
            entered.set()
            finish.wait(2)
            raise ValueError("failure")

        worker = SingleWorker(operation, "sys05-test-error")
        try:
            worker.submit((1, 1), None)
            self.assertTrue(entered.wait(2))
            worker.close()  # Must return even while the adapter is blocked.
            finish.set()
            worker.thread.join(2)
            self.assertIsNone(worker.take())
            self.assertFalse(worker.thread.is_alive())
        finally:
            finish.set()
            worker.close()

    def test_worker_exception_reaches_the_result_slot(self):
        def broken(value):
            raise ValueError("failure")

        worker = SingleWorker(broken, "sys05-test-exception")
        try:
            worker.submit((1, 1), None)
            with worker._condition:
                self.assertTrue(
                    worker._condition.wait_for(
                        lambda: worker._result is not None, timeout=2
                    )
                )
            result = worker.take()
            self.assertEqual(result.token, (1, 1))
            self.assertIsInstance(result.error, ValueError)
            self.assertFalse(worker.busy)
        finally:
            worker.close()

    def test_camera_publishes_owned_readonly_frames_and_releases_on_shutdown(self):
        published, continue_read = Event(), Event()
        original = Frame()
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = True, original
        release_thread = []
        camera.release.side_effect = lambda: release_thread.append(get_ident())
        feed = CameraFeed(lambda: camera)
        publish = feed._publish

        def observe(packet):
            publish(packet)
            published.set()
            continue_read.wait(2)

        feed._publish = observe
        try:
            generation = feed.restart()
            self.assertTrue(published.wait(2))
            packet = feed.latest()
            self.assertEqual(packet.generation, generation)
            self.assertEqual(packet.sequence, 1)
            self.assertIsNot(packet.frame, original)
            self.assertFalse(packet.frame.flags.writeable)
            continue_read.set()
            feed.close()
            feed.thread.join(2)
            self.assertEqual(release_thread, [feed.thread.ident])
            self.assertIsNone(feed.latest())
        finally:
            continue_read.set()
            feed.close()

    def test_camera_failure_releases_on_owner_thread_and_retry_recovers(self):
        failed = Mock()
        failed.isOpened.return_value = False
        released = Event()
        failed.release.side_effect = released.set
        delivered, unblock = Event(), Event()
        camera = Mock()
        camera.isOpened.return_value = True

        def read():
            delivered.set()
            unblock.wait(2)
            return True, Frame()

        camera.read.side_effect = read
        camera_threads = []
        cameras = iter((failed, camera))

        def open_camera():
            camera_threads.append(get_ident())
            return next(cameras)

        feed = CameraFeed(open_camera)
        try:
            first = feed.restart()
            self.assertTrue(released.wait(2))
            self.assertEqual(feed.latest().generation, first)
            self.assertIsNotNone(feed.latest().error)
            second = feed.restart()
            self.assertTrue(delivered.wait(2))
            self.assertIsNone(feed.latest())
            # Retry while read is blocked: the old read cannot republish.
            third = feed.restart()
            unblock.set()
            feed.close()
            feed.thread.join(2)
            self.assertGreater(third, second)
            self.assertIsNone(feed.latest())
            camera.release.assert_called_once()
            self.assertTrue(all(thread != get_ident() for thread in camera_threads))
        finally:
            unblock.set()
            feed.close()


if __name__ == "__main__":
    unittest.main()
