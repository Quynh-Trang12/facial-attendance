"""Deterministic CPU-detector, display-stability and instrumentation regressions."""

from dataclasses import asdict
import io
import sys
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from facial_attendance.ui.diagnostics import Diagnostics
from facial_attendance.ui.face_detection import (
    FaceDetectorUnavailable, ScaledFaceDetector, YuNetFaceDetector, YUNET_SHA256, bounded_size, map_box,
)
from facial_attendance.ui.live_preview import (
    CameraFeed, LiveController, WorkResult, evaluate_adapters, prepare_preview_image,
)
from facial_attendance.ui.webcam_prototype import StubAdapters
import test_webcam_live as legacy


class ScaledDetectorTests(unittest.TestCase):
    def setUp(self):
        self.cv2 = SimpleNamespace(INTER_AREA=3, resize=Mock())
        self.enterContext(patch.dict(sys.modules, {"cv2": self.cv2}))

    def test_downscales_once_and_maps_outward_with_actual_non_integer_ratios(self):
        frame = legacy.Frame(853, 481)
        self.cv2.resize.return_value = legacy.Frame(640, 361)
        detector = Mock(return_value=[(7, 9, 100, 150), (600, 300, 640, 361)])
        result = ScaledFaceDetector(detector)(frame)
        self.cv2.resize.assert_called_once_with(frame, (640, 361), interpolation=3)
        self.assertEqual(result, [(9, 11, 134, 200), (799, 399, 853, 481)])
        detector.assert_called_once_with(self.cv2.resize.return_value)
        self.assertEqual(frame.shape, (481, 853, 3))

    def test_small_source_is_not_enlarged_and_no_face_is_preserved(self):
        frame = legacy.Frame(320, 240)
        detector = Mock(return_value=[])
        self.assertEqual(ScaledFaceDetector(detector)(frame), [])
        self.cv2.resize.assert_not_called()
        detector.assert_called_once_with(frame)

    def test_portrait_and_extreme_aspect_ratio_dimensions_are_nonempty(self):
        self.assertEqual(bounded_size(481, 853), (361, 640))
        self.assertEqual(bounded_size(1, 10000), (1, 640))
        self.assertEqual(map_box((0, 0, 1, 640), (1, 10000), (1, 640)), (0, 0, 1, 10000))

    def test_invalid_regions_are_rejected_before_scaling(self):
        for box in ((-1, 0, 20, 30), (0, 0, 641, 20), (0, 0, True, 3), (0, 0, 2.5, 3)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                map_box(box, (1920, 1080), (640, 360))

    def test_preview_resizes_bgr_before_conversion(self):
        image = SimpleNamespace(size=(640, 360))
        cv2 = SimpleNamespace(INTER_AREA=3, COLOR_BGR2RGB=4, resize=Mock(return_value="small-bgr"),
                              cvtColor=Mock(return_value="small-rgb"))
        pil = SimpleNamespace(fromarray=Mock(return_value=image))
        frame = legacy.Frame(1920, 1080)
        with patch.dict(sys.modules, {"cv2": cv2}), patch("PIL.Image.fromarray", pil.fromarray):
            self.assertIs(prepare_preview_image(frame), image)
        cv2.resize.assert_called_once_with(frame, (640, 360), interpolation=3)
        cv2.cvtColor.assert_called_once_with("small-bgr", 4)
        pil.fromarray.assert_called_once_with("small-rgb")


class YuNetTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.detect.return_value = 1, [self.face()]
        self.cv2 = SimpleNamespace(
            __version__="4.13.0", FaceDetectorYN=SimpleNamespace(create=Mock(return_value=self.model)),
            dnn=SimpleNamespace(DNN_BACKEND_OPENCV=3, DNN_TARGET_CPU=0),
            error=type("CVError", (Exception,), {}),
        )
        self.enterContext(patch.dict(sys.modules, {"cv2": self.cv2}))
        self.read = self.enterContext(patch("facial_attendance.ui.face_detection.Path.read_bytes", return_value=b"fixture"))
        self.digest = self.enterContext(patch("facial_attendance.ui.face_detection.sha256"))
        self.digest.return_value.hexdigest.return_value = YUNET_SHA256
        self.detector = YuNetFaceDetector("fixture.onnx")
        self.frame = legacy.Frame(640, 360)

    @staticmethod
    def face(x=10.2, y=20.8, w=100.1, h=90.1, score=.99):
        return [x, y, w, h, *([0.0] * 10), score]

    def test_hash_checked_cpu_api_loaded_once_and_half_open_boxes_rounded_outward(self):
        self.assertEqual(self.detector(self.frame), [(10, 20, 111, 111)])
        self.detector(self.frame)
        self.read.assert_called_once()
        self.cv2.FaceDetectorYN.create.assert_called_once_with("fixture.onnx", "", (640, 360), .9, .3, 5000, 3, 0)
        self.model.setInputSize.assert_called_with((640, 360))

    def test_no_face_multiple_faces_and_partial_edge_box(self):
        self.model.detect.return_value = 1, None
        self.assertEqual(self.detector(self.frame), [])
        self.model.detect.return_value = 1, [self.face(x=-2), self.face(x=400)]
        self.assertEqual(self.detector(self.frame), [(0, 20, 99, 111), (400, 20, 501, 111)])

    def test_corrupt_missing_model_or_incompatible_api_fail_without_fallback_or_network(self):
        self.digest.return_value.hexdigest.return_value = "wrong"
        with self.assertRaisesRegex(FaceDetectorUnavailable, "checksum"):
            self.detector(self.frame)
        self.read.side_effect = OSError("missing")
        with self.assertRaisesRegex(FaceDetectorUnavailable, "missing"):
            self.detector(self.frame)
        self.cv2.__version__ = "5.0.0"
        with self.assertRaisesRegex(FaceDetectorUnavailable, "OpenCV"):
            self.detector(self.frame)
        self.cv2.FaceDetectorYN.create.assert_not_called()

    def test_one_bad_row_blocks_whole_batch(self):
        for bad in (self.face(w=0), self.face(x=900), self.face(score=float("nan")),
                    self.face(score=1.1), self.face(score=.5), self.face(x=True), [1, 2, 3]):
            with self.subTest(bad=bad):
                self.model.detect.return_value = 1, [self.face(), bad]
                with self.assertRaises(ValueError):
                    self.detector(self.frame)

    def test_load_and_inference_failures_are_reported(self):
        self.cv2.FaceDetectorYN.create.side_effect = self.cv2.error("invalid ONNX")
        with self.assertRaises(FaceDetectorUnavailable):
            self.detector(self.frame)
        self.cv2.FaceDetectorYN.create.side_effect = None
        self.model.detect.side_effect = self.cv2.error("inference failed")
        with self.assertRaises(self.cv2.error):
            self.detector(self.frame)


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.live = LiveController()
        self.frame = legacy.Frame()

    def detect(self, sequence, timestamp, boxes):
        token = 1, sequence
        self.live.begin_sample(token, timestamp)
        return self.live.detected(WorkResult(token, boxes), self.frame)

    def test_smoothing_only_changes_display_not_request_or_frame_association(self):
        self.detect(1, 10, [legacy.BOX])
        raw = (110, 100, 310, 250)
        request = self.detect(2, 10.3, [raw])
        boxes, _, unconfirmed = self.live.visual_regions(10.31)
        self.assertEqual(boxes, [(107, 100, 307, 250)])
        self.assertEqual(request.bbox, raw)
        self.assertIs(request.frame, self.frame)
        self.assertFalse(unconfirmed)

    def test_one_miss_holds_only_unconfirmed_outline_and_clears_every_model_field(self):
        request = self.detect(1, 10, [legacy.BOX])
        batch = evaluate_adapters((request, StubAdapters("Live match")))
        self.live.completed(WorkResult((1, 1), batch))
        self.detect(2, 10.3, [])
        self.assertEqual(self.live.controller.state.ui_state, "no_face")
        self.assertEqual(self.live.visual_regions(10.31), ([legacy.BOX], (1280, 720), True))
        state = asdict(self.live.controller.state)
        self.assertTrue(all(state[name] is None for name in ("recognition", "liveness", "emotion")))
        self.assertFalse(state["can_save"])
        self.assertFalse(self.live.completed(WorkResult((1, 1), batch)))
        self.assertEqual(self.live.visual_regions(10.451), ([], None, False))

    def test_multiple_faces_errors_and_reset_clear_smoothing_history(self):
        self.detect(1, 10, [legacy.BOX])
        self.detect(2, 10.3, [legacy.BOX, (500, 100, 600, 200)])
        self.assertEqual(self.live.controller.state.ui_state, "multiple_faces")
        self.assertIsNone(self.live.stability.box)
        self.assertIsNone(self.detect(3, 10.4, []))
        self.assertEqual(self.live.visual_regions(10.41), ([], None, False))
        for invalidate in (self.live.reset, lambda: self.live.fail("camera_unavailable", "Camera failure")):
            self.detect(4, 10.5, [legacy.BOX])
            invalidate()
            self.assertEqual(self.live.visual_regions(10.51), ([], None, False))


class SchedulingTests(unittest.TestCase):
    setUp = legacy.LiveAppTests.setUp
    packet = legacy.LiveAppTests.packet
    run_one_face = legacy.LiveAppTests.run_one_face

    def test_detector_throttles_and_chooses_latest_frame_after_busy_work(self):
        app = self.app
        app._tick()
        app.detector.finish([])
        app._tick()
        for sequence in range(2, 8):
            self.now = 10 + sequence * .03
            self.packet(sequence)
            app._tick()
            self.assertFalse(app.detector.busy)
        self.now = 10.31
        newest = self.packet(20)
        app._tick()
        self.assertEqual(app.detector.job, ((1, 20), newest.frame))
        self.assertTrue(app.detector.busy)

    def test_gui_budget_subtracts_rendering_time_and_counts_dropped_preview_frames(self):
        app = self.app
        app.metrics = Diagnostics(True, clock=lambda: self.now)
        def draw(frame):
            self.now += .015
        app._draw_frame.side_effect = draw
        self.packet(10)
        app._tick()
        self.assertEqual(app.root.after.call_args.args[0], 18)
        self.now += .1
        self.packet(15)
        app._tick()
        report = app.metrics.snapshot(force=True)
        self.assertEqual(report["counts"]["preview_dropped"], 13)
        self.assertEqual(report["counts"]["previewed"], 2)

    def test_hidden_json_is_not_serialized_and_opening_debug_refreshes_it(self):
        app = self.app
        app._debug_visible = False
        app._render()
        app.payload.insert.assert_not_called()
        app.debug = Mock()
        app._toggle_debug()
        app.debug.pack.assert_called_once()
        app.payload.insert.assert_called_once()

    def test_unchanged_regions_are_not_recreated_each_gui_tick(self):
        self.run_one_face()
        app = self.app
        app.canvas.create_rectangle.reset_mock()
        for _ in range(10):
            app._tick()
        app.canvas.create_rectangle.assert_not_called()

    def test_processing_outline_is_unconfirmed_and_model_results_are_cleared(self):
        self.run_one_face()
        app = self.app
        self.now += .31
        self.packet(2)
        app._tick()
        self.assertIsNone(app.controller.state.recognition)
        self.assertIsNone(app.controller.state.liveness)
        self.assertIsNone(app.controller.state.emotion)
        self.assertEqual(app.canvas.create_rectangle.call_args.kwargs["dash"], (4, 4))

    def test_expiry_is_checked_again_after_a_slow_tk_render(self):
        app = self.app
        app._tick()
        app.detector.finish([legacy.BOX])
        app._tick()
        app.inference.finish(evaluate_adapters(app.inference.job[1]))
        self.packet(2)
        def delayed_render(frame):
            self.now += 1.01
        app._draw_frame.side_effect = delayed_render
        app._tick()
        self.assertEqual(app.controller.state.errors[0].code, "camera_unavailable")
        self.assertIsNone(app.controller.state.recognition)
        self.assertEqual(app.live.visual_regions(self.now), ([], None, False))


class DiagnosticsTests(unittest.TestCase):
    def test_disabled_instrumentation_and_rate_limited_aggregate_output(self):
        disabled = Diagnostics()
        disabled.count("captured")
        self.assertIsNone(disabled.snapshot(force=True))
        now = [0.0]
        metrics = Diagnostics(True, clock=lambda: now[0])
        metrics.count("captured", 150)
        metrics.count("previewed", 100)
        metrics.duration("detector", .01)
        metrics.duration("detector", .03)
        now[0] = 4.9
        self.assertIsNone(metrics.snapshot())
        now[0] = 5
        with patch("sys.stderr", new_callable=io.StringIO) as stream:
            report = metrics.report()
        self.assertEqual(report["capture_fps"], 30)
        self.assertEqual(report["preview_fps"], 20)
        self.assertEqual(report["timings_ms"]["detector"], {"mean": 20, "max": 30, "n": 2})
        self.assertEqual(stream.getvalue().count("SYS05_PERF"), 1)
        self.assertIsNone(metrics.report())

    def test_camera_pacing_subtracts_blocking_read_and_copy_time(self):
        feed = CameraFeed.__new__(CameraFeed)
        feed.open_camera = Mock()
        camera = feed.open_camera.return_value
        camera.read.return_value = True, legacy.Frame()
        feed.metrics = Diagnostics()
        feed._wake = Mock()
        feed._stop = Mock()
        feed._stop.is_set.side_effect = [False, False, True]
        feed._lock = Lock()
        feed._generation = 1
        feed._current = Mock(side_effect=[True, False])
        feed._publish = Mock()
        with patch("facial_attendance.ui.live_preview.monotonic", side_effect=[0, .020, .022, .023]):
            feed._run()
        self.assertAlmostEqual(feed._stop.wait.call_args.args[0], 1 / 30 - .023)
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
