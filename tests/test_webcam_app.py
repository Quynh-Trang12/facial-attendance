"""Exercise webcam GUI state transitions using fake widgets and camera frames."""

from dataclasses import asdict
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from facial_attendance.ui.face_detection import FaceDetectorUnavailable, HaarFaceDetector
from facial_attendance.ui.webcam_app import AUTOMATIC, MANUAL, WebcamApp
from facial_attendance.ui.webcam_prototype import Completion, PrototypeController, SCENARIOS


class Frame:
    dtype = "uint8"

    def __init__(self, width=1280, height=720):
        self.shape = height, width, 3
        self.flags = SimpleNamespace(writeable=True)
        self.slices = []

    def copy(self):
        return Frame(self.shape[1], self.shape[0])

    def __getitem__(self, selection):
        self.slices.append(selection)
        y, x = selection
        return Frame(x.stop - x.start, y.stop - y.start)


def variable(value):
    return Mock(**{"get.return_value": value})


def drag(app, start=(50, 50), end=(150, 125)):
    app._start_drag(SimpleNamespace(x=start[0], y=start[1]))
    app._end_drag(SimpleNamespace(x=end[0], y=end[1]))


class WebcamAppTests(unittest.TestCase):
    def setUp(self):
        app = self.app = WebcamApp.__new__(WebcamApp)
        app.root = Mock()
        app.open_camera = Mock()
        app.controller = PrototypeController()
        app.detect_faces = Mock(return_value=[(100, 100, 300, 250)])
        app.camera = Mock()
        app.latest = Frame()
        app.frozen = app.photo = None
        app.preview_size = (640, 360)
        app.boxes = []
        app.drag_start = app.drag_outline = None
        app.evaluated = False
        app.action = variable("attendance")
        app.scenario = variable("Live match")
        app.selection_mode = variable(MANUAL)
        app._timer = "timer"
        for name in ("canvas", "hint", "summary", "payload", "freeze_button", "run_button", "clear_button"):
            setattr(app, name, Mock())
        self.cv2_error = type("OpenCVError", (Exception,), {})
        self.enterContext(patch.dict(sys.modules, {"cv2": SimpleNamespace(error=self.cv2_error)}))

    def test_automatic_uses_frozen_source_and_draws_scaled_overlay(self):
        app = self.app
        app.selection_mode.get.return_value = AUTOMATIC
        app.freeze()
        self.assertIsNot(app.frozen, app.latest)
        self.assertFalse(app.frozen.flags.writeable)
        app.detect_faces.assert_called_once_with(app.frozen)
        self.assertEqual(app.boxes, [(100, 100, 300, 250)])
        app.canvas.create_rectangle.assert_called_once_with(
            50, 50, 150, 125, outline="#ffce65", width=2, tags="region"
        )
        drag(app)
        self.assertEqual(len(app.boxes), 1)
        app.evaluate()
        self.assertEqual(app.frozen.slices, [(slice(100, 250), slice(100, 300))])
        self.assertEqual(app.controller.demo_identity, "Demo Person")
        self.assertFalse(app.controller.state.can_save)

    def test_filtered_haar_boxes_drive_both_overlay_and_controller(self):
        app = self.app
        app.selection_mode.get.return_value = AUTOMATIC
        app.detect_faces = HaarFaceDetector()
        classifier = Mock()
        classifier.empty.return_value = False
        cv2 = sys.modules["cv2"]
        cv2.CascadeClassifier = Mock(return_value=classifier)
        cv2.data = SimpleNamespace(haarcascades="bundle")
        cv2.cvtColor = Mock()
        cv2.COLOR_BGR2GRAY = 6
        self.enterContext(patch("facial_attendance.ui.face_detection.Path.is_file", return_value=True))

        face = (100, 100, 200, 200)
        eye = (140, 140, 70, 60)
        classifier.detectMultiScale.return_value = [eye, face]
        app.freeze()
        self.assertEqual(app.boxes, [(100, 100, 300, 300)])
        app.canvas.create_rectangle.assert_called_once_with(
            50, 50, 150, 150, outline="#ffce65", width=2, tags="region"
        )
        app.evaluate()
        self.assertEqual(app.frozen.slices, [(slice(100, 300), slice(100, 300))])
        self.assertEqual(app.controller.state.ui_state, "live")
        self.assertFalse(app.controller.state.can_save)

        app.retry()
        app.latest = Frame()
        app.preview_size = (640, 360)
        app.canvas.create_rectangle.reset_mock()
        classifier.detectMultiScale.return_value = [eye, face, (500, 130, 120, 120)]
        app.freeze()
        self.assertEqual(app.boxes, [(100, 100, 300, 300), (500, 130, 620, 250)])
        self.assertEqual(app.canvas.create_rectangle.call_args_list, [
            call(50, 50, 150, 150, outline="#ffce65", width=2, tags="region"),
            call(250, 65, 310, 125, outline="#ffce65", width=2, tags="region"),
        ])
        app.evaluate()
        self.assertEqual(app.frozen.slices, [])
        self.assertEqual(app.controller.state.ui_state, "multiple_faces")
        self.assertFalse(app.controller.state.can_save)

    def test_manual_selection_scales_and_clamps_at_non_integer_ratios(self):
        app = self.app
        app.latest = Frame(853, 481)
        app.preview_size = (638, 360)
        app.freeze()
        drag(app, start=(7, 9), end=(1000, 500))
        self.assertEqual(app.boxes, [(9, 12, 853, 481)])
        app.detect_faces.assert_not_called()
        app.evaluate()
        self.assertEqual(app.frozen.slices, [(slice(12, 481), slice(9, 853))])

    def test_automatic_empty_and_multiple_faces_do_not_run_stubs(self):
        app = self.app
        app.selection_mode.get.return_value = AUTOMATIC
        for boxes, code in (([], "no_face"), ([(0, 0, 40, 40), (200, 200, 300, 300)], "multiple_faces")):
            with self.subTest(code=code):
                app.detect_faces.return_value = boxes
                app.freeze()
                app.evaluate()
                state = app.controller.state
                self.assertEqual(state.ui_state, code)
                self.assertEqual(state.errors[0].code, code)
                self.assertTrue(state.errors[0].blocks_persistence)
                self.assertIsNone(state.recognition)
                self.assertIsNone(state.liveness)
                self.assertIsNone(state.emotion)
                self.assertFalse(state.can_save)
                self.assertEqual(app.frozen.slices, [])
                app.retry()
                app.latest = Frame()
                app.preview_size = (640, 360)

    def test_invalid_detection_is_a_typed_error_and_manual_fallback_works(self):
        app = self.app
        app.selection_mode.get.return_value = AUTOMATIC
        for failure, code in (
            (FaceDetectorUnavailable("Missing Haar"), "model_unavailable"),
            (ValueError("Bad rectangle"), "invalid_input"),
            (self.cv2_error("Detector failed"), "inference_error"),
        ):
            with self.subTest(code=code):
                app.detect_faces.side_effect = failure
                app.freeze()
                frozen = app.frozen
                app.evaluate()
                self.assertEqual(app.controller.state.errors[0].module, "detector")
                self.assertEqual(app.controller.state.errors[0].code, code)
                self.assertEqual(frozen.slices, [])
                app.selection_mode.get.return_value = MANUAL
                app._change_selection_mode(None)
                self.assertIs(app.frozen, frozen)
                drag(app)
                app.evaluate()
                self.assertEqual(app.controller.demo_identity, "Demo Person")
                app.retry()
                app.latest = Frame()
                app.preview_size = (640, 360)
                app.selection_mode.get.return_value = AUTOMATIC

    def test_switching_modes_replaces_selection_and_clears_old_results(self):
        app = self.app
        app.freeze()
        drag(app)
        app.evaluate()
        old_context = app.controller.state.context
        app.selection_mode.get.return_value = AUTOMATIC
        app.detect_faces.return_value = [(200, 200, 400, 350)]
        app._change_selection_mode(None)
        self.assertEqual(app.boxes, [(200, 200, 400, 350)])
        self.assertIsNone(app.controller.state.recognition)
        self.assertIsNone(app.controller.state.context)
        app.evaluate()
        self.assertNotEqual(app.controller.state.context.attempt_id, old_context.attempt_id)
        app.selection_mode.get.return_value = MANUAL
        app._change_selection_mode(None)
        self.assertEqual(app.boxes, [])
        self.assertIsNone(app.controller.state.emotion)
        app.canvas.delete.assert_called_with("region")

    def test_clear_regions_invalidates_context_and_evaluation_returns_no_face(self):
        app = self.app
        app.freeze()
        drag(app)
        app.evaluate()
        context = app.controller.state.context
        app._clear_regions()
        self.assertFalse(app.evaluated)
        self.assertEqual(app.boxes, [])
        app.canvas.delete.assert_called_with("region")
        self.assertIsNone(app.controller.state.context)
        self.assertNotEqual(app.controller.attempt_id, context.attempt_id)
        app.evaluate()
        self.assertEqual(app.controller.state.ui_state, "no_face")

    def test_scenario_operation_and_retry_discard_old_capture(self):
        app = self.app
        for change in ("scenario", "action", "retry"):
            with self.subTest(change=change):
                app.latest = Frame()
                app.preview_size = (640, 360)
                app.freeze()
                drag(app)
                app.evaluate()
                self.assertEqual(app.controller.demo_identity, "Demo Person")
                old = app.controller.state
                late = Completion(old.context, old.recognition)
                if change == "scenario":
                    app.scenario.get.return_value = "Live unknown"
                    app._change_run_settings(None)
                    app.scenario.get.return_value = "Live match"
                elif change == "action":
                    app.action.get.return_value = "registration"
                    app._change_run_settings(None)
                    self.assertEqual(app.controller.action, "registration")
                else:
                    app.retry()
                self.assertIsNone(app.frozen)
                self.assertIsNone(app.controller.state.recognition)
                self.assertEqual(app.boxes, [])
                self.assertFalse(app.controller.receive(late))

    def test_all_scenarios_keep_save_disabled_in_both_modes(self):
        app = self.app
        for action in ("attendance", "registration"):
            for scenario in SCENARIOS:
                with self.subTest(action=action, scenario=scenario):
                    app.latest = Frame()
                    app.preview_size = (640, 360)
                    app.action.get.return_value = action
                    app.scenario.get.return_value = scenario
                    app.freeze()
                    drag(app)
                    app.evaluate()
                    payload = json.loads(app.payload.insert.call_args.args[1])
                    self.assertEqual(payload["context"]["action"], action)
                    self.assertFalse(payload["can_save"])
                    if scenario in {"Spoof", "Uncertain"}:
                        self.assertIsNone(payload["recognition"])
                        self.assertEqual(app.controller.demo_identity, "Unavailable")
                        app._clear_regions()
                        self.assertIsNone(app.frozen)
                    else:
                        app.retry()

    def test_camera_read_failure_clears_results_and_retry_reopens(self):
        app = self.app
        app.freeze()
        drag(app)
        app.evaluate()
        old_camera = app.camera
        # Simulate a backend read failure when preview resumes.
        app.frozen = None
        old_camera.read.return_value = (False, None)
        app._tick()
        old_camera.release.assert_called_once()
        self.assertIsNone(app.camera)
        self.assertIsNone(app.latest)
        self.assertEqual(app.boxes, [])
        self.assertIsNone(app.controller.state.recognition)
        self.assertEqual(app.controller.state.errors[0].code, "camera_unavailable")
        app.retry()
        self.assertIs(app.camera, app.open_camera.return_value)
        self.assertEqual(app.controller.state.ui_state, "idle")
        app.camera.read.return_value = (True, Frame())
        with patch.object(app, "_draw_frame") as draw:
            app._tick()
        draw.assert_called_once_with(app.latest)

    def test_failed_open_and_close_release_resources(self):
        app = self.app
        previous_camera = app.camera
        failed_camera = app.open_camera.return_value
        failed_camera.isOpened.return_value = False
        app.start()
        previous_camera.release.assert_called_once()
        failed_camera.release.assert_called_once()
        self.assertIsNone(app.camera)
        self.assertEqual(app.controller.state.errors[0].code, "camera_unavailable")
        failed_camera.isOpened.return_value = True
        app.retry()
        self.assertIs(app.camera, failed_camera)
        app.close()
        self.assertEqual(failed_camera.release.call_count, 2)
        app.root.after_cancel.assert_called_once_with("timer")
        app.root.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
