"""SYS-05 contract tests without camera hardware or optional ML packages."""

from dataclasses import asdict, replace
import json
import unittest

from facial_attendance.ui.webcam_prototype import (
    Completion,
    Emotion,
    Liveness,
    PrototypeController,
    StubAdapters,
    run_stubs,
)


class FrameFixture:
    shape = (80, 100, 3)
    dtype = "uint8"

    def __getitem__(self, indexes):
        return indexes


class WebcamPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.controller = PrototypeController()
        self.frame = FrameFixture()
        self.box = (10, 20, 50, 60)

    def capture(self):
        request = self.controller.begin_capture(self.frame, [self.box])
        self.assertIsNotNone(request)
        return request

    def test_crop_uses_half_open_original_frame_coordinates(self):
        request = self.capture()
        self.assertIs(request.frame, self.frame)
        self.assertEqual(request.face_crop, (slice(20, 60), slice(10, 50)))
        self.assertEqual(request.context.contract_version, "sys02-v1")
        self.assertEqual(request.context.action, "attendance")

    def test_invalid_frame_and_box_do_not_dispatch(self):
        for box in ((-1, 0, 4, 5), (0, 0, 101, 80), (4, 0, 4, 5), (True, 0, 4, 5)):
            with self.subTest(box=box):
                self.assertIsNone(self.controller.begin_capture(self.frame, [box]))
                self.assertEqual(self.controller.state.errors[0].code, "invalid_input")
        for shape, dtype in (((80, 100), "uint8"), ((80, 100, 4), "uint8"),
                             ((0, 100, 3), "uint8"), ((80, 100, 3), "float32")):
            with self.subTest(shape=shape, dtype=dtype):
                self.frame.shape, self.frame.dtype = shape, dtype
                self.assertIsNone(self.controller.begin_capture(self.frame, [self.box]))
                self.assertEqual(self.controller.state.errors[0].code, "invalid_input")

    def test_live_scenarios_serialize_contract_and_never_allow_save(self):
        for scenario, accepted, identity, class_id in (
            ("Live match", True, "Demo Person", 2), ("Live unknown", False, "Unknown", 1)
        ):
            for action in ("attendance", "registration"):
                with self.subTest(scenario=scenario, action=action):
                    self.controller.reset(action)
                    request = self.capture()
                    run_stubs(self.controller, request, StubAdapters(scenario))
                    payload = json.loads(json.dumps(asdict(self.controller.state), allow_nan=False))
                    self.assertEqual(payload["context"]["action"], action)
                    self.assertEqual(payload["liveness"]["source"], "stub")
                    self.assertEqual(payload["recognition"]["accepted"], accepted)
                    self.assertIsNone(payload["recognition"]["model_version"])
                    self.assertEqual(payload["emotion"]["class_id"], class_id)
                    self.assertEqual(self.controller.demo_identity, identity)
                    self.assertFalse(payload["can_save"])

    def test_face_loss_multiple_regions_and_camera_error_clear_results(self):
        for boxes, code in (([], "no_face"), ([self.box, self.box], "multiple_faces"),
                            (None, "camera_unavailable")):
            with self.subTest(code=code):
                request = self.capture()
                adapters = StubAdapters("Live match")
                run_stubs(self.controller, request, adapters)
                if boxes is None:
                    self.controller.capture_error(code, "Camera read failed.")
                else:
                    self.assertIsNone(self.controller.begin_capture(self.frame, boxes))
                self.assertEqual(self.controller.state.errors[0].code, code)
                self.assertIsNone(self.controller.state.context)
                self.assertIsNone(self.controller.state.emotion)
                self.assertIsNone(self.controller.state.recognition)
                self.assertIsNone(self.controller.state.liveness)
                self.assertFalse(self.controller.receive(adapters.liveness(request)))
                self.assertFalse(self.controller.state.can_save)

    def test_result_from_wrong_context_is_discarded(self):
        request = self.capture()
        adapters = StubAdapters("Live match")
        for field, other in (
            ("contract_version", "old"), ("attempt_id", "old"), ("action", "registration"),
            ("capture_id", "old"), ("face_id", "old"),
        ):
            with self.subTest(field=field):
                result = adapters.liveness(request)
                mismatched = replace(result, context=replace(result.context, **{field: other}))
                self.assertFalse(self.controller.receive(mismatched))
                self.assertEqual(self.controller.state.ui_state, "checking")
        second = self.capture()
        self.assertNotEqual(second.context.capture_id, request.context.capture_id)
        self.assertFalse(self.controller.receive(adapters.liveness(request)))
        self.controller.reset("registration")
        self.capture()
        self.assertFalse(self.controller.receive(adapters.liveness(second)))

    def test_provisional_identity_clears_on_spoof_and_requires_retry(self):
        request = self.capture()
        match = StubAdapters("Live match")
        self.assertTrue(self.controller.receive(match.recognition(request)))
        self.assertEqual(self.controller.demo_identity, "Unavailable")
        self.assertTrue(self.controller.receive(StubAdapters("Spoof").liveness(request)))
        self.assertIsNone(self.controller.state.recognition)
        self.assertIsNone(self.controller.state.emotion)
        self.assertFalse(self.controller.receive(match.liveness(request)))
        self.assertIsNone(self.controller.begin_capture(self.frame, [self.box]))
        self.controller.reset()
        retry = self.capture()
        self.assertNotEqual(retry.context.attempt_id, request.context.attempt_id)
        run_stubs(self.controller, retry, match)
        self.assertEqual(self.controller.demo_identity, "Demo Person")
        self.assertFalse(self.controller.state.can_save)

    def test_uncertain_skips_recognition_and_rejects_previous_capture(self):
        request = self.capture()
        run_stubs(self.controller, request, StubAdapters("Uncertain"))
        self.assertEqual(self.controller.state.ui_state, "uncertain")
        self.assertIsNone(self.controller.state.recognition)
        self.assertEqual(self.controller.state.emotion.label, "Negative")
        self.capture()
        self.assertFalse(self.controller.receive(StubAdapters("Live match").liveness(request)))

    def test_bad_liveness_is_an_operational_error(self):
        for result in (Liveness("live", "demo_result", float("nan")),
                       Liveness("live", "demo_result", None),
                       Liveness("live", "demo_result", 1.1),
                       Liveness("live", "accepted_live", 0.08, source="model")):
            with self.subTest(result=result):
                request = self.capture()
                self.controller.receive(Completion(request.context, result))
                self.assertEqual(self.controller.state.ui_state, "error")
                self.assertIsNone(self.controller.state.liveness)
                self.assertTrue(self.controller.state.errors[0].blocks_persistence)

    def test_invalid_emotion_clears_display_only(self):
        request = self.capture()
        run_stubs(self.controller, request, StubAdapters("Live match"))
        for emotion in (Emotion(2, "Neutral"), Emotion(3, "Surprise"),
                        Emotion(2, "Positive", float("nan")), Emotion(2, "Positive", 1.1)):
            with self.subTest(emotion=emotion):
                self.controller.receive(Completion(request.context, emotion))
                self.assertIsNone(self.controller.state.emotion)
                self.assertEqual(self.controller.demo_identity, "Demo Person")
                self.assertFalse(self.controller.state.errors[-1].blocks_persistence)

    def test_adapter_exception_clears_payload_and_reports_typed_error(self):
        class FailingLiveness(StubAdapters):
            def liveness(self, request):
                raise RuntimeError("example adapter failure")

        request = self.capture()
        run_stubs(self.controller, request, FailingLiveness("Live match"))
        self.assertEqual(self.controller.state.ui_state, "error")
        self.assertIsNone(self.controller.state.recognition)
        self.assertIsNone(self.controller.state.liveness)
        error = self.controller.state.errors[0]
        self.assertEqual((error.module, error.code), ("liveness", "inference_error"))
        self.assertTrue(error.blocks_persistence)


if __name__ == "__main__":
    unittest.main()
