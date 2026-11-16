"""Owner-side SYS-02 audit regressions; no physical camera or trained adapters."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest

from facial_attendance.ui.webcam_prototype import (
    Completion, Emotion, FaceRequest, Liveness, OperationError, PrototypeController,
    Recognition, SCENARIOS, StubAdapters, frame_size, run_stubs,
)
from facial_attendance.ui.live_preview import LiveController, WorkResult, evaluate_adapters
import test_webcam_live as legacy


class ContractAuditTests(unittest.TestCase):
    def setUp(self):
        self.controller = PrototypeController()
        self.frame = legacy.Frame(100, 80)
        self.request = self.controller.begin_capture(self.frame, [(10, 20, 50, 60)])

    def test_orphan_and_untyped_completions_are_rejected_without_active_capture(self):
        self.controller.reset()
        self.assertFalse(self.controller.receive(Completion(None, Liveness("live", "demo_result", .1))))
        self.assertFalse(self.controller.receive(None))
        self.assertEqual(self.controller.state.ui_state, "idle")
        self.assertIsNone(self.controller.state.liveness)

    def test_malformed_frames_and_regions_produce_typed_invalid_input(self):
        for frame in (None, SimpleNamespace(shape=None), SimpleNamespace(shape=(80, 100, 3)),
                      SimpleNamespace(shape=(True, 100, 3), dtype="uint8")):
            with self.subTest(frame=frame):
                self.assertIsNone(self.controller.begin_capture(frame, [(0, 0, 10, 10)]))
                self.assertEqual(self.controller.state.errors[0].code, "invalid_input")
        for box in (None, 1, "1234", [1, 2], [1, 2, 3, object()]):
            with self.subTest(box=box):
                self.assertIsNone(self.controller.begin_capture(self.frame, [box]))
                self.assertEqual(self.controller.state.errors[0].code, "invalid_input")

    def test_unknown_action_is_rejected(self):
        for action in ("save", True, {}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.controller.reset(action)

    def test_malformed_adapter_response_is_typed_and_emotion_failure_is_display_only(self):
        for module in ("liveness", "recognition", "emotion"):
            with self.subTest(module=module):
                self.controller.reset()
                request = self.controller.begin_capture(self.frame, [(10, 20, 50, 60)])
                adapters = StubAdapters("Live match")
                setattr(adapters, module, lambda request: None)
                run_stubs(self.controller, request, adapters)
                error = self.controller.state.errors[0]
                self.assertEqual((error.module, error.code), (module, "invalid_input"))
                self.assertEqual(error.blocks_persistence, module != "emotion")
                self.assertIsNone(self.controller.state.emotion)
                self.assertEqual(self.controller.demo_identity, "Demo Person" if module == "emotion" else "Unavailable")
                self.assertFalse(self.controller.state.can_save)

    def test_wrong_result_type_cannot_masquerade_as_another_adapter(self):
        adapters = StubAdapters("Live match")
        adapters.recognition = lambda request: Completion(request.context, Emotion(2, "Positive"))
        run_stubs(self.controller, self.request, adapters)
        self.assertEqual(self.controller.state.errors[0].module, "recognition")
        self.assertIsNone(self.controller.state.recognition)
        self.assertIsNone(self.controller.state.emotion)

    def test_stale_adapter_response_is_discarded_without_relabelling_context(self):
        old = self.request
        self.controller.reset()
        current = self.controller.begin_capture(self.frame, [(10, 20, 50, 60)])
        adapters = StubAdapters("Live match")
        adapters.liveness = lambda request: Completion(old.context, Liveness("live", "demo_result", .1))
        self.assertEqual(run_stubs(self.controller, current, adapters), ())
        self.assertEqual(self.controller.state.context, current.context)
        self.assertIsNone(self.controller.state.liveness)

    def test_error_severity_is_owned_by_controller(self):
        for module in ("liveness", "recognition", "emotion"):
            self.controller.reset()
            request = self.controller.begin_capture(self.frame, [(10, 20, 50, 60)])
            run_stubs(self.controller, request, StubAdapters("Live match"))
            error = OperationError(module, "inference_error", "Controlled failure", module == "emotion")
            self.assertTrue(self.controller.receive(Completion(request.context, error)))
            self.assertEqual(self.controller.state.errors[-1].blocks_persistence, module != "emotion")
            self.assertEqual(self.controller.demo_identity, "Demo Person" if module == "emotion" else "Unavailable")

    def test_uncertain_liveness_clears_a_provisional_identity(self):
        self.controller.receive(Completion(self.request.context, Recognition("demo", "Demo Person", .9, True)))
        self.controller.receive(Completion(self.request.context, Liveness("uncertain", "demo_result", .45)))
        self.assertIsNone(self.controller.state.recognition)
        self.assertEqual(self.controller.demo_identity, "Unavailable")

    def test_malformed_current_batch_fails_closed(self):
        live = LiveController()
        live.begin_sample((1, 1), 1.0)
        live.detected(WorkResult((1, 1), [(10, 20, 50, 60)]), self.frame)
        self.assertFalse(live.completed(WorkResult((1, 1), None)))
        self.assertEqual(live.controller.state.errors[0].code, "invalid_input")

    def test_distinct_bgr_pixels_and_exact_half_open_crop_are_preserved(self):
        import numpy as np
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (11, 22, 33)
        frame.flags.writeable = False
        request = self.controller.begin_capture(frame, [(10, 20, 50, 60)])
        self.assertEqual(frame_size(request.frame), (100, 80))
        self.assertTrue(np.shares_memory(request.face_crop, frame))
        self.assertEqual(request.face_crop.shape, (40, 40, 3))
        self.assertEqual(request.face_crop[0, 0].tolist(), [11, 22, 33])
        self.assertFalse(request.face_crop.flags.writeable)
        for scenario in SCENARIOS:
            local = PrototypeController()
            local.state = self.controller.state
            completions = evaluate_adapters((request, StubAdapters(scenario)))
            self.assertTrue(all(item.context == request.context for item in completions))
            for completion in completions:
                local.receive(completion)
            self.assertFalse(asdict(local.state)["can_save"])


class SchedulingAuditTests(unittest.TestCase):
    setUp = legacy.LiveAppTests.setUp
    packet = legacy.LiveAppTests.packet

    def test_finished_adapter_slot_is_freed_before_new_detection_dispatch(self):
        app = self.app
        app._tick()
        app.detector.finish([legacy.BOX])
        app._tick()
        old_job = app.inference.job
        self.now += .31
        self.packet(2)
        app._tick()  # Next detector starts while previous adapter is busy.
        app.detector.finish([legacy.BOX])
        app.inference.finish(evaluate_adapters(old_job[1]))
        app._tick()  # Both results are ready in this same Tk tick.
        self.assertEqual(app.inference.job[0], (1, 2))
        self.assertTrue(app.inference.busy)
        self.assertIsNone(app.controller.state.recognition)
        app.inference.finish(evaluate_adapters(app.inference.job[1]))
        app._tick()
        self.assertEqual(app.controller.demo_identity, "Demo Person")


if __name__ == "__main__":
    unittest.main()
