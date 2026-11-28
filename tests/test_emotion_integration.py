"""UI-04 Emotion tests: explicit fixture weights versus optional supplied best.pt."""

import io
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import get_ident
from unittest.mock import Mock, patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from facial_attendance.emotion.inference import (
    CHECKPOINT_SHA256,
    LABEL_MAPPING,
    EmotionCheckpointMismatch,
    EmotionInference,
    EmotionInferenceError,
    EmotionUnavailable,
    prediction_from_logits,
    preprocess_bgr_crop,
    validate_payload,
    weights_version,
)
from facial_attendance.ui.emotion_adapter import (
    MixedEmotionAdapters,
    TrainedEmotionAdapter,
)
from facial_attendance.ui.live_preview import (
    CameraPacket,
    LiveController,
    SingleWorker,
    WorkResult,
    evaluate_adapters,
)
from facial_attendance.ui.webcam_prototype import Completion, PrototypeController
from tests import test_webcam_live as legacy


@unittest.skipIf(torch is None, "Optional UI-04 PyTorch runtime is not installed")
class EmotionCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from facial_attendance.emotion.model import EmotionCNN

        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.fixture_model = EmotionCNN()
        state = cls.fixture_model.state_dict()
        cls.payload = {
            "meta": {
                "architecture": "EmotionCNN",
                "input_size": 48,
                "channels": 1,
                "scale": 255,
                "threshold": None,
                "threshold_source": "not_applicable_argmax",
                "label_mapping": LABEL_MAPPING.copy(),
                "model_id": "synthetic-test-fixture",
                "model_version": weights_version(state),
            },
            "state_dict": state,
            "config": {"model": {"channels": [32, 64, 128], "dropout": 0.3}},
            "epoch": 9,
        }
        data = io.BytesIO()
        torch.save(cls.payload, data)
        cls.data = data.getvalue()
        cls.directory = TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "fixture.pt"
        cls.path.write_bytes(cls.data)
        cls.digest = sha256(cls.data).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def engine(self):
        return EmotionInference(self.path, expected_sha256=self.digest)

    def request(self):
        frame = np.full((80, 100, 3), (11, 22, 33), dtype=np.uint8)
        frame.flags.writeable = False
        controller = PrototypeController()
        return controller, controller.begin_capture(frame, [(10, 20, 50, 60)])

    def mixed(self, scenario="Live match"):
        adapters = MixedEmotionAdapters(scenario, self.path)
        adapters.emotion_adapter = TrainedEmotionAdapter(
            self.path, inference=self.engine()
        )
        return adapters

    def test_default_hash_pin_rejects_fixture_before_deserialization(self):
        with patch("torch.load") as load, self.assertRaises(EmotionCheckpointMismatch):
            EmotionInference(self.path).load()
        load.assert_not_called()

    def test_load_is_cpu_weights_only_strict_eval_and_cached(self):
        engine = self.engine()
        with patch("torch.load", wraps=torch.load) as load:
            engine.load()
            engine.load()
        load.assert_called_once()
        self.assertEqual(
            load.call_args.kwargs, {"map_location": "cpu", "weights_only": True}
        )
        self.assertTrue(all(not module.training for module in engine.model.modules()))
        self.assertTrue(
            all(
                value.device.type == "cpu"
                for value in engine.model.state_dict().values()
            )
        )
        self.assertEqual(
            engine.provenance["model_version"],
            weights_version(engine.model.state_dict()),
        )

    def test_metadata_mapping_architecture_preprocessing_and_epoch_are_validated(self):
        cases = [
            ("architecture", "Other"),
            ("input_size", 64),
            ("channels", True),
            ("scale", 1),
            ("label_mapping", {"negative": 2, "neutral": 1, "positive": 0}),
            ("model_id", ""),
            ("model_version", "0" * 64),
            ("threshold_source", "guessed"),
        ]
        for key, value in cases:
            bad = deepcopy(self.payload)
            bad["meta"][key] = value
            with self.subTest(key=key), self.assertRaises(EmotionCheckpointMismatch):
                validate_payload(bad, self.fixture_model)
        for key, value in (
            ("epoch", 10),
            ("epoch", True),
            ("config", {"model": {"channels": [16, 32, 64], "dropout": 0.3}}),
        ):
            bad = deepcopy(self.payload)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(EmotionCheckpointMismatch):
                validate_payload(bad, self.fixture_model)

    def test_state_keys_shapes_dtypes_finiteness_and_version_are_validated(self):
        for mode in ("missing", "shape", "dtype", "nonfinite", "changed"):
            bad = deepcopy(self.payload)
            key = "features.0.weight"
            if mode == "missing":
                del bad["state_dict"][key]
            if mode == "shape":
                bad["state_dict"][key] = torch.zeros(1)
            if mode == "dtype":
                bad["state_dict"][key] = bad["state_dict"][key].double()
            if mode == "nonfinite":
                bad["state_dict"][key].fill_(float("nan"))
            if mode == "changed":
                bad["state_dict"][key].add_(0.1)
            with self.subTest(mode=mode), self.assertRaises(EmotionCheckpointMismatch):
                validate_payload(bad, self.fixture_model)

    def test_missing_file_has_explicit_model_unavailable_error(self):
        with self.assertRaises(EmotionUnavailable):
            EmotionInference(self.path.parent / "missing.pt").load()

    def test_preprocessing_uses_bgr_order_and_training_pillow_grayscale(self):
        crop = np.zeros((64, 32, 3), dtype=np.uint8)
        crop[:] = (255, 0, 0)
        original = crop.copy()
        crop.flags.writeable = False
        tensor = preprocess_bgr_crop(crop)
        self.assertEqual(tuple(tensor.shape), (1, 1, 48, 48))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(tensor.is_contiguous())
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 29 / 255, places=6)
        tensor.zero_()
        np.testing.assert_array_equal(crop, original)
        self.assertFalse(np.shares_memory(tensor.numpy(), crop))

    def test_noncontiguous_crop_and_full_value_range_are_supported(self):
        source = np.full((90, 100, 3), 255, dtype=np.uint8)
        tensor = preprocess_bgr_crop(source[::2, ::2])
        self.assertEqual(float(tensor.min()), 1.0)
        self.assertEqual(float(tensor.max()), 1.0)

    def test_invalid_crops_are_rejected(self):
        for invalid in (
            None,
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        ):
            with (
                self.subTest(type=type(invalid).__name__),
                self.assertRaises(ValueError),
            ):
                preprocess_bgr_crop(invalid)

    def test_softmax_mapping_is_explicit_and_not_an_accuracy_measure(self):
        for class_id, label in enumerate(("Negative", "Neutral", "Positive")):
            logits = torch.zeros(1, 3)
            logits[0, class_id] = 2
            prediction = prediction_from_logits(logits)
            self.assertEqual((prediction.class_id, prediction.label), (class_id, label))
            self.assertAlmostEqual(sum(prediction.probabilities), 1, places=6)
            self.assertGreater(prediction.confidence, 0.7)
        for invalid in (
            torch.zeros(3),
            torch.zeros(1, 4),
            torch.zeros(1, 3, dtype=torch.float64),
            torch.tensor([[0.0, float("nan"), 1.0]]),
        ):
            with (
                self.subTest(shape=tuple(invalid.shape)),
                self.assertRaises(EmotionInferenceError),
            ):
                prediction_from_logits(invalid)

    def test_model_rejects_wrong_input_dimensions(self):
        with self.assertRaises(ValueError):
            self.fixture_model(torch.zeros(1, 3, 48, 48))

    def test_adapter_emits_original_context_and_model_provenance_without_source_mutation(
        self,
    ):
        controller, request = self.request()
        before = request.frame.copy()
        completion = TrainedEmotionAdapter(self.path, inference=self.engine())(request)
        self.assertEqual(completion.context, request.context)
        self.assertEqual(completion.result.source, "model")
        self.assertEqual(completion.result.checkpoint_sha256, self.digest)
        self.assertTrue(controller.receive(completion))
        self.assertFalse(controller.state.can_save)
        np.testing.assert_array_equal(request.frame, before)

    def test_invalid_crop_and_version_are_context_bound_display_only_errors(self):
        _, request = self.request()
        adapter = TrainedEmotionAdapter(self.path, inference=self.engine())
        invalid = replace(request, bbox=(-1, 0, 10, 10))
        result = adapter(invalid)
        self.assertEqual(
            (
                result.result.code,
                result.result.module,
                result.result.blocks_persistence,
            ),
            ("invalid_input", "emotion", False),
        )
        wrong = replace(
            request, context=replace(request.context, contract_version="sys02-v2")
        )
        self.assertEqual(adapter(wrong).result.code, "version_mismatch")

    def test_model_error_clears_only_emotion_and_stub_paths_cannot_save(self):
        controller, request = self.request()
        adapters = self.mixed()
        adapters.emotion_adapter.inference.predict_bgr = Mock(
            side_effect=RuntimeError("controlled inference failure")
        )
        for completion in evaluate_adapters((request, adapters)):
            controller.receive(completion)
        self.assertEqual(controller.demo_identity, "Demo Person")
        self.assertIsNone(controller.state.emotion)
        self.assertEqual(controller.state.errors[0].code, "inference_error")
        self.assertFalse(controller.state.can_save)

    def test_spoof_skips_trained_emotion_and_uncertain_still_blocks_recognition(self):
        for scenario in ("Spoof", "Uncertain"):
            controller, request = self.request()
            adapters = self.mixed(scenario)
            with patch.object(
                adapters.emotion_adapter.inference,
                "predict_bgr",
                wraps=adapters.emotion_adapter.inference.predict_bgr,
            ) as predict:
                for completion in evaluate_adapters((request, adapters)):
                    controller.receive(completion)
            self.assertEqual(predict.call_count, 0 if scenario == "Spoof" else 1)
            self.assertIsNone(controller.state.recognition)
            self.assertFalse(controller.state.can_save)
            self.assertEqual(controller.state.emotion is None, scenario == "Spoof")

    def test_missing_trained_provenance_is_not_accepted_as_a_model_result(self):
        controller, request = self.request()
        result = TrainedEmotionAdapter(self.path, inference=self.engine())(request)
        controller.receive(
            Completion(request.context, replace(result.result, model_version=None))
        )
        self.assertIsNone(controller.state.emotion)
        self.assertEqual(controller.state.errors[0].module, "emotion")

    def test_reset_camera_error_and_face_loss_reject_trained_completions(self):
        for invalidate in (
            lambda live: live.reset(),
            lambda live: live.fail("camera_unavailable", "Camera lost"),
            lambda live: (
                live.begin_sample((1, 2), 10.3),
                live.detected(WorkResult((1, 2), []), self.request()[1].frame),
            ),
        ):
            controller, request = self.request()
            live = LiveController()
            live.begin_sample((1, 1), 10)
            active = live.detected(WorkResult((1, 1), [request.bbox]), request.frame)
            batch = evaluate_adapters((active, self.mixed()))
            self.assertTrue(live.completed(WorkResult((1, 1), batch)))
            invalidate(live)
            self.assertFalse(live.completed(WorkResult((1, 1), batch)))
            self.assertIsNone(live.controller.state.emotion)

    def test_real_model_work_runs_on_the_bounded_worker_thread(self):
        _, request = self.request()
        adapters = self.mixed()
        adapters.emotion_adapter.inference.load()
        threads = []
        hook = adapters.emotion_adapter.inference.model.register_forward_pre_hook(
            lambda module, inputs: threads.append(get_ident())
        )
        worker = SingleWorker(evaluate_adapters, "emotion-test")
        try:
            self.assertTrue(worker.submit((1, 1), (request, adapters)))
            with worker._condition:
                self.assertTrue(
                    worker._condition.wait_for(
                        lambda: worker._result is not None, timeout=5
                    )
                )
            result = worker.take()
            self.assertIsNone(result.error)
            self.assertEqual(result.value[-1].result.source, "model")
            self.assertNotEqual(threads, [get_ident()])
        finally:
            hook.remove()
            worker.close()
        self.assertFalse(worker.thread.is_alive())


@unittest.skipIf(torch is None, "Optional UI-04 PyTorch runtime is not installed")
class MixedDisplayTests(unittest.TestCase):
    setUp = legacy.LiveAppTests.setUp
    packet = legacy.LiveAppTests.packet

    def test_gui_labels_real_emotion_separately_from_simulated_identity_and_liveness(
        self,
    ):
        app = self.app
        frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
        app.feed.latest.return_value = CameraPacket(1, 1, self.now, frame)
        # Avoid depending on test class ordering: inject a valid model-sourced completion.
        from facial_attendance.ui.webcam_prototype import Emotion, StubAdapters

        adapters = StubAdapters("Live match")
        adapters.emotion = lambda request: Completion(
            request.context,
            Emotion(
                2,
                "Positive",
                0.8,
                source="model",
                model_id="display-fixture",
                model_version="a" * 64,
                checkpoint_sha256="b" * 64,
                preprocessing_version="fixture",
                selected_epoch=9,
            ),
        )
        app.adapters = adapters
        app._trained_emotion_enabled = True
        app.model_info = Mock()
        app._tick()
        app.detector.finish([legacy.BOX])
        app._tick()
        app.inference.finish(evaluate_adapters(app.inference.job[1]))
        app._tick()
        summary = app.summary.set.call_args.args[0]
        self.assertIn("Identity (simulated)", summary)
        self.assertIn("Liveness (simulated)", summary)
        self.assertIn("Emotion (trained CNN): Positive", summary)
        self.assertIn("epoch 9", app.model_info.set.call_args.args[0])
        self.assertFalse(app.controller.state.can_save)


@unittest.skipUnless(
    torch is not None and os.environ.get("FA_EMOTION_CHECKPOINT"),
    "Set FA_EMOTION_CHECKPOINT to test the supplied checkpoint",
)
class SuppliedEmotionCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ["FA_EMOTION_CHECKPOINT"])
        cls.engine = EmotionInference(cls.path).load()

    def test_supplied_checkpoint_hash_identity_epoch_and_weights_version(self):
        self.assertEqual(sha256(self.path.read_bytes()).hexdigest(), CHECKPOINT_SHA256)
        self.assertEqual(self.engine.provenance["selected_epoch"], 9)
        self.assertEqual(
            self.engine.provenance["model_id"],
            "20261114T180653198248Z-emotion-cnn-e161192-s42",
        )
        self.assertEqual(
            self.engine.provenance["model_version"],
            "546da19237b1ed2549f657c535102071fa0c951d428412746819e71a137b7a20",
        )

    def test_supplied_model_produces_finite_three_class_logits_and_probabilities(self):
        tensor = preprocess_bgr_crop(
            np.full((80, 100, 3), (11, 22, 33), dtype=np.uint8)
        )
        with torch.inference_mode():
            logits = self.engine.model(tensor)
        self.assertEqual(tuple(logits.shape), (1, 3))
        prediction = prediction_from_logits(logits)
        self.assertAlmostEqual(sum(prediction.probabilities), 1, places=6)
        self.assertIn(prediction.label, ("Negative", "Neutral", "Positive"))

    def test_supplied_adapter_is_genuine_model_output_with_original_context(self):
        frame = np.full((80, 100, 3), 100, dtype=np.uint8)
        frame.flags.writeable = False
        controller = PrototypeController()
        request = controller.begin_capture(frame, [(10, 20, 50, 60)])
        result = TrainedEmotionAdapter(self.path, inference=self.engine)(request)
        self.assertEqual(result.context, request.context)
        self.assertEqual(result.result.source, "model")
        self.assertTrue(controller.receive(result))
        self.assertFalse(controller.state.can_save)


if __name__ == "__main__":
    unittest.main()
