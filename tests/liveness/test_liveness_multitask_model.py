"""Behavior checks for the consolidated liveness multitask model module group."""

import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RUNTIME_0 = all(
    importlib.util.find_spec(package) for package in ("torch", "torchvision")
)


if RUNTIME_0:
    import torch

    from facial_attendance.liveness.experiments.multitask_experiment import (
        MultitaskTrainingConfig,
    )
    from facial_attendance.liveness.models.binary_model import (
        RESNET_ARCHITECTURE,
        build_model,
        load_checkpoint,
        normalization,
        save_checkpoint,
        state_version,
    )
    from facial_attendance.liveness.models.multitask_model import (
        SOURCE_TYPES,
        MultitaskLivenessModel,
        annotation_targets,
        deterministic_geometry_resize,
        multitask_loss,
    )


@unittest.skipUnless(RUNTIME_0, "Requires the existing project runtime")
class MultitaskModelTests(unittest.TestCase):
    def test_geometry_preserves_author_spatial_grid(self):
        model = MultitaskLivenessModel(
            build_model(RESNET_ARCHITECTURE), geometry=True
        ).eval()
        spatial = torch.zeros(1, 512, 7, 7)
        self.assertEqual(tuple(model.depth_map(spatial).shape), (1, 1, 7, 7))
        self.assertEqual(tuple(model.reflection_map(spatial).shape), (1, 3, 7, 7))

    def test_geometry_matches_published_padding_sigmoid_resize_sequence(self):
        model = MultitaskLivenessModel(
            build_model(RESNET_ARCHITECTURE), geometry=True
        ).eval()
        torch.manual_seed(7)
        spatial = torch.randn(2, 512, 7, 7)
        with torch.no_grad():
            for head in (model.depth_map, model.reflection_map):
                # The independent functional reference includes border pixels;
                # an unpadded 5x5 head must not pass this numerical comparison.
                reference = torch.nn.functional.conv2d(
                    spatial, head.weight, padding=1
                ).sigmoid()
                reference = torch.nn.functional.interpolate(
                    reference, size=(14, 14), mode="bilinear", align_corners=False
                )
                actual = deterministic_geometry_resize(head(spatial).sigmoid())
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_control_skips_heads_with_identical_loss_gradients_and_sgd_state(self):
        torch.manual_seed(12)
        reference = MultitaskLivenessModel(
            build_model(RESNET_ARCHITECTURE), geometry=True
        ).train()
        optimized = copy.deepcopy(reference)
        images = torch.randn(2, 3, 96, 96)
        targets = self.targets()
        old_outputs = reference.forward_tasks(images)
        old_loss = torch.nn.functional.cross_entropy(
            old_outputs["binary_classes"], targets["binary"]
        )
        with (
            patch.object(
                optimized.face_attributes,
                "forward",
                side_effect=AssertionError("Unused head called"),
            ),
            patch.object(
                optimized.spoof_type,
                "forward",
                side_effect=AssertionError("Unused head called"),
            ),
            patch.object(
                optimized.illumination,
                "forward",
                side_effect=AssertionError("Unused head called"),
            ),
            patch.object(
                optimized.depth_map,
                "forward",
                side_effect=AssertionError("Unused head called"),
            ),
            patch.object(
                optimized.reflection_map,
                "forward",
                side_effect=AssertionError("Unused head called"),
            ),
        ):
            outputs = optimized.forward_tasks(images, auxiliary=False)
        loss, parts = multitask_loss(
            outputs, {"binary": targets["binary"]}, auxiliary=False
        )
        self.assertEqual(set(parts), {"binary"})
        self.assertTrue(torch.equal(loss, old_loss))
        old_loss.backward()
        loss.backward()
        for old, new in zip(
            reference.parameters(), optimized.parameters(), strict=True
        ):
            self.assertEqual(old.grad is None, new.grad is None)
            if old.grad is not None:
                self.assertTrue(torch.equal(old.grad, new.grad))
        for model in (reference, optimized):
            torch.optim.SGD(
                model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001
            ).step()
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, optimized.state_dict()[name]), name)

    def targets(self):
        return {
            "binary": torch.tensor([0, 1]),
            "spoof_type": torch.tensor([0, 6]),
            "illumination": torch.tensor([0, 2]),
            "face_attributes": torch.zeros(2, 40),
        }

    def test_every_ablation_skips_disabled_heads_and_only_selected_heads_receive_gradients(
        self,
    ):
        from contextlib import ExitStack

        from facial_attendance.liveness.experiments.aenet_experiment import (
            SUPERVISION_VARIANTS,
        )
        from facial_attendance.liveness.models.multitask_model import (
            ALL_SOURCE_TYPES,
            GEOMETRIC_TASKS,
            SEMANTIC_TASKS,
        )

        torch.set_num_threads(1)
        torch.manual_seed(42)
        initial = MultitaskLivenessModel(
            build_model(RESNET_ARCHITECTURE),
            source_types=ALL_SOURCE_TYPES,
            geometry=True,
        )
        images = torch.randn(2, 3, 224, 224)
        target = self.targets()
        target["depth_map"] = torch.zeros(2, 1, 14, 14)
        target["depth_map"][0] = 0.2
        target["reflection_map"] = torch.zeros(2, 3, 14, 14)
        target["reflection_map"][1] = 0.2
        for arm, selected in SUPERVISION_VARIANTS.items():
            with self.subTest(arm=arm):
                model = copy.deepcopy(initial).train()
                with ExitStack() as contexts:
                    for name in SEMANTIC_TASKS + GEOMETRIC_TASKS:
                        if name not in selected:
                            contexts.enter_context(
                                patch.object(
                                    getattr(model, name),
                                    "forward",
                                    side_effect=AssertionError(
                                        "Disabled head computed"
                                    ),
                                )
                            )
                    outputs = model.forward_tasks(
                        images, auxiliary=bool(selected), tasks=selected
                    )
                loss, parts = multitask_loss(
                    outputs, target, auxiliary=bool(selected), tasks=selected
                )
                self.assertEqual(set(parts), {"binary", *selected})
                loss.backward()
                for name in SEMANTIC_TASKS + GEOMETRIC_TASKS:
                    gradient = getattr(model, name).weight.grad
                    if name in selected:
                        self.assertGreater(float(gradient.abs().sum()), 0.0)
                    else:
                        self.assertIsNone(gradient)

    def test_source_labels_and_no_attribute_spoof_targets(self):
        for source_type in SOURCE_TYPES:
            attributes = [0] * 44
            attributes[40], attributes[41], attributes[43] = (
                source_type,
                int(source_type != 0),
                int(source_type != 0),
            )
            result = annotation_targets(attributes)
            self.assertEqual(SOURCE_TYPES[result["spoof_type"]], source_type)
            self.assertEqual(result["binary"], int(source_type != 0))
            self.assertEqual(tuple(result["face_attributes"].shape), (40,))
        attributes = [0] * 44
        attributes[3] = 1
        self.assertEqual(annotation_targets(attributes)["face_attributes"][3], 1)
        attributes[40], attributes[41], attributes[43] = 1, 1, 1
        with self.assertRaisesRegex(ValueError, "no-attribute"):
            annotation_targets(attributes)
        attributes = [0] * 44
        attributes[3] = -1
        with self.assertRaisesRegex(ValueError, "must be binary"):
            annotation_targets(attributes)
        for source_type, illumination, binary in (
            (4, 1, 1),
            (0, 1, 0),
            (1, 0, 1),
            (1, 5, 1),
            (1, 1, 0),
        ):
            attributes = [0] * 44
            attributes[40], attributes[41], attributes[43] = (
                source_type,
                illumination,
                binary,
            )
            with self.assertRaises(ValueError):
                annotation_targets(attributes)

    def test_auxiliary_only_loss_reaches_all_heads_and_backbone(self):
        torch.manual_seed(4)
        model = MultitaskLivenessModel(build_model()).eval()
        tasks = model.forward_tasks(torch.randn(2, 3, 64, 64))
        loss, parts = multitask_loss(tasks, self.targets())
        expected = (
            parts["binary"]
            + parts["face_attributes"]
            + 0.1 * parts["spoof_type"]
            + 0.01 * parts["illumination"]
        )
        self.assertTrue(torch.allclose(loss, expected))
        (loss - parts["binary"]).backward()
        for name in ("face_attributes", "spoof_type", "illumination"):
            self.assertGreater(getattr(model, name).weight.grad.abs().sum().item(), 0)
        self.assertGreater(
            model.binary_model.features[0][0].weight.grad.abs().sum().item(), 0
        )
        # Zero-valued gradients can be present through subtraction; the binary
        # head must receive no effective update from auxiliary supervision.
        self.assertEqual(
            model.binary_model.classifier[-1].weight.grad.abs().sum().item(), 0
        )

    def test_binary_control_does_not_train_auxiliary_heads(self):
        model = MultitaskLivenessModel(build_model()).eval()
        tasks = model.forward_tasks(torch.randn(2, 3, 64, 64))
        loss, parts = multitask_loss(tasks, self.targets(), auxiliary=False)
        loss.backward()
        self.assertTrue(torch.equal(loss, parts["binary"]))
        for name in ("face_attributes", "spoof_type", "illumination"):
            self.assertIsNone(getattr(model, name).weight.grad)

    def test_resnet_softmax_equivalence_and_checkpoint_roundtrip(self):
        torch.manual_seed(8)
        binary = build_model(RESNET_ARCHITECTURE).eval()
        model = MultitaskLivenessModel(binary).eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            tasks = model.forward_tasks(images)
            self.assertTrue(torch.equal(binary(images).reshape(-1), tasks["binary"]))
            self.assertTrue(
                torch.allclose(
                    tasks["binary"].sigmoid(), tasks["binary_classes"].softmax(1)[:, 1]
                )
            )
        mean, std = normalization(RESNET_ARCHITECTURE)
        meta = {
            "architecture": RESNET_ARCHITECTURE,
            "label_mapping": {"live": 0, "spoof": 1},
            "input_size": 224,
            "mean": list(mean),
            "std": list(std),
            "optimizer_updates": 1,
            "threshold": 0.4,
            "threshold_source": "validation EER operating point",
            "model_version": state_version(binary.state_dict()),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "liveness-inference.pt"
            save_checkpoint(path, binary.state_dict(), meta)
            restored, _ = load_checkpoint(path)
            with torch.inference_mode():
                self.assertTrue(torch.equal(restored(images), binary(images)))
            save_checkpoint(path, binary.state_dict(), {**meta, "mean": [0.5] * 3})
            with self.assertRaisesRegex(ValueError, "contract"):
                load_checkpoint(path)

    def test_resnet_binary_loss_is_cross_entropy(self):
        model = MultitaskLivenessModel(build_model(RESNET_ARCHITECTURE)).eval()
        tasks = model.forward_tasks(torch.randn(2, 3, 64, 64))
        _, parts = multitask_loss(tasks, self.targets())
        expected = torch.nn.functional.cross_entropy(
            tasks["binary_classes"], self.targets()["binary"]
        )
        self.assertTrue(torch.equal(expected, parts["binary"]))
        parts["face_attributes"].backward()
        self.assertGreater(
            model.binary_model.backbone.conv1.weight.grad.abs().sum().item(), 0
        )

    def test_resource_and_optimization_budgets_reject_invalid_values(self):
        MultitaskTrainingConfig().validate()
        for values in (
            {"epochs": 0},
            {"max_seconds": 0},
            {"epoch_samples": 100},
            {"workers": -1},
            {"learning_rate": float("nan")},
            {"momentum": 1.0},
        ):
            with self.assertRaises(ValueError):
                MultitaskTrainingConfig(**values).validate()


RUNTIME_1 = all(
    importlib.util.find_spec(name) for name in ("torch", "torchvision", "sklearn")
)


if RUNTIME_1:
    import torch

    from facial_attendance.liveness.models.binary_model import build_model
    from facial_attendance.liveness.models.multitask_model import MultitaskLivenessModel
    from facial_attendance.liveness.training.multitask_training import (
        gradients_are_finite,
        train_multitask_epoch,
    )


@unittest.skipUnless(RUNTIME_1, "Requires the project ML runtime")
class TrainingProgressTests(unittest.TestCase):
    def test_gradient_check_preserves_values_and_rejects_nonfinite(self):
        parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(3)]
        self.assertTrue(gradients_are_finite(parameters))
        parameters[0].grad = torch.tensor([0.0, 1.0, -2.0])
        original = parameters[0].grad.clone()
        self.assertTrue(gradients_are_finite(parameters))
        self.assertTrue(torch.equal(original, parameters[0].grad))
        for value in (float("nan"), float("inf"), -float("inf")):
            parameters[2].grad = torch.tensor([0.0, value, 1.0])
            self.assertFalse(gradients_are_finite(parameters))

    def test_first_progress_event_has_initialized_elapsed_time(self):
        # This is an optimizer-loop regression fixture, not CPU fallback for
        # the real CUDA experiment. Mock time exercises the 30-second branch.
        model = MultitaskLivenessModel(build_model())
        images = torch.randn(2, 3, 64, 64)
        targets = {
            "binary": torch.tensor([0, 1]),
            "face_attributes": torch.zeros(2, 40),
            "spoof_type": torch.tensor([0, 1]),
            "illumination": torch.tensor([0, 1]),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        captured = StringIO()
        with (
            patch("torch.Tensor.cuda", lambda tensor, **kwargs: tensor),
            patch(
                "facial_attendance.liveness.training.multitask_training.make_loader",
                return_value=[(images, targets)],
            ),
            patch(
                "facial_attendance.liveness.training.multitask_training.time.monotonic",
                side_effect=[0.0, 31.0, 31.0, 31.0],
            ),
            redirect_stdout(captured),
        ):
            result = train_multitask_epoch(
                model,
                [0, 1],
                optimizer,
                SimpleNamespace(batch_size=2),
                epoch=1,
                auxiliary=False,
            )
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertEqual(result["samples"], 2)
        progress = json.loads(captured.getvalue())
        self.assertEqual(progress["phase"], "training")
        self.assertEqual(progress["elapsed_seconds"], 31.0)
        self.assertEqual(progress["optimizer_updates"], 1)


if __name__ == "__main__":
    unittest.main()
