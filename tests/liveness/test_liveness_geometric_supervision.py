"""Behavior checks for the consolidated liveness geometric supervision module group."""

import importlib.util
import unittest

import numpy as np
import torch
from PIL import Image

from facial_attendance.liveness.datasets.image_augmentation import (
    DeviceColorParameters,
    GeometricColorAugmentation,
    apply_device_color,
)

RUNTIME_0 = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


if RUNTIME_0:
    import torch
    from PIL import Image

    from facial_attendance.liveness.datasets.geometric_cache import geometric_targets
    from facial_attendance.liveness.datasets.image_augmentation import (
        GeometricColorAugmentation,
    )
    from facial_attendance.liveness.models.binary_model import (
        AENET_ARCHITECTURE,
        build_model,
        face_transform,
    )
    from facial_attendance.liveness.models.multitask_model import (
        ALL_SOURCE_TYPES,
        MultitaskLivenessModel,
        annotation_targets,
        deterministic_geometry_resize,
        multitask_loss,
    )


@unittest.skipUnless(RUNTIME_0, "Requires the project ML runtime")
class GeometricSupervisionTests(unittest.TestCase):
    def test_required_teacher_and_source_zero_conventions(self):
        with self.assertRaisesRegex(ValueError, "teacher map"):
            geometric_targets(0)
        with self.assertRaisesRegex(ValueError, "teacher map"):
            geometric_targets(1)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            geometric_targets(0, depth_prediction=torch.zeros(1, 20, 20))
        for bad in (
            torch.ones(3, 14, 14),
            torch.full((1, 14, 14), float("nan")),
            torch.full((1, 14, 14), 2.0),
        ):
            with self.assertRaises(ValueError):
                geometric_targets(0, depth_prediction=bad)
        live = geometric_targets(0, depth_prediction=torch.ones(1, 28, 28))
        spoof = geometric_targets(1, reflection_prediction=torch.ones(3, 28, 28) * 0.5)
        self.assertEqual(tuple(live["depth_map"].shape), (1, 14, 14))
        self.assertEqual(torch.count_nonzero(live["reflection_map"]), 0)
        self.assertEqual(torch.count_nonzero(spoof["depth_map"]), 0)
        self.assertTrue(
            torch.equal(spoof["reflection_map"], torch.full((3, 14, 14), 0.5))
        )
        geometric_targets(1, reflection_prediction=torch.zeros(3, 14, 14))

    def test_full_source_scope_and_released_input_range(self):
        attributes = [0] * 44
        attributes[40], attributes[41], attributes[43] = 10, 3, 1
        self.assertEqual(
            annotation_targets(attributes, source_types=ALL_SOURCE_TYPES)["spoof_type"],
            10,
        )
        with self.assertRaises(ValueError):
            annotation_targets(attributes)
        pixel = face_transform(architecture=AENET_ARCHITECTURE)(
            Image.new("RGB", (40, 40), (255, 0, 128))
        )[:, 0, 0]
        self.assertTrue(torch.allclose(pixel, torch.tensor([1.0, 0.0, 128 / 255])))

    def test_color_augmentation_preserves_map_coordinates(self):
        targets = geometric_targets(0, depth_prediction=torch.ones(1, 14, 14))
        transform = GeometricColorAugmentation(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        image, transformed = transform(Image.new("RGB", (60, 80)), targets)
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertIs(transformed, targets)

    def test_deterministic_resize_matches_bilinear_values_and_gradients(self):
        first = torch.randn(2, 3, 5, 5, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        observed = deterministic_geometry_resize(first)
        expected = torch.nn.functional.interpolate(
            second, size=(14, 14), mode="bilinear", align_corners=False
        )
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6))
        observed.square().sum().backward()
        expected.square().sum().backward()
        self.assertTrue(torch.allclose(first.grad, second.grad, atol=1e-5))

    def test_geometry_backprop_and_binary_export(self):
        torch.manual_seed(7)
        binary = build_model(AENET_ARCHITECTURE).eval()
        model = MultitaskLivenessModel(
            binary, source_types=ALL_SOURCE_TYPES, geometry=True
        ).eval()
        images = torch.randn(2, 3, 224, 224)
        outputs = model.forward_tasks(images)
        targets = {
            "binary": torch.tensor([0, 1]),
            "face_attributes": torch.zeros(2, 40),
            "spoof_type": torch.tensor([0, 10]),
            "illumination": torch.tensor([0, 2]),
            "depth_map": torch.stack([torch.ones(1, 14, 14), torch.zeros(1, 14, 14)]),
            "reflection_map": torch.stack(
                [torch.zeros(3, 14, 14), torch.ones(3, 14, 14)]
            ),
        }
        loss, parts = multitask_loss(outputs, targets)
        self.assertEqual(tuple(outputs["spoof_type"].shape), (2, 11))
        self.assertEqual(tuple(outputs["reflection_map"].shape), (2, 3, 14, 14))
        expected = (
            parts["binary"]
            + parts["face_attributes"]
            + 0.1 * parts["spoof_type"]
            + 0.01 * parts["illumination"]
            + 0.1 * parts["depth_map"]
            + 0.1 * parts["reflection_map"]
        )
        self.assertTrue(torch.allclose(loss, expected))
        (0.1 * parts["depth_map"] + 0.1 * parts["reflection_map"]).backward()
        self.assertGreater(model.depth_map.weight.grad.abs().sum(), 0)
        self.assertGreater(model.reflection_map.weight.grad.abs().sum(), 0)
        self.assertGreater(binary.backbone.conv1.weight.grad.abs().sum(), 0)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(images), binary(images)))
        targets["depth_map"][1] = 1
        with self.assertRaisesRegex(ValueError, "zero-target"):
            multitask_loss(outputs, targets)
        del targets["depth_map"]
        with self.assertRaisesRegex(ValueError, "Missing verified"):
            multitask_loss(outputs, targets)


RUNTIME_1 = importlib.util.find_spec("torch") is not None


if RUNTIME_1:
    import torch

    from facial_attendance.liveness.models.depth_teacher import (
        SourceConv,
        prnet_face_input,
        rasterize_depth,
    )


@unittest.skipUnless(RUNTIME_1, "Requires torch")
class DepthProjectionTests(unittest.TestCase):
    def test_z_buffer_keeps_nearest_overlapping_triangle(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.2],
                [13.0, 0.0, 0.2],
                [0.0, 13.0, 0.2],
                [0.0, 0.0, 0.8],
                [13.0, 0.0, 0.8],
                [0.0, 13.0, 0.8],
            ]
        )
        triangles = torch.tensor([[0, 1, 2], [3, 4, 5]])
        depth = rasterize_depth(vertices, triangles)
        self.assertAlmostEqual(depth[0, 2, 2].item(), 0.8, places=6)
        self.assertEqual(depth[0, 13, 13], 0)
        with self.assertRaises(ValueError):
            rasterize_depth(vertices, torch.tensor([[0, 1, 99]]))

    def test_same_padding_preserves_source_shapes(self):
        images = torch.ones(1, 1, 8, 8)
        for stride in (1, 2):
            layer = SourceConv(1, 1, 4, stride, normalized=False, activation=None)
            self.assertEqual(layer(images).shape[-2:], (8 // stride, 8 // stride))
            layer = SourceConv(
                1, 1, 4, stride, transpose=True, normalized=False, activation=None
            )
            self.assertEqual(layer(images).shape[-2:], (8 * stride, 8 * stride))

    def test_supplied_box_crop_uses_source_1_6_expansion(self):
        image = torch.ones(3, 100, 100)
        cropped, alignment = prnet_face_input(image)
        self.assertEqual(cropped.shape, (1, 3, 256, 256))
        left, top, scale, width, height = alignment
        self.assertEqual((left, top, width, height), (-30.0, -30.0, 100, 100))
        self.assertAlmostEqual(scale, 255 / 160)
        self.assertTrue(torch.all(cropped[0, :, 128, 128] == 1))
        self.assertTrue(torch.all(cropped[0, :, 0, 0] == 0))


RUNTIME_2 = importlib.util.find_spec("torch") is not None


if RUNTIME_2:
    import torch

    from facial_attendance.liveness.models.reflection_teacher import (
        tensorflow_bilinear_resize,
    )


@unittest.skipUnless(RUNTIME_2, "Requires torch")
class ReflectionTeacherTests(unittest.TestCase):
    def test_legacy_resize_uses_asymmetric_coordinates_and_border_clamping(self):
        image = torch.tensor([[[[0.0, 2.0], [4.0, 6.0]]]])
        resized = tensorflow_bilinear_resize(image, (4, 4))
        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 2.0],
                        [2.0, 3.0, 4.0, 4.0],
                        [4.0, 5.0, 6.0, 6.0],
                        [4.0, 5.0, 6.0, 6.0],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.equal(resized, expected))
        self.assertTrue(torch.equal(tensorflow_bilinear_resize(image, (2, 2)), image))


class DeviceAugmentationTests(unittest.TestCase):
    def test_exact_existing_recipe_with_distinct_per_image_seeds(self):
        torch.set_num_threads(1)
        targets = {
            "depth_map": torch.zeros(1, 14, 14),
            "reflection_map": torch.zeros(3, 14, 14),
        }
        for hue in (0.05, 0):
            cpu = GeometricColorAugmentation(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=hue
            )
            plan = DeviceColorParameters(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=hue
            )
            expected, raw, orders, factors, shifts = [], [], [], [], []
            for index in range(8):
                image = Image.fromarray(
                    np.random.default_rng(index).integers(
                        0, 256, (53, 71, 3), dtype=np.uint8
                    )
                )
                torch.manual_seed(42 + index)
                expected.append(cpu(image, targets)[0])
                torch.manual_seed(42 + index)
                pixels, metadata = plan(image, targets)
                raw.append(pixels)
                orders.append(metadata["_color_order"])
                factors.append(metadata["_color_factors"])
                shifts.append(metadata["_color_hue"])
            actual = apply_device_color(
                torch.stack(raw),
                torch.stack(orders),
                torch.stack(factors),
                torch.stack(shifts),
            )
            self.assertTrue(
                torch.equal(actual, torch.stack(expected)),
                f"Mismatched pixels: {int((actual != torch.stack(expected)).sum())}",
            )

    def test_invalid_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            apply_device_color(
                torch.zeros(2, 3, 10, 10),
                torch.zeros(2, 4, dtype=torch.int64),
                torch.ones(2, 3),
                torch.zeros(2, dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
