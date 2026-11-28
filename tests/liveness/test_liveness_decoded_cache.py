"""Decoded caching preserves the existing PIL augmentation input exactly."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

from facial_attendance.liveness.datasets.decoded_cache import (
    DecodedFaces,
    _verified_arrays,
    build_decoded_cache,
)
from facial_attendance.liveness.datasets.image_augmentation import (
    GeometricColorAugmentation,
)


class Source:
    samples = [0, 1]
    targets = [{"binary": 0}, {"binary": 1}]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        return Image.fromarray(
            rng.integers(0, 256, (57, 83, 3), dtype=np.uint8)
        ), self.targets[index]

    def bindings(self):
        return {"inventory_sha256": "fixed", "samples": 2}


class DecodedCacheTests(unittest.TestCase):
    def test_sparse_corrections_preserve_base_and_augmentation_input(self):
        source = Source()
        with tempfile.TemporaryDirectory() as directory:
            base = build_decoded_cache(source, directory, workers=2)
            path = Path(directory)
            original = (path / "decoded-faces.npy").read_bytes()
            np.save(path / "row-indices.npy", np.array([1], dtype=np.int64))
            pixels = np.full((1, 224, 224, 3), 23, dtype=np.uint8)
            np.save(path / "face-pixels.npy", pixels)
            receipt = json.loads((path / "decoded-faces.json").read_text())
            receipt["corrections"] = {
                "schema": "liveness-face-row-corrections-v1",
                "rows": 1,
                "files": {
                    name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ("row-indices.npy", "face-pixels.npy")
                },
            }
            (path / "decoded-faces.json").write_text(json.dumps(receipt))
            corrected = DecodedFaces(source, path)
            self.assertEqual(corrected[0][0].tobytes(), base[0][0].tobytes())
            self.assertEqual(corrected[1][0].tobytes(), pixels[0].tobytes())
            self.assertEqual((path / "decoded-faces.npy").read_bytes(), original)
            transform = GeometricColorAugmentation(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )
            targets = {
                "depth_map": torch.zeros(1, 14, 14),
                "reflection_map": torch.zeros(3, 14, 14),
            }
            torch.manual_seed(42)
            expected, _ = transform(Image.fromarray(pixels[0]), targets)
            torch.manual_seed(42)
            actual, _ = transform(corrected[1][0], targets)
            self.assertTrue(torch.equal(actual, expected))
            with (path / "face-pixels.npy").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "correction checksum"):
                DecodedFaces(source, path)

    def test_cache_preserves_crop_resize_and_seeded_augmentation(self):
        source = Source()
        transform = GeometricColorAugmentation(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        with tempfile.TemporaryDirectory() as directory:
            cached = build_decoded_cache(source, directory, workers=2)
            for index in range(2):
                image, _ = source[index]
                restored, _ = cached[index]
                self.assertEqual(
                    Resize((224, 224))(image).tobytes(), restored.tobytes()
                )
                targets = {
                    "depth_map": torch.zeros(1, 14, 14),
                    "reflection_map": torch.zeros(3, 14, 14),
                }
                for seed in (42, 891):
                    torch.manual_seed(seed)
                    expected, _ = transform(image, targets)
                    torch.manual_seed(seed)
                    actual, _ = transform(restored, targets)
                    self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(cached.bindings(), source.bindings())
            _verified_arrays.clear()
            with patch(
                "facial_attendance.liveness.datasets.decoded_cache.hashlib.file_digest",
                side_effect=AssertionError("Unnecessary duplicate SHA"),
            ):
                reopened = DecodedFaces(source, directory)
                self.assertEqual(reopened[0][0].tobytes(), cached[0][0].tobytes())
            array = Path(directory) / "decoded-faces.npy"
            with array.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "content changed"):
                DecodedFaces(source, directory)

    def test_partial_cache_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "decoded-faces.npy").write_bytes(b"incomplete")
            with self.assertRaisesRegex(FileExistsError, "Incomplete"):
                build_decoded_cache(Source(), directory)


if __name__ == "__main__":
    unittest.main()
