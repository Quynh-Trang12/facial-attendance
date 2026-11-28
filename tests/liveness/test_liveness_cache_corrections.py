"""Row corrections preserve historical arrays and reject incompatible inputs."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from facial_attendance.liveness.datasets.decoded_cache import DecodedFaces

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/preparation/correct_liveness_caches.py"
)
spec = importlib.util.spec_from_file_location("cache_corrections", SCRIPT)
corrections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corrections)


class CacheCorrectionTests(unittest.TestCase):
    def test_apply_preserves_base_rows_and_rejects_changed_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "packet"
            faces = root / "base-faces"
            geometry = root / "base-geometry"
            geometry.mkdir()
            binding = {
                "inventory_sha256": "inventory",
                "annotations_sha256": "annotations",
                "samples": 2,
                "crop_recipe": "new-crop",
            }
            packet.mkdir()
            report = {"splits": {}}
            for split in ("train", "validation"):
                directory = packet / split
                directory.mkdir()
                np.save(directory / "row-indices.npy", np.array([1], dtype=np.int64))
                np.save(
                    directory / "face-pixels.npy",
                    np.full((1, 224, 224, 3), 17, dtype=np.uint8),
                )
                base = faces / split
                base.mkdir(parents=True)
                np.save(
                    base / "decoded-faces.npy",
                    np.zeros((2, 224, 224, 3), dtype=np.uint8),
                )
                receipt = {
                    "data": {k: v for k, v in binding.items() if k != "crop_recipe"},
                    "shape": [2, 224, 224, 3],
                    "sha256": corrections.file_sha256(base / "decoded-faces.npy"),
                }
                corrections.write_json(base / "decoded-faces.json", receipt)
                report["splits"][split] = {
                    "data": binding,
                    "changed_rows": 1,
                    "files": {
                        name: corrections.file_sha256(directory / name)
                        for name in ("row-indices.npy", "face-pixels.npy")
                    },
                }
            teacher = {
                "depth_checkpoint_sha256": "depth",
                "reflection_checkpoint_sha256": "reflection",
                "mesh_sha256": {"mesh": "mesh"},
                "source_sha256": {"model": "model"},
            }
            np.save(
                geometry / "geometric-targets.npy",
                np.zeros((2, 4, 14, 14), dtype=np.float32),
            )
            corrections.write_json(
                geometry / "geometric-cache.json",
                {
                    "bindings": dict(teacher),
                    "array_sha256": corrections.file_sha256(
                        geometry / "geometric-targets.npy"
                    ),
                },
            )
            delta = packet / "geometry"
            delta.mkdir()
            np.save(
                delta / "geometric-targets.npy",
                np.full((1, 4, 14, 14), 0.25, dtype=np.float32),
            )
            corrections.write_json(
                delta / "geometric-cache.json", {"bindings": dict(teacher)}
            )
            report["geometry_array_sha256"] = corrections.file_sha256(
                delta / "geometric-targets.npy"
            )
            corrections.write_json(packet / "corrections.json", report)
            args = SimpleNamespace(
                corrections=packet,
                face_cache=faces,
                geometric_cache=geometry,
                output=root / "corrected",
            )
            base_digest = corrections.file_sha256(faces / "train/decoded-faces.npy")
            corrections.apply(args)
            corrected = np.load(args.output / "faces/train/decoded-faces.npy")
            self.assertFalse(corrected[0].any())
            self.assertTrue((corrected[1] == 17).all())
            self.assertEqual(
                corrections.file_sha256(faces / "train/decoded-faces.npy"), base_digest
            )
            corrected_maps = np.load(args.output / "geometry/geometric-targets.npy")
            self.assertFalse(corrected_maps[0].any())
            self.assertTrue((corrected_maps[1] == 0.25).all())
            metadata = json.loads(
                (args.output / "faces/train/decoded-faces.json").read_text()
            )
            self.assertEqual(metadata["data"]["crop_recipe"], "new-crop")
            self.assertEqual(metadata["base_array_sha256"], base_digest)
            args.output = root / "sparse"
            args.sparse_faces = True
            corrections.apply(args)

            class Source:
                samples = [0, 1]
                targets = [{"binary": 0}, {"binary": 1}]

                def __len__(self):
                    return 2

                def bindings(self):
                    return binding

            with closing(DecodedFaces(Source(), args.output / "faces/train")) as sparse:
                self.assertTrue((np.asarray(sparse[1][0]) == 17).all())
            self.assertEqual(
                corrections.file_sha256(args.output / "faces/train/decoded-faces.npy"),
                base_digest,
            )
            teacher["depth_checkpoint_sha256"] = "different"
            corrections.write_json(
                delta / "geometric-cache.json", {"bindings": teacher}
            )
            args.output = root / "wrong-teacher"
            with self.assertRaisesRegex(ValueError, "Teacher changes"):
                corrections.apply(args)


if __name__ == "__main__":
    unittest.main()
