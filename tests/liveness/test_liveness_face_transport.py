"""Lossless transport must reproduce the historical trainer cache exactly."""

import importlib.util
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facial_attendance.liveness.datasets.decoded_cache import build_decoded_cache

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/release/transport_liveness_face_cache.py"
)
SPEC = importlib.util.spec_from_file_location("face_cache_transport", SCRIPT)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class Source:
    def __init__(self, root="original"):
        self.root = root
        self.samples = list(range(12))
        self.targets = [{"binary": index % 2} for index in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pixels = np.random.default_rng(index).integers(
            0, 256, (57, 83, 3), dtype=np.uint8
        )
        return Image.fromarray(pixels), self.targets[index]

    def bindings(self):
        return {
            "root": self.root,
            "inventory_sha256": "inventory",
            "annotations_sha256": "annotations",
            "samples": len(self),
        }


class FaceCacheTransportTests(unittest.TestCase):
    def test_reconstructs_exact_legacy_array_without_accessing_source_images(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
            root = Path(temporary)
            source = Source()
            reference = resources.enter_context(
                closing(build_decoded_cache(source, root / "reference", workers=2))
            )
            expected = transport.sha256(root / "reference/decoded-faces.npy")
            inventory = transport.pack_faces(
                source,
                root / "transport",
                workers=2,
                shard_mib=1,
                expected_sha256=expected,
            )
            self.assertTrue(inventory["completed"])
            self.assertEqual(inventory["array_sha256"], expected)
            self.assertGreater(len(inventory["shards"]), 1)
            relocated = Source(root="another-device")
            with patch.object(
                Source,
                "__getitem__",
                side_effect=AssertionError("Source image accessed"),
            ):
                restored = resources.enter_context(
                    closing(
                        transport.materialize_faces(
                            relocated, root / "transport", root / "restored"
                        )
                    )
                )
                reopened = resources.enter_context(
                    closing(
                        transport.materialize_faces(
                            relocated, root / "transport", root / "restored"
                        )
                    )
                )
            self.assertEqual(
                transport.sha256(root / "restored/decoded-faces.npy"), expected
            )
            for index in range(len(source)):
                self.assertEqual(
                    reference[index][0].tobytes(), restored[index][0].tobytes()
                )
                self.assertEqual(
                    restored[index][0].tobytes(), reopened[index][0].tobytes()
                )

    def test_corrupt_shard_and_wrong_membership_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Source()
            inventory = transport.pack_faces(
                source, root / "transport", workers=2, shard_mib=1
            )
            record = inventory["shards"][0]
            with self.assertRaisesRegex(ValueError, "Invalid shard filename"):
                transport.shard_path(
                    root / "transport", {**record, "file": "../outside.tar"}
                )
            with patch.object(
                Source,
                "bindings",
                return_value={**source.bindings(), "annotations_sha256": "changed"},
            ):
                with self.assertRaisesRegex(ValueError, "source split assignment"):
                    transport.materialize_faces(
                        source, root / "transport", root / "wrong-membership"
                    )
            path = root / "transport" / record["file"]
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "checksum"):
                transport.materialize_faces(
                    source, root / "transport", root / "corrupt"
                )

    def test_historical_pixel_mismatch_cannot_be_marked_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "historical face cache"):
                transport.pack_faces(
                    Source(), root, workers=2, shard_mib=1, expected_sha256="0" * 64
                )
            import json

            self.assertFalse(
                json.loads((root / "face-cache-shards.json").read_text())["completed"]
            )


if __name__ == "__main__":
    unittest.main()
