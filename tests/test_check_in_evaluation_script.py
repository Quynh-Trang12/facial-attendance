"""The quality and k-shot steps end to end on generated images and an untrained checkpoint."""

import importlib.util
import unittest

# These steps read images and run the network; the dependency jobs have all three.
if any(importlib.util.find_spec(name) is None for name in ("numpy", "cv2", "torch")):
    raise unittest.SkipTest("numpy, OpenCV or torch is not installed in this environment")

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

from facial_attendance.recognition.model import FaceEmbeddingNet, build_meta, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_check_in_quality.py"


def save_face(path: Path, seed: int, size=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="check-in-script-fixture-")
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        data = cls.root / "data"
        seed = 0
        for identity in ("n000", "n001", "n002", "n003"):
            for split in ("train_data", "val_data", "test_data"):
                for index in range(2):
                    seed += 1
                    save_face(data / "classification_data" / split / identity / f"{index}.png", seed)
        for name in ("a", "b", "c"):
            seed += 1
            save_face(data / "verification_data" / f"{name}.png", seed)
        (data / "verification_pairs_val.txt").write_text(
            "verification_data/a.png verification_data/b.png 1\nverification_data/b.png verification_data/c.png 0\n")
        cls.val_pairs = cls.root / "val-pairs.txt"
        cls.val_pairs.write_text("classification_data/val_data/n000/0.png classification_data/val_data/n001/0.png 0\n")
        # One excluded test image leaves n003 with three images, too few for k = 3.
        excluded = data / "classification_data/test_data/n003/1.png"
        digest = hashlib.sha256(excluded.read_bytes()).hexdigest()
        cls.exclusions = cls.root / "exclusions.tsv"
        cls.exclusions.write_text(
            f"classification_data/test_data/n003/1.png\tduplicate\t{digest}\tclassification_data/test_data/n003/0.png\n")
        cls.data = data

        torch.manual_seed(0)
        model = FaceEmbeddingNet(None, embedding_dim=16).eval()
        meta = build_meta(model, model_id="run-m", approach="metric") | {"threshold": 0.5, "threshold_source": "test"}
        cls.checkpoint = cls.root / "deploy.pt"
        save_checkpoint(cls.checkpoint, model, meta)

    def run_script(self, *args):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, env=env)

    def data_args(self):
        return ["--data-root", self.data, "--exclusions", self.exclusions]

    def test_quality_measures_each_image_of_a_trial_list_once(self):
        for trials, expected in (("official", ["verification_data/a.png", "verification_data/b.png",
                                               "verification_data/c.png"]),
                                 (self.val_pairs, ["classification_data/val_data/n000/0.png",
                                                   "classification_data/val_data/n001/0.png"])):
            output = self.root / f"quality-{len(expected)}.csv"
            result = self.run_script("quality", *self.data_args(), "--trials", trials, "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["path"] for row in rows], expected)
            self.assertTrue(all(float(row["sharpness"]) > 0 and row["size"] == "48" for row in rows))
            summary = json.loads(Path(f"{output}.json").read_text())
            self.assertEqual(summary["images"], len(expected))

    def test_kshot_embeds_val_and_test_images_and_skips_short_identities(self):
        output = self.root / "kshot"
        result = self.run_script("kshot", *self.data_args(), "--checkpoint", self.checkpoint,
                                 "--output", output, "--device", "cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((output / "kshot.json").read_text())
        self.assertEqual(record["images_embedded"], 15)
        self.assertEqual((record["identities"], record["skipped_identities"]), (3, 1))
        self.assertEqual(record["tau"], 0.5)
        self.assertEqual(set(record["k"]), {"1", "2", "3"})
        with np.load(output / "kshot-embeddings.npz") as saved:
            self.assertEqual(saved["raw_embeddings"].shape, (15, 16))
            self.assertNotIn("classification_data/test_data/n003/1.png", saved["paths"].tolist())

    def test_existing_output_is_kept(self):
        existing = self.root / "existing"
        existing.mkdir()
        result = self.run_script("kshot", *self.data_args(), "--checkpoint", self.checkpoint, "--output", existing)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
