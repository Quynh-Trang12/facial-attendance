"""Embedding network, checkpoint format, scoring and evaluator exports on generated data."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this environment")

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from facial_attendance.recognition.data_loader import ClassificationFaces
from facial_attendance.recognition.embeddings import (
    export_predictions,
    export_trial_embeddings,
    score_images,
    score_pair,
)
from facial_attendance.recognition.model import (
    META_KEYS,
    FaceEmbeddingNet,
    build_meta,
    eval_transform,
    load_checkpoint,
    model_version,
    save_checkpoint,
)
from test_validation_pairs import build_layout


class FaceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        directory = tempfile.TemporaryDirectory(prefix="face-model-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = FaceEmbeddingNet(num_classes=5).eval()

    def test_embedding_is_fixed_length_and_feeds_the_head(self):
        images = torch.randn(3, 3, 64, 64)
        embedding = self.model.embed(images)
        self.assertEqual(tuple(embedding.shape), (3, 512))
        self.assertTrue(torch.equal(self.model(images), self.model.head(embedding)))
        # The 3x3 stride-1 stem keeps 64x64 maps in the first block.
        self.assertEqual(self.model.backbone.conv1.stride, (1, 1))
        self.assertIsInstance(self.model.backbone.maxpool, torch.nn.Identity)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_checkpoint_round_trip_carries_schema_meta(self):
        meta = build_meta(self.model, model_id="run-x", approach="classification")
        self.assertEqual(set(META_KEYS), set(meta))
        self.assertEqual((meta["input_size"], meta["embedding_dim"]), (64, 512))
        self.assertIsNone(meta["threshold"])
        file = self.root / "best.pt"
        save_checkpoint(file, self.model, meta)
        loaded, loaded_meta = load_checkpoint(file)
        self.assertEqual(loaded_meta, meta)
        images = torch.randn(2, 3, 64, 64)
        self.assertTrue(torch.allclose(loaded.embed(images), self.model.embed(images)))

    def test_model_version_tracks_weights_not_meta(self):
        state = self.model.state_dict()
        version = model_version(state)
        self.assertEqual(len(version), 12)
        self.assertEqual(model_version(state), version)
        state["embedding.bias"] = state["embedding.bias"] + 1
        self.assertNotEqual(model_version(state), version)

    def test_tampered_weights_are_refused(self):
        meta = build_meta(self.model, model_id="run-x", approach="classification")
        meta["model_version"] = "000000000000"
        save_checkpoint(self.root / "bad.pt", self.model, meta)
        with self.assertRaises(ValueError):
            load_checkpoint(self.root / "bad.pt")

    def test_pair_scores(self):
        same = score_pair(np.array([1.0, 0.0]), np.array([2.0, 0.0]))
        self.assertAlmostEqual(same["cosine"], 1.0)
        self.assertAlmostEqual(same["euclidean_distance"], 1.0)
        self.assertAlmostEqual(same["euclidean_distance_normalized"], 0.0)
        opposite = score_pair(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
        self.assertAlmostEqual(opposite["cosine"], -1.0)
        with self.assertRaises(ValueError):
            score_pair(np.zeros(2), np.ones(2))

    def test_eval_transform_resizes_and_normalizes(self):
        from PIL import Image

        tensor = eval_transform()(Image.new("RGB", (100, 80), color=(124, 116, 104)))
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertLess(tensor.abs().max().item(), 0.01)  # ImageNet mean maps to ~0


class ExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="face-export-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.layout = build_layout(self.root / "data", {f"n{i}": 2 for i in range(4)})
        (self.layout.root / "verification_data" / "b.png").write_bytes(
            (self.layout.root / "verification_data" / "a.png").read_bytes()
        )
        self.layout.pairs_file.write_text("verification_data/a.png b.png 1\na.png b.png 0\n")
        torch.manual_seed(0)
        self.model = FaceEmbeddingNet(num_classes=4).eval()
        self.meta = build_meta(self.model, model_id="run-x", approach="classification")

    def test_two_image_scoring(self):
        a = self.layout.root / "verification_data" / "a.png"
        result = score_images(self.model, self.meta, a, a)
        self.assertAlmostEqual(result["cosine"], 1.0, places=5)
        self.assertEqual(result["embedding_dim"], 512)

    def test_official_embeddings_keyed_by_trial_strings(self):
        output = self.root / "out" / "official.npz"
        summary = export_trial_embeddings(self.model, self.meta, self.layout, self.layout.pairs_file, output)
        self.assertEqual(summary["images"], 3)
        with np.load(output, allow_pickle=False) as data:
            self.assertEqual(list(data["paths"]), ["verification_data/a.png", "b.png", "a.png"])
            self.assertEqual(data["raw_embeddings"].shape, (3, 512))
            self.assertEqual(data["raw_embeddings"].dtype, np.float32)
        with self.assertRaises(FileExistsError):
            export_trial_embeddings(self.model, self.meta, self.layout, self.layout.pairs_file, output)

    def test_predictions_csv_matches_dataset(self):
        summary = export_predictions(self.model, self.meta, self.layout, "test_data", self.root / "pred")
        with (self.root / "pred" / "test_data-predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), summary["images"])
        self.assertEqual(set(rows[0]), {"id", "true", "predicted"})
        test = ClassificationFaces(self.layout, "test_data", class_to_idx=ClassificationFaces(self.layout, "train_data").class_to_idx)
        self.assertEqual([int(r["true"]) for r in rows], [s.target for s in test.samples])


if __name__ == "__main__":
    unittest.main()
