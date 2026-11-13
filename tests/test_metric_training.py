"""Tiny CPU triplet runs on generated faces: P×K batches, selection, resume, warm start."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this environment")

import json
from pathlib import Path
import tempfile

import torch

from facial_attendance.recognition.embeddings import score_images
from facial_attendance.recognition.metric_training import (
    MetricConfig,
    load_metric_config,
    train_metric,
)
from facial_attendance.recognition.model import FaceEmbeddingNet, build_meta, load_checkpoint, save_checkpoint
from test_validation_pairs import build_layout

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**changes) -> MetricConfig:
    # The fixture has 3 training images per identity, so K = 3 uses every one.
    settings = dict(
        epochs=2, identities_per_batch=2, images_per_identity=3, margin=0.2,
        learning_rate=0.05, momentum=0.9, weight_decay=5e-4, seed=42, embedding_dim=512,
        input_size=64, eval_batch_size=8, num_workers=0, mixed_precision=True,
    )
    return MetricConfig(**(settings | changes))


class MetricTrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="face-metric-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.layout = build_layout(self.root / "data", {f"n{i}": 2 for i in range(4)})
        self.logs = []

    def run_training(self, config, **options):
        return train_metric(
            config, self.layout, output_root=self.root / "runs", exclusions_file=None,
            device=torch.device("cpu"), log=self.logs.append, **options,
        )

    def test_repository_config_matches_experiment_settings(self):
        config = load_metric_config(ROOT / "configs/face-metric.toml")
        self.assertEqual(
            (config.epochs, config.identities_per_batch, config.images_per_identity, config.batch_size,
             config.margin, config.learning_rate, config.momentum, config.weight_decay, config.seed,
             config.embedding_dim, config.input_size),
            (30, 64, 4, 256, 0.2, 0.05, 0.9, 5e-4, 42, 512, 64),
        )
        with self.assertRaises(ValueError):
            load_metric_config(ROOT / "configs/face-classification.toml")

    def test_run_records_triplet_check_selection_and_checkpoint(self):
        run = self.run_training(tiny_config())
        manifest = json.loads((run / "manifest.json").read_text())
        history = json.loads((run / "history.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertIn("-metric-", run.name)
        self.assertEqual(len(history), 2)
        for record in history:
            triplets = record["triplets"]
            # 12 training images give 2 batches of 2 identities x 3 images; each batch has
            # 2 x 3 x 2 anchor-positive pairs.
            self.assertEqual(record["train_batches"], 2)
            self.assertEqual(triplets["anchor_positive_pairs"], 2 * 12)
            self.assertEqual(triplets["relationship_check"], "passed on all 2 batches")
            self.assertLessEqual(triplets["semi_hard"] + triplets["fallback"], 24)
        best = max(history, key=lambda r: r["val_pairs_cosine_auc"])
        self.assertEqual(manifest["selection"]["selected_epoch"], best["epoch"])
        self.assertIsNone(manifest["selection"]["threshold"])
        self.assertIn("triplet loss, margin 0.2", manifest["loss"])
        self.assertEqual(manifest["config"]["batch_size"], 6)
        self.assertIsNone(manifest["initialisation"])
        model, meta = load_checkpoint(run / "best.pt")
        self.assertEqual(meta["approach"], "metric")
        self.assertIsNone(meta["num_classes"])
        self.assertEqual(meta["model_version"], manifest["output"]["model_version"])
        # The shared two-image scorer reads a metric checkpoint unchanged.
        image = next((self.layout.root / "classification_data/val_data/n0").iterdir())
        score = score_images(model, meta, image, image)
        self.assertAlmostEqual(score["cosine"], 1.0, places=5)

    def test_resume_continues_and_refuses_changed_settings(self):
        run = self.run_training(tiny_config(epochs=1))
        with self.assertRaises(ValueError):
            self.run_training(tiny_config(epochs=1, margin=0.5), resume=run)
        resumed = self.run_training(tiny_config(epochs=1, num_workers=0), resume=run)
        manifest = json.loads((resumed / "manifest.json").read_text())
        self.assertEqual(manifest["epochs_completed"], 1)
        self.assertEqual(len(manifest["sessions"]), 2)

    def test_warm_start_copies_the_classification_backbone_and_drops_the_head(self):
        source = FaceEmbeddingNet(num_classes=4)
        checkpoint = self.root / "classification-best.pt"
        save_checkpoint(checkpoint, source, build_meta(source, model_id="cls-run", approach="classification"))
        run = self.run_training(tiny_config(epochs=1, learning_rate=0.0, momentum=0.0, weight_decay=0.0), warm_start=checkpoint)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertIn("-metric-warm-", run.name)
        self.assertEqual(manifest["initialisation"]["model_id"], "cls-run")
        # With a zero learning rate the trained embedding layer still equals the source.
        model, _ = load_checkpoint(run / "best.pt")
        self.assertTrue(torch.equal(model.embedding.weight, source.embedding.weight))
        with self.assertRaises(ValueError):
            self.run_training(tiny_config(epochs=1), resume=run, warm_start=checkpoint)


if __name__ == "__main__":
    unittest.main()
