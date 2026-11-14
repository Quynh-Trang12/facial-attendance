"""Two tiny CPU epochs on generated faces: selection, resume and the run record."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this environment")

import json
from pathlib import Path
import tempfile
import unittest

import torch

from facial_attendance.recognition.model import load_checkpoint
from facial_attendance.recognition.training import TrainConfig, load_config, train
from test_validation_pairs import build_layout

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**changes) -> TrainConfig:
    settings = dict(
        epochs=2, batch_size=8, learning_rate=0.05, momentum=0.9, weight_decay=5e-4,
        seed=42, embedding_dim=512, input_size=64, eval_batch_size=8, num_workers=0,
        mixed_precision=True,
    )
    return TrainConfig(**(settings | changes))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="face-train-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.layout = build_layout(self.root / "data", {f"n{i}": 2 for i in range(4)})
        self.logs = []

    def run_training(self, config, **options):
        return train(
            config, self.layout, output_root=self.root / "runs", exclusions_file=None,
            device=torch.device("cpu"), log=self.logs.append, **options,
        )

    def test_repository_config_matches_experiment_settings(self):
        config = load_config(ROOT / "configs/face-classification.toml")
        self.assertEqual(
            (config.epochs, config.batch_size, config.learning_rate, config.momentum,
             config.weight_decay, config.seed, config.embedding_dim, config.input_size),
            (30, 256, 0.1, 0.9, 5e-4, 42, 512, 64),
        )
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs/project.toml")

    def test_run_records_selection_accuracy_and_checkpoint(self):
        run = self.run_training(tiny_config())
        manifest = json.loads((run / "manifest.json").read_text())
        history = json.loads((run / "history.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(history), 2)
        best = max(history, key=lambda r: r["val_pairs_cosine_auc"])
        self.assertEqual(manifest["selection"]["selected_epoch"], best["epoch"])
        self.assertIsNone(manifest["selection"]["threshold"])
        for split in ("val_data", "test_data"):
            self.assertEqual(manifest["classification_accuracy"][split]["images"], 8 if split == "val_data" else 4)
        model, meta = load_checkpoint(run / "best.pt")
        self.assertEqual(meta["model_version"], manifest["output"]["model_version"])
        self.assertEqual(meta["model_id"], run.name)
        self.assertIn("-classification-", run.name)
        self.assertIn("trained from scratch", manifest["architecture"]["trainable_layers"])
        self.assertEqual(manifest["data"]["identities"], 4)
        self.assertTrue((run / "val-pairs.txt").read_text())

    def test_resume_continues_from_last_epoch_and_refuses_changed_settings(self):
        run = self.run_training(tiny_config(epochs=1))
        with self.assertRaises(ValueError):
            self.run_training(tiny_config(epochs=1, learning_rate=0.2), resume=run)
        # Same settings with a longer schedule are a different run, so resume refuses too.
        with self.assertRaises(ValueError):
            self.run_training(tiny_config(epochs=2), resume=run)
        resumed = self.run_training(tiny_config(epochs=1, num_workers=0), resume=run)
        manifest = json.loads((resumed / "manifest.json").read_text())
        self.assertEqual(manifest["epochs_completed"], 1)
        self.assertEqual(len(manifest["sessions"]), 2)
        self.assertEqual(manifest["sessions"][1]["from_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
