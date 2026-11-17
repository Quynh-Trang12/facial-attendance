"""Export checks, paired bootstrap, selection and operating point on generated scores."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from facial_attendance.recognition.comparison import (
    check_export,
    operating_point,
    paired_bootstrap,
    roc_point_at,
    select_approach,
    trial_path_strings,
)


def rank_auc(labels, scores):
    """Mann-Whitney AUC with ties counted as one half."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return greater / (len(pos) * len(neg))


EXPECTED = {"metric": {"model_id": "run-m", "model_version": "abc123abc123"}}


class ExportCheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="face-comparison-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.official = self.root / "official.txt"
        self.official.write_text("verification_data/a.jpg b.jpg 1\na.jpg b.jpg 0\n")
        self.val = self.root / "val.txt"
        self.val.write_text("classification_data/val_data/x/1.jpg classification_data/val_data/y/1.jpg 0\n")
        self.export = self.root / "exports"
        self.export.mkdir()
        self.write_export(["verification_data/a.jpg", "b.jpg", "a.jpg"],
                          ["classification_data/val_data/x/1.jpg", "classification_data/val_data/y/1.jpg"])

    def write_export(self, official, val, model_id="run-m", dim=512):
        (self.export / "export-summary.json").write_text(
            json.dumps({"model_id": model_id, "model_version": "abc123abc123", "classification": None}))
        for name, paths in (("official-embeddings.npz", official), ("val-pairs-embeddings.npz", val)):
            target = self.export / name
            target.unlink(missing_ok=True)
            np.savez(target, paths=np.array(paths), raw_embeddings=np.ones((len(paths), dim), np.float32))

    def check(self):
        return check_export(self.export, "metric", official_trials=self.official, val_pairs=self.val, expected=EXPECTED)

    def test_trial_strings_keep_the_written_prefixes(self):
        self.assertEqual(trial_path_strings(self.official), {"verification_data/a.jpg", "b.jpg", "a.jpg"})

    def test_matching_export_is_accepted_with_file_hashes(self):
        result = self.check()
        self.assertEqual(result["model_id"], "run-m")
        self.assertEqual(result["files"]["official-embeddings.npz"]["images"], 3)
        self.assertEqual(len(result["files"]["val-pairs-embeddings.npz"]["sha256"]), 64)

    def test_wrong_model_is_refused(self):
        self.write_export(["verification_data/a.jpg", "b.jpg", "a.jpg"],
                          ["classification_data/val_data/x/1.jpg", "classification_data/val_data/y/1.jpg"],
                          model_id="other-run")
        with self.assertRaisesRegex(ValueError, "model_id"):
            self.check()

    def test_missing_or_extra_paths_are_refused(self):
        val = ["classification_data/val_data/x/1.jpg", "classification_data/val_data/y/1.jpg"]
        for official in (["verification_data/a.jpg", "b.jpg"], ["verification_data/a.jpg", "b.jpg", "a.jpg", "c.jpg"]):
            self.write_export(official, val)
            with self.assertRaisesRegex(ValueError, "exactly the paths"):
                self.check()

    def test_wrong_embedding_shape_is_refused(self):
        self.write_export(["verification_data/a.jpg", "b.jpg", "a.jpg"],
                          ["classification_data/val_data/x/1.jpg", "classification_data/val_data/y/1.jpg"], dim=128)
        with self.assertRaisesRegex(ValueError, "raw embeddings"):
            self.check()


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.labels = np.repeat([1, 0], 300)
        self.weak = self.labels * 0.5 + rng.normal(0, 1, 600)
        self.strong = self.labels * 2.0 + rng.normal(0, 1, 600)

    def test_identical_scores_give_a_zero_interval(self):
        result = paired_bootstrap(self.labels, self.weak, self.weak, rank_auc, replicates=200)
        self.assertEqual((result["observed"], result["low"], result["high"]), (0.0, 0.0, 0.0))
        self.assertFalse(result["excludes_zero"])

    def test_clear_difference_excludes_zero_and_is_seeded(self):
        result = paired_bootstrap(self.labels, self.weak, self.strong, rank_auc, replicates=200)
        self.assertGreater(result["low"], 0)
        self.assertTrue(result["excludes_zero"])
        self.assertAlmostEqual(result["observed"], rank_auc(self.labels, self.strong) - rank_auc(self.labels, self.weak))
        self.assertEqual(result, paired_bootstrap(self.labels, self.weak, self.strong, rank_auc, replicates=200))
        reversed_ = paired_bootstrap(self.labels, self.strong, self.weak, rank_auc, replicates=200)
        self.assertAlmostEqual(reversed_["high"], -result["low"])

    def test_misaligned_or_single_class_input_is_refused(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.labels[:-1], self.weak, self.strong, rank_auc)
        with self.assertRaises(ValueError):
            paired_bootstrap(np.ones(600, int), self.weak, self.strong, rank_auc)


class DecisionTests(unittest.TestCase):
    def test_higher_cosine_auc_is_selected_and_a_tie_is_not(self):
        self.assertEqual(select_approach({"classification": 0.9884, "metric": 0.9911}), "metric")
        self.assertEqual(select_approach({"metric": 0.97, "classification": 0.98}), "classification")
        with self.assertRaises(ValueError):
            select_approach({"classification": 0.99, "metric": 0.99})

    def test_operating_point_accepts_scores_at_the_threshold(self):
        labels = [1, 1, 1, 0, 0, 0, 0]
        scores = [0.9, 0.5, 0.2, 0.5, 0.4, 0.1, -0.3]
        point = operating_point(labels, scores, 0.5)
        self.assertEqual((point["true_accepts"], point["false_accepts"]), (2, 1))
        self.assertAlmostEqual(point["tar"], 2 / 3)
        self.assertAlmostEqual(point["far"], 1 / 4)

    def test_roc_point_is_the_one_that_accepts_scores_at_or_above_the_threshold(self):
        roc = [{"fpr": 0.0, "tpr": 0.0, "threshold": None}, {"fpr": 0.0, "tpr": 0.4, "threshold": 0.9},
               {"fpr": 0.2, "tpr": 0.8, "threshold": 0.6}, {"fpr": 1.0, "tpr": 1.0, "threshold": 0.1}]
        self.assertEqual(roc_point_at(roc, 0.7)["tpr"], 0.4)
        self.assertEqual(roc_point_at(roc, 0.6)["tpr"], 0.8)
        self.assertEqual(roc_point_at(roc, 0.95)["tpr"], 0.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed in this environment")
class HeadlessExportTests(unittest.TestCase):
    def test_metric_checkpoint_exports_embeddings_without_class_predictions(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts/export_face_embeddings.py"
        spec = importlib.util.spec_from_file_location("export_for_test", script)
        export_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_cli)
        from facial_attendance.recognition.model import FaceEmbeddingNet, build_meta, save_checkpoint
        from test_validation_pairs import build_layout

        directory = tempfile.TemporaryDirectory(prefix="headless-export-fixture-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        layout = build_layout(root / "data", {f"n{i}": 2 for i in range(4)})
        torch.manual_seed(0)
        model = FaceEmbeddingNet(None).eval()
        save_checkpoint(root / "best.pt", model, build_meta(model, model_id="run-m", approach="metric"))
        exclusions = root / "none.tsv"
        exclusions.write_text("# no exclusions\n")
        output = root / "exports"
        with contextlib.redirect_stdout(io.StringIO()):
            export_cli.main(["--data-root", str(layout.root), "--exclusions", str(exclusions),
                         "--checkpoint", str(root / "best.pt"), "--output", str(output), "--device", "cpu"])
        summary = json.loads((output / "export-summary.json").read_text())
        self.assertIsNone(summary["classification"])
        self.assertEqual(summary["official_trials"]["images"], 1)
        self.assertFalse(list(output.glob("*-predictions.csv")))


if __name__ == "__main__":
    unittest.main()
