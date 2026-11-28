"""Export checks, paired bootstrap, selection, operating point and the recheck on generated scores."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("numpy is not installed in this environment")

import contextlib
import io
import json
import tempfile
from pathlib import Path

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
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).sum()
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
        self.val.write_text(
            "classification_data/val_data/x/1.jpg classification_data/val_data/y/1.jpg 0\n"
        )
        self.export = self.root / "exports"
        self.export.mkdir()
        self.write_export(
            ["verification_data/a.jpg", "b.jpg", "a.jpg"],
            [
                "classification_data/val_data/x/1.jpg",
                "classification_data/val_data/y/1.jpg",
            ],
        )

    def write_export(self, official, val, model_id="run-m", dim=512):
        (self.export / "export-summary.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "model_version": "abc123abc123",
                    "classification": None,
                }
            )
        )
        for name, paths in (
            ("official-embeddings.npz", official),
            ("val-pairs-embeddings.npz", val),
        ):
            target = self.export / name
            target.unlink(missing_ok=True)
            np.savez(
                target,
                paths=np.array(paths),
                raw_embeddings=np.ones((len(paths), dim), np.float32),
            )

    def check(self):
        return check_export(
            self.export,
            "metric",
            official_trials=self.official,
            val_pairs=self.val,
            expected=EXPECTED,
        )

    def test_trial_strings_keep_the_written_prefixes(self):
        self.assertEqual(
            trial_path_strings(self.official),
            {"verification_data/a.jpg", "b.jpg", "a.jpg"},
        )

    def test_matching_export_is_accepted_with_file_hashes(self):
        result = self.check()
        self.assertEqual(result["model_id"], "run-m")
        self.assertEqual(result["files"]["official-embeddings.npz"]["images"], 3)
        self.assertEqual(len(result["files"]["val-pairs-embeddings.npz"]["sha256"]), 64)

    def test_wrong_model_is_refused(self):
        self.write_export(
            ["verification_data/a.jpg", "b.jpg", "a.jpg"],
            [
                "classification_data/val_data/x/1.jpg",
                "classification_data/val_data/y/1.jpg",
            ],
            model_id="other-run",
        )
        with self.assertRaisesRegex(ValueError, "model_id"):
            self.check()

    def test_missing_or_extra_paths_are_refused(self):
        val = [
            "classification_data/val_data/x/1.jpg",
            "classification_data/val_data/y/1.jpg",
        ]
        for official in (
            ["verification_data/a.jpg", "b.jpg"],
            ["verification_data/a.jpg", "b.jpg", "a.jpg", "c.jpg"],
        ):
            self.write_export(official, val)
            with self.assertRaisesRegex(ValueError, "exactly the paths"):
                self.check()

    def test_wrong_embedding_shape_is_refused(self):
        self.write_export(
            ["verification_data/a.jpg", "b.jpg", "a.jpg"],
            [
                "classification_data/val_data/x/1.jpg",
                "classification_data/val_data/y/1.jpg",
            ],
            dim=128,
        )
        with self.assertRaisesRegex(ValueError, "raw embeddings"):
            self.check()


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.labels = np.repeat([1, 0], 300)
        self.weak = self.labels * 0.5 + rng.normal(0, 1, 600)
        self.strong = self.labels * 2.0 + rng.normal(0, 1, 600)

    def test_identical_scores_give_a_zero_interval(self):
        result = paired_bootstrap(
            self.labels, self.weak, self.weak, rank_auc, replicates=200
        )
        self.assertEqual(
            (result["observed"], result["low"], result["high"]), (0.0, 0.0, 0.0)
        )
        self.assertFalse(result["excludes_zero"])

    def test_clear_difference_excludes_zero_and_is_seeded(self):
        result = paired_bootstrap(
            self.labels, self.weak, self.strong, rank_auc, replicates=200
        )
        self.assertGreater(result["low"], 0)
        self.assertTrue(result["excludes_zero"])
        self.assertAlmostEqual(
            result["observed"],
            rank_auc(self.labels, self.strong) - rank_auc(self.labels, self.weak),
        )
        self.assertEqual(
            result,
            paired_bootstrap(
                self.labels, self.weak, self.strong, rank_auc, replicates=200
            ),
        )
        reversed_ = paired_bootstrap(
            self.labels, self.strong, self.weak, rank_auc, replicates=200
        )
        self.assertAlmostEqual(reversed_["high"], -result["low"])

    def test_misaligned_or_single_class_input_is_refused(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.labels[:-1], self.weak, self.strong, rank_auc)
        with self.assertRaises(ValueError):
            paired_bootstrap(np.ones(600, int), self.weak, self.strong, rank_auc)


class DecisionTests(unittest.TestCase):
    def test_higher_cosine_auc_is_selected_and_a_tie_is_not(self):
        self.assertEqual(
            select_approach({"classification": 0.9884, "metric": 0.9911}), "metric"
        )
        self.assertEqual(
            select_approach({"metric": 0.97, "classification": 0.98}), "classification"
        )
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
        roc = [
            {"fpr": 0.0, "tpr": 0.0, "threshold": None},
            {"fpr": 0.0, "tpr": 0.4, "threshold": 0.9},
            {"fpr": 0.2, "tpr": 0.8, "threshold": 0.6},
            {"fpr": 1.0, "tpr": 1.0, "threshold": 0.1},
        ]
        self.assertEqual(roc_point_at(roc, 0.7)["tpr"], 0.4)
        self.assertEqual(roc_point_at(roc, 0.6)["tpr"], 0.8)
        self.assertEqual(roc_point_at(roc, 0.95)["tpr"], 0.0)


def load_script(name):
    script = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def brute_threshold(labels, scores, target=0.01):
    """Lowest observed score whose FAR at score >= it is at most the target, by counting."""
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    for score in sorted(set(scores)):
        if sum(n >= score for n in negatives) / len(negatives) <= target:
            return score
    return None


class RecheckScriptTests(unittest.TestCase):
    """The recheck passes on consistent records and names the check that a fault breaks."""

    def setUp(self):
        from facial_attendance.recognition.comparison import (
            EXPECTED_MODELS,
            OFFICIAL_TRIALS_SHA256,
            VAL_PAIRS_SHA256,
            file_sha256,
        )

        self.check = load_script("check_face_comparison")
        self.models, self.sha = EXPECTED_MODELS, file_sha256
        self.trials = {
            "validation": VAL_PAIRS_SHA256,
            "official": OFFICIAL_TRIALS_SHA256,
        }
        directory = tempfile.TemporaryDirectory(prefix="face-recheck-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(1)
        labels = np.repeat([1, 0], 200)
        self.scores = {}
        for step in ("validation", "official"):
            keys = [
                (f"{step}/{i}a.jpg", f"{step}/{i}b.jpg") for i in range(len(labels))
            ]
            for approach, gap in (("classification", 1.2), ("metric", 1.8)):
                cosine = np.tanh((labels * gap + rng.normal(0, 1, len(labels))) / 3)
                distance = np.abs(
                    1.5 - labels * gap * 0.5 + rng.normal(0, 0.5, len(labels))
                )
                self.scores[step, approach] = (keys, labels, cosine, distance)
        self.write_records()

    def write_run(self, step, approach):
        keys, labels, cosine, distance = self.scores[step, approach]
        folder = self.root / step / f"{step}-{approach}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = ["left,right,label,cosine,euclidean_distance,negative_euclidean"]
        rows += [
            f"{left},{right},{y},{c!r},{d!r},{-d!r}"
            for (left, right), y, c, d in zip(
                keys, labels.tolist(), cosine.tolist(), distance.tolist()
            )
        ]
        (folder / "pair_scores.csv").write_text("\n".join(rows) + "\n")
        roc = [{"fpr": 0.0, "tpr": 0.0, "threshold": None}]
        for t in sorted(set(cosine.tolist()), reverse=True):
            point = operating_point(labels, cosine, t)
            roc.append({"fpr": point["far"], "tpr": point["tar"], "threshold": t})
        auc = {
            "cosine": rank_auc(labels, cosine),
            "negative_euclidean": rank_auc(labels, -distance),
        }
        model = self.models[approach]
        metrics = {
            "mode": step,
            "run_id": model["model_id"],
            "checkpoint_id": model["model_version"],
            "git_commit": self.check.EVALUATOR_COMMIT,
            "git_tracked_dirty": False,
            "inputs": {"trials": {"sha256": self.trials[step]}},
            "metrics": {
                "cosine": {"auc": auc["cosine"], "roc": roc},
                "negative_euclidean": {"auc": auc["negative_euclidean"]},
            },
        }
        (folder / "metrics.json").write_text(json.dumps(metrics))
        return {
            "auc": auc,
            "count": len(labels),
            "positive_count": int(labels.sum()),
            "negative_count": int((labels == 0).sum()),
            "output": folder.name,
            "metrics_sha256": self.sha(folder / "metrics.json"),
            "pair_scores_sha256": self.sha(folder / "pair_scores.csv"),
        }

    def bootstrap(self, step, name):
        labels = self.scores[step, "classification"][1]
        column = {"cosine": lambda s: s[2], "negative_euclidean": lambda s: -s[3]}[name]
        first, second = (
            column(self.scores[step, a]) for a in ("classification", "metric")
        )
        return paired_bootstrap(labels, first, second, rank_auc)

    def write_records(self, threshold_shift=0):
        validation = {}
        for approach in ("classification", "metric"):
            validation[approach] = self.write_run("validation", approach)
            _, labels, cosine, _ = self.scores["validation", approach]
            tau = brute_threshold(labels.tolist(), cosine.tolist())
            point = operating_point(labels, cosine, tau)
            validation[approach]["threshold_selection"] = {
                "threshold": tau,
                "far": point["far"],
                "tar": point["tar"],
            }
        chosen = validation["metric"]["threshold_selection"]
        candidates = sorted(set(self.scores["validation", "metric"][2].tolist()))
        tau = candidates[candidates.index(chosen["threshold"]) + threshold_shift]
        self.selection = {
            "created_utc": "2026-01-01T00:00:00+00:00",
            "validation": validation,
            "bootstrap": {"cosine": self.bootstrap("validation", "cosine")},
            "decision": {
                "selected_approach": "metric",
                "selected_model_id": self.models["metric"]["model_id"],
                "selected_model_version": self.models["metric"]["model_version"],
                "threshold": tau,
                "validation_far": chosen["far"],
                "validation_tar": chosen["tar"],
                "reject_all": False,
            },
        }
        (self.root / "selection.json").write_text(json.dumps(self.selection))
        official = {
            a: self.write_run("official", a) for a in ("classification", "metric")
        }
        _, labels, cosine, _ = self.scores["official", "metric"]
        point = operating_point(labels, cosine, tau)
        summary = {
            "created_utc": "2026-01-02T00:00:00+00:00",
            "comparison_code": {"commit": "0" * 40},
            "selection_sha256": self.sha(self.root / "selection.json"),
            "decision_unchanged": self.selection["decision"],
            "official": official,
            "bootstrap": {
                n: self.bootstrap("official", n)
                for n in ("cosine", "negative_euclidean")
            },
            "operating_point": {"approach": "metric", **point},
        }
        (self.root / "official-summary.json").write_text(json.dumps(summary))

    def run_check(self):
        report = self.root / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.check.main(
                [
                    "--selection",
                    str(self.root / "selection.json"),
                    "--validation-dir",
                    str(self.root / "validation"),
                    "--official-summary",
                    str(self.root / "official-summary.json"),
                    "--official-dir",
                    str(self.root / "official"),
                    "--output",
                    str(report),
                ]
            )
        checks = json.loads(report.read_text())["checks"]
        return code, {c["check"] for c in checks if c["ok"] is False}, checks

    def test_rank_auc_matches_pairwise_count_with_ties(self):
        labels = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([0.9, 0.5, 0.5, 0.1, 0.5, 0.7])
        self.assertAlmostEqual(
            self.check.rank_auc(labels, scores), rank_auc(labels, scores)
        )

    def test_far_threshold_moves_tied_scores_together(self):
        labels = [0] * 99 + [0, 1, 1]
        scores = [0.1] * 98 + [0.8, 0.8, 0.8, 0.9]
        rule = self.check.far_threshold(labels, scores)
        self.assertEqual(rule["threshold"], 0.9)
        self.assertAlmostEqual(rule["lower_far"], 2 / 100)

    def test_consistent_records_pass(self):
        code, failed, checks = self.run_check()
        self.assertEqual((code, failed), (0, set()))
        self.assertEqual(
            sum(c["ok"] is None for c in checks), 1
        )  # the commit order needs the real history

    def test_a_threshold_above_the_rule_fails(self):
        self.write_records(threshold_shift=1)
        code, failed, _ = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn(
            "validation metric: threshold is the lowest cosine with FAR <= 1%", failed
        )

    def test_an_edited_score_file_fails_its_hash(self):
        scores = self.root / "official" / "official-classification" / "pair_scores.csv"
        scores.write_text(scores.read_text().replace(",1,", ",0,", 1))
        code, failed, _ = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn(
            "official classification: output files are the recorded ones", failed
        )

    def test_a_distance_passed_without_negation_fails(self):
        keys, labels, cosine, distance = self.scores["official", "metric"]
        self.scores["official", "metric"] = (keys, labels, cosine, -distance)
        self.write_records()
        code, failed, _ = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn(
            "official metric: score direction (cosine in [-1, 1], AUC input is minus the distance)",
            failed,
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "torch is not installed in this environment"
)
class HeadlessExportTests(unittest.TestCase):
    def test_metric_checkpoint_exports_embeddings_without_class_predictions(self):
        import torch

        script = (
            Path(__file__).resolve().parents[1] / "scripts/export_face_embeddings.py"
        )
        spec = importlib.util.spec_from_file_location("export_for_test", script)
        export_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_cli)
        from facial_attendance.recognition.model import (
            FaceEmbeddingNet,
            build_meta,
            save_checkpoint,
        )
        from tests.test_validation_pairs import build_layout

        directory = tempfile.TemporaryDirectory(prefix="headless-export-fixture-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        layout = build_layout(root / "data", {f"n{i}": 2 for i in range(4)})
        torch.manual_seed(0)
        model = FaceEmbeddingNet(None).eval()
        save_checkpoint(
            root / "best.pt",
            model,
            build_meta(model, model_id="run-m", approach="metric"),
        )
        exclusions = root / "none.tsv"
        exclusions.write_text("# no exclusions\n")
        output = root / "exports"
        with contextlib.redirect_stdout(io.StringIO()):
            export_cli.main(
                [
                    "--data-root",
                    str(layout.root),
                    "--exclusions",
                    str(exclusions),
                    "--checkpoint",
                    str(root / "best.pt"),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                ]
            )
        summary = json.loads((output / "export-summary.json").read_text())
        self.assertIsNone(summary["classification"])
        self.assertEqual(summary["official_trials"]["images"], 1)
        self.assertFalse(list(output.glob("*-predictions.csv")))


if __name__ == "__main__":
    unittest.main()
