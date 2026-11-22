"""Error-versus-discard and k-shot arithmetic on generated scores and embeddings."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("numpy is not installed in this environment")

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from facial_attendance.recognition.check_in_evaluation import (
    discard_curve,
    image_scores,
    kshot,
    limit_percentiles,
    operating_point,
    pair_quality,
    random_discard,
    rank_auc,
    select_measure,
)

ROOT = Path(__file__).resolve().parents[1]
TAU = 0.5


def brute_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def degraded_pairs(n=400, seed=12345):
    """Genuine pairs whose low-quality half scores low: quality predicts the errors.

    The seed must differ from the random-discard seeds (0..19), or one baseline run
    would draw exactly this quality and copy the informative ranking.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([1, 0], n // 2)
    quality = rng.random(n)
    scores = np.where(labels == 1, 0.8, 0.1) + rng.normal(0, 0.05, n)
    bad_genuine = (labels == 1) & (quality < 0.5)
    scores[bad_genuine] = 0.3 + rng.normal(0, 0.05, bad_genuine.sum())
    return labels, scores, quality


class MetricTests(unittest.TestCase):
    def test_rank_auc_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 300)
        scores = rng.integers(0, 20, 300) / 10  # many ties
        self.assertAlmostEqual(rank_auc(labels, scores), brute_auc(labels, scores))

    def test_rank_auc_needs_both_classes(self):
        with self.assertRaises(ValueError):
            rank_auc([1, 1], [0.2, 0.3])

    def test_operating_point_accepts_at_tau(self):
        point = operating_point([1, 1, 0, 0], [0.5, 0.4, 0.5, 0.1], 0.5)
        self.assertEqual((point["tar"], point["far"]), (0.5, 0.5))
        self.assertEqual((point["positives"], point["negatives"]), (2, 2))


class QualityScoreTests(unittest.TestCase):
    ROWS = [
        {"path": "a", "sharpness": 10, "brightness": 128, "contrast": 50, "size": 64},
        {"path": "b", "sharpness": 300, "brightness": 30, "contrast": 50, "size": 64},
        {"path": "c", "sharpness": 200, "brightness": 140, "contrast": 10, "size": 112},
    ]

    def test_higher_is_better_and_brightness_is_distance_from_mid_grey(self):
        scores = image_scores(self.ROWS)
        self.assertGreater(scores["sharpness"]["b"], scores["sharpness"]["a"])
        self.assertGreater(scores["brightness"]["a"], scores["brightness"]["c"])
        self.assertGreater(scores["brightness"]["c"], scores["brightness"]["b"])

    def test_combined_is_the_mean_rank_and_ties_share_a_rank(self):
        combined = image_scores(self.ROWS)["combined"]
        # Ranks (0..1): sharpness a0 b1 c.5; brightness a1 b0 c.5; contrast a.75 b.75 c0.
        self.assertAlmostEqual(combined["a"], (0 + 1 + 0.75) / 3)
        self.assertAlmostEqual(combined["b"], (1 + 0 + 0.75) / 3)
        self.assertAlmostEqual(combined["c"], (0.5 + 0.5 + 0) / 3)

    def test_duplicate_images_are_refused(self):
        with self.assertRaises(ValueError):
            image_scores(self.ROWS + self.ROWS[:1])

    def test_pair_quality_is_the_worse_image(self):
        scores = {"a": 1.0, "b": 3.0, "c": 2.0}
        np.testing.assert_array_equal(pair_quality([("a", "b"), ("b", "c")], scores), [1.0, 2.0])
        with self.assertRaises(ValueError):
            pair_quality([("a", "z")], scores)

    def test_percentiles_cover_each_measure(self):
        result = limit_percentiles(self.ROWS)
        self.assertEqual(set(result), {"sharpness", "brightness", "contrast", "size"})
        self.assertLessEqual(result["sharpness"]["p5"], result["sharpness"]["p95"])


class DiscardTests(unittest.TestCase):
    def test_zero_fraction_keeps_everything(self):
        labels, scores, quality = degraded_pairs()
        point = discard_curve(labels, scores, quality, TAU)[0]
        self.assertEqual(point["kept"], len(labels))
        self.assertEqual(point["tar"], operating_point(labels, scores, TAU)["tar"])

    def test_dropping_low_quality_pairs_raises_tar_more_than_random(self):
        labels, scores, quality = degraded_pairs()
        curve = discard_curve(labels, scores, quality, TAU)
        baseline = random_discard(labels, scores, TAU, seeds=5)
        self.assertEqual([p["kept"] for p in curve], [400, 380, 360, 340, 320, 280])
        self.assertGreater(curve[-1]["tar"], curve[0]["tar"])
        self.assertGreater(curve[-1]["tar"], baseline[-1]["tar"])
        self.assertEqual(baseline[0]["seeds"], 5)

    def test_random_baseline_is_reproducible(self):
        labels, scores, _ = degraded_pairs()
        self.assertEqual(random_discard(labels, scores, TAU, seeds=3), random_discard(labels, scores, TAU, seeds=3))

    def test_selection_picks_the_informative_measure(self):
        # Enough pairs that random drops barely move TAR, as with the real 7,612 pairs.
        labels, scores, quality = degraded_pairs(n=4000)
        noise = np.random.default_rng(9).random(len(labels))
        baseline = random_discard(labels, scores, TAU, seeds=5)
        choice = select_measure({"informative": discard_curve(labels, scores, quality, TAU),
                                 "noise": discard_curve(labels, scores, noise, TAU)}, baseline)
        self.assertEqual(choice["measure"], "informative")
        self.assertTrue(choice["beats_random"])

    def test_a_tie_goes_to_the_single_measure_before_combined(self):
        labels, scores, quality = degraded_pairs()
        curve = discard_curve(labels, scores, quality, TAU)
        baseline = random_discard(labels, scores, TAU, seeds=2)
        self.assertEqual(select_measure({"combined": curve, "contrast": curve}, baseline)["measure"], "contrast")


class KshotTests(unittest.TestCase):
    def identities(self, count=30, images=4, noise=0.9, seed=3):
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(count, 64))
        return {f"id{i:03d}": centres[i] + noise * rng.normal(size=(images, 64)) * np.linalg.norm(centres[i]) / 8
                for i in range(count)}

    def test_fusing_more_images_reaches_the_gallery_more_often(self):
        result = kshot(self.identities(), tau=0.6)
        tar = [result["k"][str(k)]["mean"]["tar"] for k in (1, 2, 3)]
        self.assertLess(tar[0], tar[2])
        self.assertEqual(result["identities"], 30)
        self.assertEqual(len(result["k"]["1"]["folds"]), 4)

    def test_identities_with_too_few_images_are_counted_not_used(self):
        data = self.identities()
        data["short"] = data["id000"][:3]
        result = kshot(data, tau=0.6)
        self.assertEqual((result["identities"], result["skipped_identities"]), (30, 1))

    def test_impostor_count_is_every_other_gallery(self):
        result = kshot(self.identities(count=5), tau=2.0)  # nothing is accepted
        fold = result["k"]["1"]["folds"][0]
        self.assertEqual((fold["positives"], fold["negatives"]), (5, 20))
        self.assertEqual((fold["tar"], fold["far"]), (0.0, 0.0))

    def test_needs_two_identities(self):
        with self.assertRaises(ValueError):
            kshot({"one": np.ones((4, 3))}, tau=0.5)


class DiscardCommandTests(unittest.TestCase):
    """The validation step must exist before the official step, which reuses its choice."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-evaluation-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        labels, scores, quality = degraded_pairs(n=200)
        self.pairs, self.quality = self.root / "pair_scores.csv", self.root / "quality.csv"
        with self.pairs.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["left", "right", "label", "cosine", "euclidean_distance", "negative_euclidean"])
            for i, (label, score) in enumerate(zip(labels, scores)):
                writer.writerow([f"img{i}a.jpg", f"img{i}b.jpg", label, score, 0, 0])
        with self.quality.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "sharpness", "brightness", "contrast", "size"])
            for i, value in enumerate(quality):
                writer.writerow([f"img{i}a.jpg", 1000 * value, 128, 41, 64])
                writer.writerow([f"img{i}b.jpg", 1000, 128, 41, 64])

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_check_in_quality.py"), *map(str, args)],
                              capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src")})

    def test_validation_then_official(self):
        common = ["--pair-scores", self.pairs, "--quality", self.quality, "--tau", TAU, "--no-plot"]
        official = self.run_script("discard", "official", *common, "--output", self.root / "official")
        self.assertNotEqual(official.returncode, 0)
        self.assertIn("--selection", official.stderr)

        validation = self.run_script("discard", "validation", *common, "--output", self.root / "validation")
        self.assertEqual(validation.returncode, 0, validation.stderr)
        selection = json.loads((self.root / "validation/discard-validation.json").read_text())
        self.assertEqual(selection["selection"]["measure"], "sharpness")
        self.assertIn("limits_start", selection)

        official = self.run_script("discard", "official", *common, "--output", self.root / "official",
                                   "--selection", self.root / "validation/discard-validation.json")
        self.assertEqual(official.returncode, 0, official.stderr)
        record = json.loads((self.root / "official/discard-official.json").read_text())
        self.assertEqual(record["measure"], "sharpness")
        self.assertEqual(record["pair_scores_sha256"], selection["pair_scores_sha256"])
        self.assertEqual(len(record["selected_curve"]), 6)


if __name__ == "__main__":
    unittest.main()
