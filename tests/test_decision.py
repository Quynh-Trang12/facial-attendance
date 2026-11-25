"""The gallery decision names a person only when the best match is strong and unambiguous."""

from pathlib import Path
import tempfile
import unittest

from facial_attendance.recognition.decision import (
    ACCEPTED,
    AMBIGUOUS,
    BELOW_THRESHOLD,
    EMPTY_GALLERY,
    decide,
    load_margin,
    threshold_from_meta,
)

ROOT = Path(__file__).resolve().parents[1]
# Values exactly representable in binary, so boundary cases are not rounding accidents.
TAU = 0.5
MARGIN = 0.125


class DecideTests(unittest.TestCase):
    def test_accepts_clear_best_match(self):
        decision = decide(["an", "lynn", "que"], [0.75, 0.5, 0.25], TAU, MARGIN)
        self.assertEqual((decision.label, decision.accepted, decision.reason), ("an", True, ACCEPTED))
        self.assertEqual((decision.best_score, decision.runner_up_score), (0.75, 0.5))

    def test_rejects_best_score_below_threshold(self):
        decision = decide(["an", "lynn"], [0.375, 0.0], TAU, MARGIN)
        self.assertIsNone(decision.label)
        self.assertEqual(decision.reason, BELOW_THRESHOLD)
        # The closest person is still reported for display and logging.
        self.assertEqual(decision.best_label, "an")

    def test_rejects_best_match_too_close_to_runner_up(self):
        # Both above tau: the look-alike case the threshold alone lets through.
        decision = decide(["an", "lynn"], [0.625, 0.546875], TAU, MARGIN)
        self.assertIsNone(decision.label)
        self.assertEqual(decision.reason, AMBIGUOUS)

    def test_margin_boundary_is_inclusive(self):
        self.assertTrue(decide(["an", "lynn"], [0.75, 0.625], TAU, MARGIN).accepted)
        self.assertFalse(decide(["an", "lynn"], [0.75, 0.6875], TAU, MARGIN).accepted)

    def test_threshold_boundary_is_inclusive(self):
        self.assertTrue(decide(["an", "lynn"], [0.5, 0.25], TAU, MARGIN).accepted)

    def test_single_person_gallery_uses_tau_as_runner_up(self):
        # Between tau and tau + margin: unknown, so a look-alike stranger is not named.
        between = decide(["an"], [0.5625], TAU, MARGIN)
        self.assertEqual((between.label, between.reason), (None, AMBIGUOUS))
        self.assertEqual(between.runner_up_score, TAU)
        self.assertEqual(decide(["an"], [0.625], TAU, MARGIN).label, "an")
        self.assertEqual(decide(["an"], [0.375], TAU, MARGIN).reason, BELOW_THRESHOLD)

    def test_templates_of_the_same_person_do_not_compete(self):
        # Two templates of an: s2 must come from lynn, not an's second template.
        decision = decide(["an", "an", "lynn"], [0.75, 0.74, 0.25], TAU, MARGIN)
        self.assertEqual(decision.label, "an")
        self.assertEqual(decision.runner_up_score, 0.25)
        # One person with several templates is still a one-person gallery.
        self.assertEqual(decide(["an", "an"], [0.5625, 0.55], TAU, MARGIN).reason, AMBIGUOUS)

    def test_tie_between_two_people_is_unknown_even_without_margin(self):
        for margin in (MARGIN, 0.0):
            with self.subTest(margin=margin):
                decision = decide(["an", "lynn"], [0.75, 0.75], TAU, margin)
                self.assertEqual((decision.label, decision.reason), (None, AMBIGUOUS))
                self.assertEqual(decision.best_label, "an")

    def test_zero_margin_keeps_the_threshold_only_rule(self):
        self.assertEqual(decide(["an", "lynn"], [0.75, 0.74], TAU, 0.0).label, "an")
        self.assertEqual(decide(["an"], [0.5], TAU, 0.0).label, "an")

    def test_empty_gallery_is_unknown(self):
        decision = decide([], [], TAU, MARGIN)
        self.assertEqual((decision.label, decision.reason), (None, EMPTY_GALLERY))

    def test_rejects_invalid_inputs(self):
        cases = [
            (["an"], [0.5, 0.25], TAU, MARGIN),
            (["an"], [float("nan")], TAU, MARGIN),
            (["an"], [0.5], None, MARGIN),
            (["an"], [0.5], TAU, -0.1),
            (["an"], [True], TAU, MARGIN),
        ]
        for labels, scores, tau, margin in cases:
            with self.subTest(scores=scores, tau=tau, margin=margin):
                with self.assertRaises(ValueError):
                    decide(labels, scores, tau, margin)


class SettingsTests(unittest.TestCase):
    def test_threshold_comes_from_checkpoint_meta(self):
        self.assertEqual(threshold_from_meta({"threshold": 0.462}), 0.462)
        for meta in ({}, {"threshold": None}, {"threshold": "0.462"}):
            with self.subTest(meta=meta):
                with self.assertRaises(ValueError):
                    threshold_from_meta(meta)

    def test_repository_margin_is_the_measured_value(self):
        self.assertEqual(load_margin(ROOT / "configs/recognition.toml"), 0.15)

    def test_margin_config_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "recognition.toml"
            for body in ("", "[decision]\n", "[decision]\nmargin = -0.1\n",
                         "[decision]\nmargin = \"0.15\"\n", "[decision]\nmargin = 0.15\nextra = 1\n"):
                with self.subTest(body=body):
                    config.write_text(body, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_margin(config)


if __name__ == "__main__":
    unittest.main()
