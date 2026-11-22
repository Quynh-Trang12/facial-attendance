"""Multi-frame probe fusion, the collector's states, the gallery match and the settings file."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("numpy is not installed in this environment")

from pathlib import Path
import tempfile

import numpy as np

from facial_attendance.recognition.probe_fusion import (
    Collecting,
    ProbeCollector,
    Ready,
    Retry,
    best_match,
    fuse,
    load_check_in_settings,
)
from facial_attendance.recognition.quality import QualityLimits

TAU = 0.4620
ROOT = Path(__file__).resolve().parents[1]


def unit(*values):
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


class FuseTests(unittest.TestCase):
    def test_identical_vectors_give_that_unit_vector(self):
        fused = fuse([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(fused, [0.6, 0.8])

    def test_each_frame_counts_equally_whatever_its_length(self):
        # Without normalising each frame first, the long vector would dominate.
        fused = fuse([[100.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(fused, unit(1, 1))

    def test_result_is_unit_length(self):
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(float(np.linalg.norm(fuse(rng.normal(size=(5, 512))))), 1.0)

    def test_invalid_input_is_refused(self):
        for embeddings in ([], [[0.0, 0.0]], [[1.0, np.nan]], [[1.0, 0.0], [-1.0, 0.0]], [1.0, 2.0]):
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                fuse(embeddings)


class CollectorTests(unittest.TestCase):
    def collector(self, **settings):
        values = dict(tau=TAU, k=3, min_frames=2, window_s=2.0)
        values.update(settings)
        return ProbeCollector(**values)

    def test_three_consistent_frames_are_ready(self):
        collector = self.collector()
        frames = [unit(1, 0.1, 0), unit(1, 0, 0.1), unit(1, -0.1, 0)]
        self.assertEqual(collector.add(frames[0], (), 0.0), Collecting(1, 3, ()))
        self.assertEqual(collector.add(frames[1], (), 0.3), Collecting(2, 3, ()))
        state = collector.add(frames[2], (), 0.6)
        self.assertIsInstance(state, Ready)
        self.assertEqual(state.frames_used, 3)
        np.testing.assert_allclose(state.probe, fuse(frames))
        self.assertTrue(all(score >= TAU for score in state.consistency))

    def test_failed_frames_are_not_kept_and_report_their_reasons(self):
        collector = self.collector()
        self.assertEqual(collector.add(None, ("blur",), 0.0), Collecting(0, 3, ("blur",)))
        self.assertEqual(collector.add(unit(1, 0), (), 0.1), Collecting(1, 3, ()))

    def test_a_frame_from_someone_else_makes_the_attempt_inconsistent(self):
        collector = self.collector()
        collector.add(unit(1, 0), (), 0.0)
        collector.add(unit(1, 0.05), (), 0.1)
        self.assertEqual(collector.add(unit(-1, 0.2), (), 0.2), Retry("inconsistent"))

    def test_frames_that_cancel_out_are_inconsistent_not_an_error(self):
        collector = self.collector(k=2)
        collector.add(unit(1, 0), (), 0.0)
        self.assertEqual(collector.add(unit(-1, 0), (), 0.1), Retry("inconsistent"))
        self.assertEqual(collector.add(unit(1, 0), (), 0.2), Collecting(1, 2, ()))

    def test_window_end_with_min_frames_is_ready(self):
        collector = self.collector()
        collector.add(unit(1, 0), (), 0.0)
        collector.add(unit(1, 0.1), (), 1.0)
        state = collector.add(None, ("dark",), 2.0)
        self.assertIsInstance(state, Ready)
        self.assertEqual(state.frames_used, 2)

    def test_window_end_below_min_frames_names_the_main_problem(self):
        collector = self.collector()
        collector.add(unit(1, 0), (), 0.0)
        collector.add(None, ("blur", "dark"), 0.5)
        collector.add(None, ("blur",), 1.0)
        self.assertEqual(collector.add(None, ("dark",), 2.5), Retry("blur"))

    def test_ties_between_problems_follow_prompt_priority(self):
        collector = self.collector()
        collector.add(None, ("blur",), 0.0)
        self.assertEqual(collector.add(None, ("turned",), 2.0), Retry("turned"))

    def test_window_end_with_only_good_but_too_few_frames_is_a_timeout(self):
        collector = self.collector(min_frames=3)
        collector.add(unit(1, 0), (), 0.0)
        self.assertEqual(collector.add(unit(1, 0.1), (), 2.0), Retry("timeout"))

    def test_collector_starts_again_after_a_result_or_reset(self):
        collector = self.collector(k=2, min_frames=2)
        collector.add(unit(1, 0), (), 0.0)
        self.assertIsInstance(collector.add(unit(1, 0.1), (), 0.1), Ready)
        # A result clears the kept frames, so the next attempt starts from zero.
        self.assertEqual(collector.add(None, ("blur",), 10.0), Collecting(0, 2, ("blur",)))
        collector.reset()
        # After a reset the window starts at the next frame; the old start would time out.
        self.assertEqual(collector.add(unit(0, 1), (), 50.0), Collecting(1, 2, ()))

    def test_a_kept_frame_needs_an_embedding(self):
        with self.assertRaises(ValueError):
            self.collector().add(None, (), 0.0)

    def test_time_must_not_go_backwards(self):
        collector = self.collector()
        collector.add(unit(1, 0), (), 5.0)
        with self.assertRaises(ValueError):
            collector.add(unit(1, 0), (), 4.0)

    def test_invalid_settings_are_refused(self):
        for settings in ({"k": 0}, {"min_frames": 4}, {"min_frames": 0}, {"window_s": 0.0},
                         {"tau": float("nan")}, {"k": 2.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.collector(**settings)


class BestMatchTests(unittest.TestCase):
    LABELS = ["an", "trang"]
    TEMPLATES = np.stack([unit(1, 0), unit(0, 1)])

    def test_empty_gallery_is_unknown(self):
        match = best_match(unit(1, 0), [], np.zeros((0, 2)), TAU)
        self.assertIsNone(match.label)
        self.assertIsNone(match.score)
        self.assertFalse(match.accepted)

    def test_best_template_above_tau_is_accepted(self):
        match = best_match(unit(1, 0.2), self.LABELS, self.TEMPLATES, TAU)
        self.assertEqual((match.label, match.accepted), ("an", True))
        self.assertAlmostEqual(match.score, float(unit(1, 0.2) @ unit(1, 0)))

    def test_below_tau_is_unknown_but_keeps_the_score(self):
        match = best_match(unit(1, 1.2), self.LABELS, np.stack([unit(1, -1), unit(-1, 1)]), TAU)
        self.assertIsNone(match.label)
        self.assertFalse(match.accepted)
        self.assertLess(match.score, TAU)

    def test_score_equal_to_tau_is_accepted(self):
        probe = np.array([TAU, np.sqrt(1 - TAU ** 2)])
        match = best_match(probe, ["an"], np.array([[1.0, 0.0]]), TAU)
        self.assertEqual(match.score, TAU)
        self.assertTrue(match.accepted)

    def test_labels_must_match_templates(self):
        with self.assertRaises(ValueError):
            best_match(unit(1, 0), ["an"], self.TEMPLATES, TAU)


class SettingsTests(unittest.TestCase):
    def write(self, text):
        directory = tempfile.TemporaryDirectory(prefix="check-in-settings-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "check-in.toml"
        path.write_text(text)
        return path

    def test_repository_config_loads_and_is_off(self):
        settings = load_check_in_settings(ROOT / "configs" / "check-in-quality.toml")
        self.assertFalse(settings.enabled)
        self.assertEqual((settings.k, settings.min_frames, settings.window_s), (3, 2, 2.0))

    def test_limits_are_read_when_present(self):
        path = self.write(
            "[check_in]\nenabled = true\nk = 3\nmin_frames = 2\nwindow_s = 2.0\n"
            "[check_in.limits]\nmin_sharpness = 100\nmin_brightness = 60\nmax_brightness = 200\n"
            "min_contrast = 20\nmin_size = 64\nmax_yaw = 0.25\nmax_roll = 15\n"
        )
        settings = load_check_in_settings(path)
        self.assertTrue(settings.enabled)
        self.assertIsInstance(settings.limits, QualityLimits)
        self.assertEqual(settings.limits.min_size, 64)

    def test_enabled_without_limits_is_refused(self):
        path = self.write("[check_in]\nenabled = true\nk = 3\nmin_frames = 2\nwindow_s = 2.0\n")
        with self.assertRaises(ValueError):
            load_check_in_settings(path)

    def test_misspelled_or_invalid_keys_are_refused(self):
        for text in (
            "[check_in]\nenabled = false\nk = 3\nmin_frame = 2\nwindow_s = 2.0\n",
            "[check_in]\nenabled = \"no\"\nk = 3\nmin_frames = 2\nwindow_s = 2.0\n",
            "[check_in]\nenabled = false\nk = 3\nmin_frames = 5\nwindow_s = 2.0\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_check_in_settings(self.write(text))


if __name__ == "__main__":
    unittest.main()
