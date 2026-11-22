"""Live check-in logic with generated frames, fake detections and a fake embedder."""

import importlib.util
import unittest

# The fused mode measures frames with OpenCV; the dependency jobs have both.
if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("numpy or OpenCV is not installed in this environment")

import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np

from facial_attendance.recognition.check_in_live import (
    CheckIn,
    Face,
    Gallery,
    append_log,
    calibrate_limits,
    crop,
    faces_from_yunet,
    limits_toml,
    summarize,
)
from facial_attendance.recognition.probe_fusion import CheckInSettings
from facial_attendance.recognition.quality import QualityLimits

TAU = 0.4620
META = {"model_id": "run-m", "model_version": "abc123abc123", "threshold": TAU}
LIMITS = QualityLimits(min_sharpness=100.0, min_brightness=60.0, max_brightness=200.0, min_contrast=20.0,
                       min_size=64, max_yaw=0.25, max_roll=15.0)
SETTINGS = CheckInSettings(enabled=True, k=3, min_frames=2, window_s=2.0, limits=LIMITS)
FRONTAL = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float64)


def unit(*values):
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def frame(blur=False):
    """A 160x160 RGB frame whose face area is a sharp mid-grey checkerboard."""
    tiles = (np.indices((160, 160)) // 8).sum(axis=0) % 2
    grey = np.where(tiles == 1, 200, 60).astype(np.uint8)
    image = np.repeat(grey[:, :, None], 3, axis=2)
    return cv2.GaussianBlur(image, (0, 0), 6) if blur else image


FACE = Face((10, 10, 138, 138), FRONTAL + 10)


def gallery_with(*people):
    gallery = Gallery(META["model_id"], META["model_version"], TAU)
    for label, direction in people:
        gallery.enroll(label, [direction] * 5)
    return gallery


class DetectionTests(unittest.TestCase):
    def test_yunet_rows_are_clipped_like_the_gui(self):
        row = [-3.4, 5.2, 50.0, 60.0, *FRONTAL.ravel(), 0.95]
        (face,) = faces_from_yunet(np.array([row]), width=40, height=200)
        self.assertEqual(face.box, (0, 5, 40, 66))
        np.testing.assert_array_equal(face.landmarks, FRONTAL)

    def test_empty_or_outside_boxes_are_dropped(self):
        self.assertEqual(faces_from_yunet(None, 10, 10), [])
        outside = [20.0, 20.0, 5.0, 5.0, *FRONTAL.ravel(), 0.95]
        self.assertEqual(faces_from_yunet(np.array([outside]), 10, 10), [])

    def test_crop_uses_the_box(self):
        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        self.assertEqual(crop(image, Face((10, 20, 30, 60), None)).shape, (40, 20, 3))


class GalleryTests(unittest.TestCase):
    def test_enrollment_averages_and_matches(self):
        gallery = gallery_with(("An", unit(1, 0, 0)))
        match = gallery.match(unit(1, 0.1, 0))
        self.assertEqual((match.label, match.accepted), ("An", True))

    def test_mixed_capture_duplicate_face_and_taken_name_are_refused(self):
        gallery = gallery_with(("An", unit(1, 0, 0)))
        with self.assertRaisesRegex(ValueError, "do not agree"):
            gallery.enroll("Trang", [unit(0, 1, 0)] * 4 + [unit(0, -1, 0.2)])
        with self.assertRaisesRegex(ValueError, "already matches 'An'"):
            gallery.enroll("Someone", [unit(1, 0.05, 0)] * 5)
        with self.assertRaisesRegex(ValueError, "already registered"):
            gallery.enroll("an", [unit(0, 0, 1)] * 5)
        self.assertEqual(gallery.labels, ["An"])

    def test_save_load_and_refuse_another_checkpoint(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-gallery-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gallery.npz"
        self.assertEqual(Gallery.load_or_new(path, META).labels, [])
        gallery = gallery_with(("An", unit(1, 0, 0)), ("Vy", unit(0, 1, 0)))
        gallery.save(path)
        loaded = Gallery.load_or_new(path, META)
        self.assertEqual(loaded.labels, ["An", "Vy"])
        np.testing.assert_allclose(loaded.templates, gallery.templates)
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            Gallery.load_or_new(path, META | {"model_version": "000000000000"})
        Gallery(META["model_id"], META["model_version"], TAU).save(path)
        self.assertEqual(Gallery.load_or_new(path, META).labels, [])


class CheckInTests(unittest.TestCase):
    def setUp(self):
        self.gallery = gallery_with(("An", unit(1, 0, 0)), ("Vy", unit(0, 1, 0)))
        self.calls = 0

    def embed_as(self, direction):
        def embed(face_crop):
            self.calls += 1
            return direction
        return embed

    def test_single_mode_decides_on_the_first_face_frame(self):
        check_in = CheckIn("single", self.gallery, self.embed_as(unit(1, 0.2, 0)))
        self.assertEqual(check_in.step(frame(), [FACE], 0.0).prompt, "Press SPACE to check in")
        check_in.start(1.0)
        self.assertEqual(check_in.step(frame(), [], 1.1).prompt, "No face found")
        display = check_in.step(frame(blur=True), [FACE], 1.3)
        self.assertEqual((display.decision.label, display.decision.frames_used, display.decision.end), ("An", 1, "match"))
        self.assertAlmostEqual(display.decision.elapsed_s, 0.3)
        self.assertFalse(check_in.active)

    def test_fused_mode_skips_bad_frames_without_embedding_them(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0.2, 0)), settings=SETTINGS)
        check_in.start(0.0)
        self.assertEqual(check_in.step(frame(blur=True), [FACE], 0.1).prompt, "Hold still")
        self.assertEqual(self.calls, 0)
        self.assertEqual(check_in.step(frame(), [FACE], 0.2).kept, 1)
        self.assertEqual(check_in.step(frame(), [FACE, FACE], 0.3).prompt, "One person at a time")
        check_in.step(frame(), [FACE], 0.4)
        check_in.step(frame(), [FACE], 0.5)
        display = check_in.step(frame(), [FACE], 0.6)
        self.assertEqual((display.decision.label, display.decision.frames_used), ("An", 3))
        self.assertEqual(self.calls, 4)

    def test_turned_head_is_refused_and_named(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)), settings=SETTINGS)
        check_in.start(0.0)
        turned = Face(FACE.box, FACE.landmarks + np.array([[0, 0], [0, 0], [20, 0], [0, 0], [0, 0]]))
        self.assertEqual(check_in.step(frame(), [turned], 0.1).prompt, "Face the camera")

    def test_unknown_person_is_not_recognised(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(0, 0, 1)), settings=SETTINGS)
        check_in.start(0.0)
        for moment in (0.1, 0.2):
            check_in.step(frame(), [FACE], moment)
        decision = check_in.step(frame(), [FACE], 0.3).decision
        self.assertEqual((decision.label, decision.accepted, decision.end), (None, False, "unknown"))

    def test_retry_continues_until_the_attempt_times_out(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)), settings=SETTINGS, timeout_s=5.0)
        check_in.start(0.0)
        check_in.step(frame(blur=True), [FACE], 0.0)
        self.assertEqual(check_in.step(frame(blur=True), [FACE], 2.0).prompt, "Hold still")
        self.assertTrue(check_in.active)
        decision = check_in.step(frame(), [FACE], 5.0).decision
        self.assertEqual((decision.end, decision.retries), ("timeout", 1))

    def test_fused_mode_needs_limits(self):
        with self.assertRaises(ValueError):
            CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)),
                    settings=CheckInSettings(False, 3, 2, 2.0, None))


class LogTests(unittest.TestCase):
    def test_log_and_summary(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-log-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "attempts.csv"
        gallery = gallery_with(("An", unit(1, 0, 0)))
        for mode, person, direction in (("single", "An", unit(1, 0, 0)), ("single", "An", unit(0, 1, 0)),
                                        ("single", "impostor", unit(1, 0.1, 0)), ("fused", "An", unit(1, 0, 0))):
            check_in = CheckIn("single", gallery, lambda _crop, d=direction: d)
            check_in.start(0.0)
            decision = check_in.step(frame(), [FACE], 0.5).decision
            append_log(path, person, decision.__class__(**{**decision.__dict__, "mode": mode}))
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        summary = summarize(rows)
        self.assertEqual(summary["single"]["genuine_accept_rate"], 0.5)
        self.assertEqual(summary["single"]["impostor_accept_rate"], 1.0)
        self.assertEqual(summary["single"]["decision_ms_median"], 500.0)
        self.assertEqual(summary["fused"]["people"], ["An"])
        self.assertIsNone(summary["fused"]["impostor_accept_rate"])


class CalibrationTests(unittest.TestCase):
    def rows(self, count, sharpness, yaw, seed):
        rng = np.random.default_rng(seed)
        return [{"sharpness": sharpness + rng.normal(0, 20), "brightness": 128 + rng.normal(0, 10),
                 "contrast": 50 + rng.normal(0, 5), "size": 120, "yaw": yaw + rng.normal(0, 0.03),
                 "roll": rng.normal(0, 3)} for _ in range(count)]

    def test_limits_keep_most_good_frames_and_reject_bad_ones(self):
        good = self.rows(200, 400, 0.0, 1)
        bad = self.rows(100, 80, 0.5, 2)
        result = calibrate_limits(good, bad, min_size=64)
        # Together, not each, the limits keep about 95 % of good frames.
        self.assertGreaterEqual(result["good_pass_rate"], 0.9)
        self.assertGreater(result["bad_reject_rate"], 0.95)
        self.assertEqual(result["limits"]["min_size"], 64)
        limits = QualityLimits.from_mapping(result["limits"])
        self.assertLess(limits.min_sharpness, 400)

    def test_toml_table_round_trips(self):
        import tomllib

        result = calibrate_limits(self.rows(50, 400, 0.0, 3), self.rows(10, 80, 0.5, 4), min_size=64)
        text = "[check_in]\nenabled = false\nk = 3\nmin_frames = 2\nwindow_s = 2.0\n" + limits_toml(
            result["limits"], "calibration.json")
        table = tomllib.loads(text)["check_in"]["limits"]
        self.assertEqual(QualityLimits.from_mapping(table).min_size, 64)


if __name__ == "__main__":
    unittest.main()
