"""Face crop quality measures, limits and prompts on generated images and landmarks."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("numpy or OpenCV is not installed in this environment")

import cv2
import numpy as np

from facial_attendance.recognition.quality import (
    MEASURE_SIZE,
    FaceQuality,
    QualityLimits,
    check,
    measure,
    pose,
    prompt,
)

LIMITS = QualityLimits(
    min_sharpness=100.0,
    min_brightness=60.0,
    max_brightness=200.0,
    min_contrast=20.0,
    min_size=64,
    max_yaw=0.25,
    max_roll=15.0,
)

# YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner.
FRONTAL = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float32)


def checkerboard(size=128, cell=8):
    """A sharp, mid-grey-on-average RGB test card."""
    tiles = (np.indices((size, size)) // cell).sum(axis=0) % 2
    grey = np.where(tiles == 1, 200, 60).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def good_quality(**changes):
    values = dict(sharpness=500.0, brightness=128.0, contrast=60.0, size=128, yaw=0.0, roll=0.0)
    values.update(changes)
    return FaceQuality(**values)


class MeasureTests(unittest.TestCase):
    def test_blur_lowers_sharpness(self):
        card = checkerboard()
        blurred = cv2.GaussianBlur(card, (0, 0), 4)
        self.assertGreater(measure(card).sharpness, 10 * measure(blurred).sharpness)

    def test_brightness_and_contrast_follow_the_image(self):
        card = checkerboard()
        dark = (card * 0.2).astype(np.uint8)
        flat = np.full_like(card, 128)
        self.assertLess(measure(dark).brightness, measure(card).brightness)
        self.assertAlmostEqual(measure(flat).contrast, 0.0)
        self.assertAlmostEqual(measure(flat).brightness, 128.0)

    def test_measures_are_taken_at_the_model_input_size(self):
        # The same card at two resolutions gives the same values once resized.
        small = measure(checkerboard(size=MEASURE_SIZE, cell=4))
        large = measure(cv2.resize(checkerboard(size=MEASURE_SIZE, cell=4), (256, 256),
                                   interpolation=cv2.INTER_NEAREST))
        self.assertAlmostEqual(small.sharpness, large.sharpness, delta=0.05 * small.sharpness)
        self.assertEqual(small.size, MEASURE_SIZE)
        self.assertEqual(large.size, 256)

    def test_size_is_the_shorter_side(self):
        self.assertEqual(measure(checkerboard()[:, :90]).size, 90)

    def test_grey_crops_are_accepted(self):
        self.assertEqual(measure(checkerboard()[:, :, 0]).size, 128)

    def test_without_landmarks_pose_is_unknown(self):
        quality = measure(checkerboard())
        self.assertIsNone(quality.yaw)
        self.assertIsNone(quality.roll)

    def test_invalid_crops_are_refused(self):
        for crop in (np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10, 3), np.float32),
                     np.zeros((10, 10, 4), np.uint8)):
            with self.subTest(shape=crop.shape, dtype=crop.dtype), self.assertRaises(ValueError):
                measure(crop)

    def test_measure_size_matches_the_model_input(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this environment")
        from facial_attendance.recognition.model import INPUT_SIZE
        self.assertEqual(MEASURE_SIZE, INPUT_SIZE)


class PoseTests(unittest.TestCase):
    def test_frontal_face_is_level_and_centred(self):
        yaw, roll = pose(FRONTAL)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(roll, 0.0)

    def test_nose_offset_gives_signed_yaw(self):
        turned = FRONTAL.copy()
        turned[2, 0] += 16  # nose moves 0.4 inter-eye distances towards the left eye
        yaw, _ = pose(turned)
        self.assertAlmostEqual(yaw, 0.4, places=5)
        turned[2, 0] -= 32
        self.assertAlmostEqual(pose(turned)[0], -0.4, places=5)

    def test_rotated_eye_line_gives_roll_in_degrees(self):
        angle = np.radians(20)
        centre = FRONTAL[:2].mean(axis=0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (FRONTAL - centre) @ rotation.T + centre
        yaw, roll = pose(rotated)
        self.assertAlmostEqual(roll, 20.0, places=4)
        self.assertAlmostEqual(yaw, 0.0, places=5)

    def test_swapped_eye_order_still_reads_as_level(self):
        swapped = FRONTAL[[1, 0, 2, 4, 3]]
        self.assertAlmostEqual(pose(swapped)[1], 0.0)

    def test_measure_uses_landmarks_when_given(self):
        quality = measure(checkerboard(), FRONTAL)
        self.assertAlmostEqual(quality.yaw, 0.0)
        self.assertAlmostEqual(quality.roll, 0.0)

    def test_invalid_landmarks_are_refused(self):
        coincident = np.tile(FRONTAL[:1], (5, 1))
        for landmarks in (FRONTAL[:4], coincident, np.full((5, 2), np.nan)):
            with self.subTest(landmarks=landmarks.tolist()), self.assertRaises(ValueError):
                pose(landmarks)


class CheckTests(unittest.TestCase):
    def test_good_frame_passes(self):
        self.assertEqual(check(good_quality(), LIMITS), ())

    def test_each_limit_gives_its_reason(self):
        cases = {
            "blur": good_quality(sharpness=50.0),
            "dark": good_quality(brightness=40.0),
            "bright": good_quality(brightness=230.0),
            "low_contrast": good_quality(contrast=5.0),
            "small": good_quality(size=40),
            "turned": good_quality(yaw=-0.3),
            "tilted": good_quality(roll=-20.0),
        }
        for reason, quality in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(check(quality, LIMITS), (reason,))

    def test_values_at_the_limit_pass(self):
        quality = good_quality(sharpness=100.0, brightness=60.0, contrast=20.0, size=64, yaw=0.25, roll=15.0)
        self.assertEqual(check(quality, LIMITS), ())

    def test_unknown_pose_is_not_checked(self):
        self.assertEqual(check(good_quality(yaw=None, roll=None), LIMITS), ())

    def test_reasons_come_in_prompt_priority(self):
        quality = good_quality(sharpness=10.0, brightness=20.0, size=30, yaw=0.9)
        self.assertEqual(check(quality, LIMITS), ("small", "turned", "blur", "dark"))
        self.assertEqual(prompt(check(quality, LIMITS)), "Move closer to the camera")

    def test_prompt_is_none_for_a_good_frame(self):
        self.assertIsNone(prompt(()))
        self.assertEqual(prompt(("blur",)), "Hold still")


class LimitsTests(unittest.TestCase):
    MAPPING = {
        "min_sharpness": 100, "min_brightness": 60, "max_brightness": 200, "min_contrast": 20,
        "min_size": 64, "max_yaw": 0.25, "max_roll": 15,
    }

    def test_mapping_builds_limits(self):
        self.assertEqual(QualityLimits.from_mapping(self.MAPPING), LIMITS)

    def test_missing_extra_or_invalid_values_are_refused(self):
        bad = [
            {k: v for k, v in self.MAPPING.items() if k != "max_yaw"},
            {**self.MAPPING, "max_pitch": 10},
            {**self.MAPPING, "min_sharpness": "high"},
            {**self.MAPPING, "min_brightness": 210},
            {**self.MAPPING, "max_roll": float("nan")},
            {**self.MAPPING, "min_size": -1},
            {**self.MAPPING, "max_yaw": True},
        ]
        for mapping in bad:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                QualityLimits.from_mapping(mapping)


if __name__ == "__main__":
    unittest.main()
