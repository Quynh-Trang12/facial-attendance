"""Registration pose acceptance and source-frame landmark coordinates."""

import unittest
from math import cos, radians, sin
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from facial_attendance.liveness.models.face_detector import YuNetFaceDetector
from facial_attendance.ui.registration_pose import (
    FacePose,
    RegistrationPoseEstimator,
    measure_pose,
)
from facial_attendance.ui.trial_profiles import TrialProfiles


def landmark_fixture(*, yaw=0, pitch=0, roll=0):
    """Project an approximate 3D face into an unmirrored camera frame."""
    yaw, pitch, roll = map(radians, (yaw, pitch, roll))
    result = []
    for x, y, z in ((-32, 0, 0), (32, 0, 0), (0, 32, 29), (-23, 59, 0), (23, 59, 0)):
        x, z = x * cos(yaw) + z * sin(yaw), z * cos(yaw) - x * sin(yaw)
        y = y * cos(pitch) + z * sin(pitch)
        result.append(
            (
                120 + 2 * (x * cos(roll) - y * sin(roll)),
                90 + 2 * (x * sin(roll) + y * cos(roll)),
            )
        )
    return tuple(result)


class RegistrationPoseTests(unittest.TestCase):
    def test_all_five_poses_and_reversed_or_wrong_axes_in_mirrored_preview(self):
        directions = [pose.direction for pose in TrialProfiles.POSES]
        neutral = measure_pose(landmark_fixture())
        poses = ({}, {"yaw": 23}, {"yaw": -23}, {"pitch": -17}, {"pitch": 17})
        for actual, args in enumerate(poses):
            measurement = measure_pose(landmark_fixture(**args))
            expected_steps = (
                (True, False, False, False, False)
                if actual == 0
                else tuple(index == actual for index in range(5))
            )
            with self.subTest(actual=TrialProfiles.POSES[actual].instruction):
                self.assertEqual(
                    tuple(
                        measurement.matches(direction, neutral)
                        for direction in directions
                    ),
                    expected_steps,
                )

    def test_compensates_modest_roll_and_rejects_unreliable_landmarks(self):
        frontal = measure_pose(landmark_fixture(roll=14))
        self.assertTrue(frontal.matches(None, None))
        self.assertIsNone(measure_pose(landmark_fixture(roll=29)))
        self.assertIsNone(measure_pose(((1, 1),) * 5))
        self.assertIsNone(measure_pose(((float("nan"), 0),) * 5))

    def test_malformed_landmarks_and_nonfinite_measurements_cannot_match(self):
        for landmarks in (None, (), ((1,),) * 5, ((1, 2, 3),) * 5, (("x", 1),) * 5):
            with self.subTest(landmarks=landmarks):
                self.assertIsNone(measure_pose(landmarks))
        neutral = FacePose(0, 0.5)
        for invalid in (FacePose(float("inf"), 0.5), FacePose(0, float("nan"))):
            self.assertFalse(invalid.matches(None, None))
            self.assertFalse(invalid.matches((-1, 0), neutral))
            self.assertFalse(neutral.matches((0, 1), invalid))

    def test_yunet_reports_landmarks_in_original_coordinates_only_when_requested(self):
        detector = object.__new__(YuNetFaceDetector)
        detector.engine = Mock()
        coordinates = [*range(20, 30)]
        detector.engine.detect.return_value = (
            1,
            [[12, 18, 40, 50, *coordinates, 0.99]],
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [(24, 36, 80, 100)])
        self.assertEqual(
            detector.detect(frame, with_landmarks=True),
            [((24, 36, 80, 100), ((40, 42), (44, 46), (48, 50), (52, 54), (56, 58)))],
        )
        self.assertEqual(detector.engine.detect.call_args.args[0].shape, (360, 640, 3))

    def test_landmarks_follow_actual_rounded_resize_ratios_for_portrait_frames(self):
        detector = object.__new__(YuNetFaceDetector)
        detector.engine = Mock()
        coordinates = np.array(landmark_fixture()).reshape(-1).tolist()
        detector.engine.detect.return_value = (
            1,
            [[12, 18, 40, 50, *coordinates, 0.99]],
        )
        frame = np.zeros((1001, 751, 3), dtype=np.uint8)
        [(box, points)] = detector.detect(frame, with_landmarks=True)
        self.assertEqual([box], detector.detect(frame))
        expected = np.array(landmark_fixture()) * (751 / 480, 1001 / 640)
        np.testing.assert_allclose(points, expected)
        detector.engine.setInputSize.assert_called_with((480, 640))

    def test_invalid_landmarks_request_fails_without_changing_box_only_callers(self):
        detector = object.__new__(YuNetFaceDetector)
        detector.engine = Mock()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for row in ([12, 18, 40, 50], [12, 18, 40, 50, *([float("nan")] * 10), 0.99]):
            with self.subTest(row=row):
                detector.engine.detect.return_value = (1, [row])
                self.assertEqual(detector.detect(frame), [(12, 18, 40, 50)])
                with self.assertRaisesRegex(ValueError, "invalid face landmarks"):
                    detector.detect(frame, with_landmarks=True)

    def test_requires_one_matching_face_on_the_exact_selected_frame(self):
        estimator = RegistrationPoseEstimator("not-loaded.onnx")
        face = (20, 20, 100, 120), landmark_fixture()
        estimator.detector = Mock()
        request = SimpleNamespace(frame=object(), bbox=(20, 20, 120, 140))
        estimator.detector.detect.return_value = [face]
        self.assertEqual(estimator(request), measure_pose(face[1]))
        estimator.detector.detect.assert_called_once_with(
            request.frame, with_landmarks=True
        )
        for faces in ([], [face, face], [((150, 20, 100, 120), face[1])]):
            estimator.detector.detect.return_value = faces
            self.assertIsNone(estimator(request))
        estimator.detector.detect.return_value = [(face[0], None)]
        self.assertIsNone(estimator(request))

    def test_unavailable_detector_cannot_supply_pose_evidence(self):
        estimator = RegistrationPoseEstimator("not-loaded.onnx")
        estimator.detector = Mock()
        request = SimpleNamespace(frame=object(), bbox=(20, 20, 120, 140))
        for error in (OSError("missing"), ValueError("invalid landmarks")):
            estimator.detector.detect.side_effect = error
            self.assertIsNone(estimator(request))
