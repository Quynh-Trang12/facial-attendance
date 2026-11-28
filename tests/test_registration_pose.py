"""Registration pose acceptance and source-frame landmark coordinates."""

from math import cos, radians, sin
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from facial_attendance.liveness.face_detector import YuNetFaceDetector
from facial_attendance.ui.registration_pose import RegistrationPoseEstimator, measure_pose
from facial_attendance.ui.trial_profiles import TrialProfiles


def landmark_fixture(*, yaw=0, pitch=0, roll=0):
    """Project an approximate 3D face into an unmirrored camera frame."""
    yaw, pitch, roll = map(radians, (yaw, pitch, roll))
    result = []
    for x, y, z in ((-32, 0, 0), (32, 0, 0), (0, 32, 29), (-23, 59, 0), (23, 59, 0)):
        x, z = x * cos(yaw) + z * sin(yaw), z * cos(yaw) - x * sin(yaw)
        y = y * cos(pitch) + z * sin(pitch)
        result.append((120 + 2 * (x * cos(roll) - y * sin(roll)),
                       90 + 2 * (x * sin(roll) + y * cos(roll))))
    return tuple(result)


class RegistrationPoseTests(unittest.TestCase):
    def test_all_five_poses_and_reversed_or_wrong_axes_in_mirrored_preview(self):
        directions = [pose.direction for pose in TrialProfiles.POSES]
        neutral = measure_pose(landmark_fixture())
        poses = ({}, {"yaw": 23}, {"yaw": -23}, {"pitch": -17}, {"pitch": 17})
        for actual, args in enumerate(poses):
            measurement = measure_pose(landmark_fixture(**args))
            expected_steps = (True, False, False, False, False) if actual == 0 else tuple(
                index == actual for index in range(5)
            )
            with self.subTest(actual=TrialProfiles.POSES[actual].instruction):
                self.assertEqual(tuple(measurement.matches(direction, neutral) for direction in directions),
                                 expected_steps)

    def test_compensates_modest_roll_and_rejects_unreliable_landmarks(self):
        frontal = measure_pose(landmark_fixture(roll=14))
        self.assertTrue(frontal.matches(None, None))
        self.assertIsNone(measure_pose(landmark_fixture(roll=29)))
        self.assertIsNone(measure_pose(((1, 1),) * 5))
        self.assertIsNone(measure_pose(((float("nan"), 0),) * 5))

    def test_yunet_reports_landmarks_in_original_coordinates_only_when_requested(self):
        detector = object.__new__(YuNetFaceDetector)
        detector.engine = Mock()
        coordinates = [*range(20, 30)]
        detector.engine.detect.return_value = (1, [[12, 18, 40, 50, *coordinates, .99]])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [(24, 36, 80, 100)])
        self.assertEqual(detector.detect(frame, with_landmarks=True),
                         [((24, 36, 80, 100), ((40, 42), (44, 46), (48, 50), (52, 54), (56, 58)))])
        self.assertEqual(detector.engine.detect.call_args.args[0].shape, (360, 640, 3))

    def test_requires_one_matching_face_on_the_exact_selected_frame(self):
        estimator = RegistrationPoseEstimator("not-loaded.onnx")
        face = (20, 20, 100, 120), landmark_fixture()
        estimator.detector = Mock()
        request = SimpleNamespace(frame=object(), bbox=(20, 20, 120, 140))
        estimator.detector.detect.return_value = [face]
        self.assertEqual(estimator(request), measure_pose(face[1]))
        estimator.detector.detect.assert_called_once_with(request.frame, with_landmarks=True)
        for faces in ([], [face, face], [((150, 20, 100, 120), face[1])]):
            estimator.detector.detect.return_value = faces
            self.assertIsNone(estimator(request))
