"""Estimate coarse registration directions from YuNet's five face landmarks."""

from dataclasses import dataclass
from math import hypot, isfinite

from .emotion_presentation import geometry


@dataclass(frozen=True)
class FacePose:
    horizontal: float
    vertical: float

    def matches(self, direction, neutral):
        if direction is None:
            return abs(self.horizontal) <= .14 and .42 <= self.vertical <= .68
        if neutral is None:
            return False

        dx, dy = direction
        horizontal = self.horizontal - neutral.horizontal
        vertical = self.vertical - neutral.vertical
        if dx:
            # Source x is reversed on the mirrored preview; screen left is source right.
            return -dx * horizontal >= .12 and abs(vertical) <= .14
        return dy * vertical >= .08 and abs(horizontal) <= .16


def measure_pose(landmarks):
    if len(landmarks) != 5 or not all(isfinite(value) for point in landmarks for value in point):
        return None
    right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
    eye_dx, eye_dy = left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]
    eye_width = hypot(eye_dx, eye_dy)
    if eye_width < 1 or eye_dx <= 0 or abs(eye_dy) > eye_dx * .36:
        return None
    ux, uy = eye_dx / eye_width, eye_dy / eye_width
    eyes = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    mouth = ((right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2)
    mouth_dx, mouth_dy = mouth[0] - eyes[0], mouth[1] - eyes[1]
    nose_dx, nose_dy = nose[0] - eyes[0], nose[1] - eyes[1]
    face_height = -uy * mouth_dx + ux * mouth_dy
    if face_height < eye_width * .3:
        return None
    # Compare the nose with the eye–mouth centerline in a roll-adjusted basis.
    vertical = (-uy * nose_dx + ux * nose_dy) / face_height
    horizontal = (ux * (nose_dx - vertical * mouth_dx)
                  + uy * (nose_dy - vertical * mouth_dy)) / eye_width
    return FacePose(horizontal, vertical)


class RegistrationPoseEstimator:
    def __init__(self, detector_path):
        self.detector_path = detector_path
        self.detector = None

    def __call__(self, request):
        import cv2

        from ..liveness.face_detector import YuNetFaceDetector

        # This detector is confined to the existing reference worker.
        try:
            if self.detector is None:
                self.detector = YuNetFaceDetector(self.detector_path)
            faces = self.detector.detect(request.frame, with_landmarks=True)
        except (OSError, ValueError, cv2.error):
            return None
        if len(faces) != 1:
            return None
        (x, y, width, height), landmarks = faces[0]
        if geometry(request.bbox, (x, y, x + width, y + height))[0] < .70:
            return None
        return measure_pose(landmarks)
