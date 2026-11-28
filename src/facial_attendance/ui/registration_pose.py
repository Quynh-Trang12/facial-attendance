"""Estimate coarse registration directions from YuNet's five face landmarks."""

from dataclasses import dataclass
from math import hypot, isfinite

from .emotion_presentation import geometry


@dataclass(frozen=True)
class FacePose:
    horizontal: float
    vertical: float

    def matches(self, direction, neutral):
        if not isfinite(self.horizontal) or not isfinite(self.vertical):
            return False
        if direction is None:
            return abs(self.horizontal) <= 0.14 and 0.42 <= self.vertical <= 0.68
        if neutral is None:
            return False
        if not isfinite(neutral.horizontal) or not isfinite(neutral.vertical):
            return False

        dx, dy = direction
        horizontal = self.horizontal - neutral.horizontal
        vertical = self.vertical - neutral.vertical
        if dx:
            # Source x is reversed on the mirrored preview; screen left is source right.
            return -dx * horizontal >= 0.12 and abs(vertical) <= 0.14
        return dy * vertical >= 0.08 and abs(horizontal) <= 0.16


def measure_pose(landmarks):
    try:
        if len(landmarks) != 5 or any(len(point) != 2 for point in landmarks):
            return None
        landmarks = tuple((float(x), float(y)) for x, y in landmarks)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for point in landmarks for value in point):
        return None
    right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
    eye_dx, eye_dy = left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]
    eye_width = hypot(eye_dx, eye_dy)
    if eye_width < 1 or eye_dx <= 0 or abs(eye_dy) > eye_dx * 0.36:
        return None
    ux, uy = eye_dx / eye_width, eye_dy / eye_width
    eyes = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    mouth = ((right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2)
    mouth_dx, mouth_dy = mouth[0] - eyes[0], mouth[1] - eyes[1]
    nose_dx, nose_dy = nose[0] - eyes[0], nose[1] - eyes[1]
    face_height = -uy * mouth_dx + ux * mouth_dy
    if face_height < eye_width * 0.3:
        return None
    # Compare the nose with the eye–mouth centerline in a roll-adjusted basis.
    vertical = (-uy * nose_dx + ux * nose_dy) / face_height
    horizontal = (
        ux * (nose_dx - vertical * mouth_dx) + uy * (nose_dy - vertical * mouth_dy)
    ) / eye_width
    if not isfinite(horizontal) or not isfinite(vertical):
        return None
    return FacePose(horizontal, vertical)


class RegistrationPoseEstimator:
    def __init__(self, detector_path):
        self.detector_path = detector_path
        self.detector = None

    def __call__(self, request):
        import cv2

        from ..liveness.models.face_detector import YuNetFaceDetector

        # Load OpenCV and this worker-owned detector only when checking a
        # reference; neither Tk startup nor CLI help needs the pose runtime.
        try:
            if self.detector is None:
                self.detector = YuNetFaceDetector(self.detector_path)
            faces = self.detector.detect(request.frame, with_landmarks=True)
        except (OSError, ValueError, cv2.error):
            return None
        if len(faces) != 1:
            return None
        (x, y, width, height), landmarks = faces[0]
        if geometry(request.bbox, (x, y, x + width, y + height))[0] < 0.70:
            return None
        return measure_pose(landmarks)
