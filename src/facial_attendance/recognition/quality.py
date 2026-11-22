"""Cheap face crop quality measures for the check-in quality gate.

A blurred, badly lit, small or turned face gives a noisy embedding, which can
reject a registered person at the deployed threshold. These measures flag such
frames before matching, so the user can be told what to fix.

Sharpness, brightness and contrast are measured after resizing the crop to the
model input size, so a value means the same for a webcam crop and a dataset
image of a different resolution. Size is measured on the original crop. Head
pose needs the five YuNet landmarks; without them it is not checked.
"""

from dataclasses import dataclass, fields
from math import atan2, degrees, isfinite
from numbers import Integral, Real

import numpy as np

# Same as recognition.model.INPUT_SIZE; kept here so the gate does not import torch.
MEASURE_SIZE = 64

# Most actionable first: a prompt names only the first reason.
REASONS = ("small", "turned", "tilted", "blur", "dark", "bright", "low_contrast")
PROMPTS = {
    "small": "Move closer to the camera",
    "turned": "Face the camera",
    "tilted": "Keep your head level",
    "blur": "Hold still",
    "dark": "More light, please",
    "bright": "Too bright, move out of the glare",
    "low_contrast": "Improve the lighting",
}


@dataclass(frozen=True)
class FaceQuality:
    """Measured values; yaw and roll are None when no landmarks were given."""

    sharpness: float
    brightness: float
    contrast: float
    size: int
    yaw: float | None
    roll: float | None


@dataclass(frozen=True)
class QualityLimits:
    """Pass/fail limits; a value exactly at a limit passes."""

    min_sharpness: float
    min_brightness: float
    max_brightness: float
    min_contrast: float
    min_size: int
    max_yaw: float
    max_roll: float

    @classmethod
    def from_mapping(cls, mapping) -> "QualityLimits":
        """Build limits from a config table, refusing missing, extra or invalid values."""
        names = [field.name for field in fields(cls)]
        if set(mapping) != set(names):
            raise ValueError(f"Expected quality limit keys: {', '.join(names)}")
        values = {}
        for name in names:
            value = mapping[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"Quality limit {name!r} must be a finite number of at least 0")
            values[name] = float(value)
        if not isinstance(mapping["min_size"], Integral):
            raise ValueError("Quality limit 'min_size' must be a whole number of pixels")
        values["min_size"] = int(mapping["min_size"])
        if values["min_brightness"] > values["max_brightness"]:
            raise ValueError("min_brightness must not exceed max_brightness")
        return cls(**values)


def pose(landmarks) -> tuple[float, float]:
    """Return (yaw, roll) from YuNet's five landmarks.

    Yaw is the signed offset of the nose tip from the eye midpoint, along the eye
    line, in inter-eye distances: 0 for a frontal face, growing as the head turns.
    Roll is the angle of the eye line in degrees, folded into [-90, 90] so the
    order of the two eyes does not matter.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Landmarks must be five finite (x, y) points")
    right_eye, left_eye, nose = points[0], points[1], points[2]
    eye_line = left_eye - right_eye
    distance = float(np.linalg.norm(eye_line))
    if distance == 0:
        raise ValueError("The two eye landmarks coincide")
    direction = eye_line / distance
    yaw = float((nose - (right_eye + left_eye) / 2) @ direction) / distance
    roll = degrees(atan2(eye_line[1], eye_line[0]))
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180
    return yaw, roll


def measure(crop, landmarks=None) -> FaceQuality:
    """Measure one face crop: an RGB (H, W, 3) or grey (H, W) uint8 array."""
    # Imported here so the limits and fusion code work where OpenCV is not installed.
    import cv2

    crop = np.asarray(crop)
    if crop.dtype != np.uint8 or crop.ndim not in (2, 3) or (crop.ndim == 3 and crop.shape[2] != 3):
        raise ValueError("A face crop must be a uint8 RGB or grey image")
    if min(crop.shape[:2]) == 0:
        raise ValueError("A face crop must not be empty")
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    small = cv2.resize(grey, (MEASURE_SIZE, MEASURE_SIZE), interpolation=cv2.INTER_AREA)
    yaw, roll = pose(landmarks) if landmarks is not None else (None, None)
    return FaceQuality(
        # Variance of the Laplacian: few sharp edges, as in a blurred face, give a low value.
        sharpness=float(cv2.Laplacian(small, cv2.CV_64F).var()),
        brightness=float(small.mean()),
        contrast=float(small.std()),
        size=int(min(crop.shape[:2])),
        yaw=yaw,
        roll=roll,
    )


def check(quality: FaceQuality, limits: QualityLimits) -> tuple[str, ...]:
    """Return the failed checks in prompt priority; an empty tuple means the frame passes."""
    failed = {
        "small": quality.size < limits.min_size,
        "turned": quality.yaw is not None and abs(quality.yaw) > limits.max_yaw,
        "tilted": quality.roll is not None and abs(quality.roll) > limits.max_roll,
        "blur": quality.sharpness < limits.min_sharpness,
        "dark": quality.brightness < limits.min_brightness,
        "bright": quality.brightness > limits.max_brightness,
        "low_contrast": quality.contrast < limits.min_contrast,
    }
    return tuple(reason for reason in REASONS if failed[reason])


def prompt(reasons) -> str | None:
    """The message for the most actionable reason, or None for a good frame."""
    return PROMPTS[reasons[0]] if reasons else None
