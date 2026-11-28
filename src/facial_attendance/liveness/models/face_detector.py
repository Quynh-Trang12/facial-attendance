"""Supported OpenCV-5 face localization, separate from the trained PAD model."""

import math
from pathlib import Path

import cv2

from facial_attendance.liveness.dataset_files import file_sha256, source_root

YUNET_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URL = f"https://huggingface.co/opencv/face_detection_yunet/resolve/{YUNET_REVISION}/{YUNET_FILENAME}"
DEFAULT_MODEL = source_root().parents[2] / "artifacts/detectors" / YUNET_FILENAME
DETECTION_MAX_SIDE = 640


class YuNetFaceDetector:
    """Use the publisher's 0.9 score/0.3 NMS defaults; no physical-test tuning."""

    def __init__(self, model=None):
        self.path = Path(model or DEFAULT_MODEL).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                "Prepare the auxiliary detector with scripts/preparation/prepare_liveness_detector.py"
            )
        if file_sha256(self.path) != YUNET_SHA256:
            raise ValueError("Auxiliary face detector differs from the pinned model")
        self.engine = cv2.FaceDetectorYN_create(
            str(self.path), "", (320, 320), 0.9, 0.3, 5000
        )

    def detect(self, frame):
        height, width = frame.shape[:2]
        # Full-HD inputs can put large faces outside YuNet's useful scale.
        # Localize on a bounded image, but return source coordinates so PAD
        # still receives pixels cropped from the original frame.
        scale = min(1.0, DETECTION_MAX_SIDE / max(width, height))
        detection_width = max(1, round(width * scale))
        detection_height = max(1, round(height * scale))
        detection_frame = (
            frame
            if scale == 1.0
            else cv2.resize(
                frame, (detection_width, detection_height), interpolation=cv2.INTER_AREA
            )
        )
        self.engine.setInputSize((detection_width, detection_height))
        _, faces = self.engine.detect(detection_frame)
        scale_x, scale_y = width / detection_width, height / detection_height
        boxes = []
        for face in faces if faces is not None else ():
            x, y, w, h = (float(v) for v in face[:4])
            left, top = max(0, math.floor(x * scale_x)), max(0, math.floor(y * scale_y))
            right = min(width, math.ceil((x + w) * scale_x))
            bottom = min(height, math.ceil((y + h) * scale_y))
            if right > left and bottom > top:
                boxes.append((left, top, right - left, bottom - top))
        return boxes
