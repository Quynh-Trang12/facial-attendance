"""Find face regions on a frozen SYS-05 source frame."""

from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from .webcam_prototype import Box, box_in_frame, frame_size

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray


class FaceDetectorUnavailable(RuntimeError):
    """The OpenCV wheel cannot load the bundled Haar detector."""


def _box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def _is_redundant(candidate: Box, kept: Box) -> bool:
    x1, y1, x2, y2 = candidate
    kx1, ky1, kx2, ky2 = kept
    overlap = max(0, min(x2, kx2) - max(x1, kx1)) * max(0, min(y2, ky2) - max(y1, ky1))
    area = _box_area(candidate)
    kept_area = _box_area(kept)
    # A small, fully enclosed feature can be a Haar false positive. For
    # similarly sized windows, require strong overlap before dropping either.
    nested_feature = overlap == area and area <= 0.5 * kept_area
    near_duplicate = overlap / (area + kept_area - overlap) >= 0.8
    return nested_feature or near_duplicate


def _suppress_redundant_boxes(boxes: list[Box]) -> list[Box]:
    kept: list[Box] = []
    for candidate in sorted(boxes, key=lambda box: (-_box_area(box), box)):
        if not any(_is_redundant(candidate, box) for box in kept):
            kept.append(candidate)
    return kept


class HaarFaceDetector:
    """Load Haar once and return validated, filtered source coordinates."""

    def __init__(self):
        self._classifier = None

    def __call__(self, frame: "NDArray[uint8]") -> list[Box]:
        import cv2

        width, height = frame_size(frame)
        if self._classifier is None:
            factory = getattr(cv2, "CascadeClassifier", None)
            data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if factory is None or data_dir is None:
                raise FaceDetectorUnavailable("This OpenCV wheel lacks the bundled Haar detector.")
            cascade = Path(data_dir) / "haarcascade_frontalface_default.xml"
            if not cascade.is_file():
                raise FaceDetectorUnavailable("The bundled Haar cascade XML is missing.")
            try:
                classifier = factory(str(cascade))
            except cv2.error as error:
                raise FaceDetectorUnavailable("OpenCV could not load the bundled Haar cascade.") from error
            if classifier.empty():
                raise FaceDetectorUnavailable("OpenCV could not load the bundled Haar cascade.")
            self._classifier = classifier

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        boxes = []
        for rect in detections:
            # Check the raw types before integer conversion and do arithmetic in
            # Python ints to avoid overflowing a NumPy scalar.
            try:
                x, y, w, h = rect
            except (TypeError, ValueError) as error:
                raise ValueError("Haar returned an invalid face rectangle.") from error
            if not all(
                isinstance(value, Integral) and not isinstance(value, bool) for value in (x, y, w, h)
            ):
                raise ValueError("Haar returned an invalid face rectangle.")
            x, y, w, h = map(int, (x, y, w, h))
            box = (x, y, x + w, y + h)
            if not box_in_frame(box, width, height):
                raise ValueError("Haar returned an empty or out-of-frame face rectangle.")
            boxes.append(box)
        return _suppress_redundant_boxes(boxes)
