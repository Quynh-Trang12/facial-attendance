"""Find face regions on an immutable SYS-05 source snapshot."""

from numbers import Integral, Real
from math import ceil, floor, isfinite
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from .webcam_prototype import Box, box_in_frame, frame_size

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray


class FaceDetectorUnavailable(RuntimeError):
    """A configured detector's API, model or classifier is unavailable."""


YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL = Path("artifacts/sys05-yunet") / YUNET_FILENAME


def bounded_size(width, height, max_edge=640):
    """Preserve aspect ratio without enlarging; return actual integer dimensions."""
    scale = min(1.0, max_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def map_box(box, source_size, detected_size):
    """Outward rounding preserves coverage at non-integer resize ratios."""
    width, height = source_size
    small_width, small_height = detected_size
    if not box_in_frame(box, small_width, small_height):
        raise ValueError("Invalid detector region before coordinate conversion.")
    x1, y1, x2, y2 = map(int, box)
    mapped = (x1 * width // small_width, y1 * height // small_height,
              (x2 * width + small_width - 1) // small_width,
              (y2 * height + small_height - 1) // small_height)
    if not box_in_frame(mapped, width, height):
        raise ValueError("Invalid source region after coordinate conversion.")
    return mapped


class ScaledFaceDetector:
    """Bound detector input cost; preserve the original source for the adapters."""

    def __init__(self, detector, max_edge=640):
        if type(max_edge) is not int or max_edge < 1:
            raise ValueError("Detector max edge must be a positive integer.")
        self.detector = detector
        self.max_edge = max_edge

    def __call__(self, frame):
        import cv2
        source_size = frame_size(frame)
        small_size = bounded_size(*source_size, self.max_edge)
        small = (frame if source_size == small_size else
                 cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA))
        return [map_box(box, source_size, small_size) for box in self.detector(small)]


class YuNetFaceDetector:
    """OpenCV 4.x CPU YuNet; local hash-pinned model only, never downloads.

    Wrap with ScaledFaceDetector to bound input size. Glasses robustness must
    be physically evaluated; choosing a DNN is not an accuracy claim.
    """

    def __init__(self, model_path=YUNET_MODEL):
        self.model_path = Path(model_path)
        self._model = None

    def __call__(self, frame):
        import cv2
        width, height = frame_size(frame)
        if self._model is None:
            factory = getattr(getattr(cv2, "FaceDetectorYN", None), "create", None)
            if factory is None or not str(cv2.__version__).startswith("4."):
                raise FaceDetectorUnavailable("YuNet needs the isolated OpenCV 4.13 CPU API.")
            try:
                digest = sha256(self.model_path.read_bytes()).hexdigest()
            except OSError as error:
                raise FaceDetectorUnavailable(
                    "YuNet model missing. Run scripts/setup_sys05_yunet.py --download explicitly."
                ) from error
            if digest != YUNET_SHA256:
                raise FaceDetectorUnavailable("YuNet model checksum mismatch; refusing this file.")
            try:
                self._model = factory(str(self.model_path), "", (width, height), 0.9, 0.3, 5000,
                                      cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            except cv2.error as error:
                raise FaceDetectorUnavailable("OpenCV could not load the pinned YuNet model.") from error
        self._model.setInputSize((width, height))
        _, faces = self._model.detect(frame)
        if faces is None:
            return []
        boxes = []
        for face in faces:
            if len(face) != 15 or not all(
                isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))
                for value in face
            ):
                raise ValueError("YuNet returned invalid detection values.")
            x, y, w, h = map(float, face[:4])
            score = float(face[14])
            if w <= 0 or h <= 0 or not 0.9 <= score <= 1:
                raise ValueError("YuNet returned invalid geometry or confidence.")
            # Neural boxes can legitimately cross an image edge. Clip only
            # intersecting, finite, positive rectangles; reject the whole batch
            # if any region is empty/fully outside. Never invent a region.
            box = max(0, floor(x)), max(0, floor(y)), min(width, ceil(x + w)), min(height, ceil(y + h))
            if not box_in_frame(box, width, height):
                raise ValueError("YuNet returned an out-of-frame face region.")
            boxes.append(box)
        return boxes


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
