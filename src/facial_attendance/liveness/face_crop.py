"""Decode the supplied face crop consistently for auditing and loading."""

import math
from pathlib import Path

from PIL import Image


def bbox_path(path: Path) -> Path:
    return path.with_name(path.stem + "_BB.txt")


def face_box(box_file: Path, size: tuple[int, int]) -> tuple[tuple[int, ...], float]:
    """Return the clipped source-pixel box and retained area fraction.

    Small boundary clipping follows the source protocol. A box losing over half
    its area is rejected rather than silently becoming a full-image crop. This
    geometric guard does not establish that the box contains a face.
    """
    values = box_file.read_text(encoding="utf-8").split()
    if len(values) != 5:
        raise ValueError("Expected x y width height confidence in _BB.txt")
    x, y, width, height, confidence = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, width, height, confidence)):
        raise ValueError("Nonfinite bounding box")
    if width <= 0 or height <= 0 or not 0 < confidence <= 1:
        raise ValueError("Invalid bounding-box dimensions/confidence")
    w, h = size
    x, y = int(x * w / 224), int(y * h / 224)
    width, height = int(width * w / 224), int(height * h / 224)
    box = (max(0, x), max(0, y), min(w, x + width), min(h, y + height))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Empty or out-of-image bounding box")
    retained = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    if retained < 0.5:
        raise ValueError("Bounding box loses over half its area outside image")
    return box, retained


def read_face(path: Path, *, box_file: Path | None = None) -> Image.Image:
    """Apply the supplied single box in the author's 224-coordinate system.

    No detector is run here. Missing/invalid boxes stop acceptance; annotations
    do not independently prove that exactly one face is visible in the image.
    """
    try:
        with Image.open(path) as source:
            source.load()
            box, _ = face_box(box_file or bbox_path(path), source.size)
            return source.convert("RGB").crop(box)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read liveness face {path}: {error}") from error
