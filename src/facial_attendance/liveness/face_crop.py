"""Decode the supplied face crop consistently for auditing and loading."""

import math
from pathlib import Path

from PIL import Image


def bbox_path(path: Path) -> Path:
    return path.with_name(path.stem + "_BB.txt")


def read_face(path: Path, *, box_file: Path | None = None) -> Image.Image:
    """Apply the supplied single box in the author's 224-coordinate system.

    No detector is run here. Missing/invalid boxes stop acceptance; annotations
    do not independently prove that exactly one face is visible in the image.
    """
    try:
        values = (box_file or bbox_path(path)).read_text(encoding="utf-8").split()
        if len(values) != 5:
            raise ValueError("Expected x y width height confidence in _BB.txt")

        x, y, width, height, confidence = map(float, values)
        if not all(math.isfinite(v) for v in (x, y, width, height, confidence)):
            raise ValueError("Nonfinite bounding box")
        if width <= 0 or height <= 0 or not 0 < confidence <= 1:
            raise ValueError("Invalid bounding-box dimensions/confidence")

        with Image.open(path) as source:
            source.load()
            w, h = source.size

            x, y = int(x * w / 224), int(y * h / 224)
            width, height = int(width * w / 224), int(height * h / 224)
            box = (max(0, x), max(0, y), min(w, x + width), min(h, y + height))
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Empty or out-of-image bounding box")

            return source.convert("RGB").crop(box)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read liveness face {path}: {error}") from error
