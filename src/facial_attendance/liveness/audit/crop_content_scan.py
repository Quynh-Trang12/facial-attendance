"""Audit native source crops without treating a supplied face box as ground truth.

Size bands and detector disagreements are diagnostics for content review, not
acceptance thresholds or permission to replace labels/split assignments.
"""

import csv
import hashlib
import io
import json
import math
import threading

import cv2
import numpy as np
from PIL import Image

from facial_attendance.liveness.models.face_detector import YuNetFaceDetector

SPLIT_ANNOTATIONS = {
    "train": "reviewed_project_train.json",
    "validation": "project_validation.json",
    "test": "metas/intra_test/test_label.json",
}
SIZE_BANDS = (16, 32, 64, 112, 224)
_detector_state = threading.local()


def reviewed_crop_disposition(review):
    """Use source-content findings; detector success/count is not a veto."""
    verdict = review.get("decision", "unreviewed")
    confirmed = {
        "confirmed_wrong_roi",
        "confirmed_wrong_target",
        "confirmed_unusable_crop",
        "confirmed_valid_usable_roi",
    }
    if verdict in confirmed and not all(
        isinstance(review.get(field), str) and review[field].strip()
        for field in ("finding", "review_method", "evidence")
    ):
        raise ValueError("Confirmed crop decision requires review evidence")
    if verdict in {
        "confirmed_wrong_roi",
        "confirmed_wrong_target",
        "confirmed_unusable_crop",
    }:
        return "quarantine"
    if verdict == "confirmed_valid_usable_roi":
        return "retain"
    if verdict in {
        "unreviewed",
        "confirmed_valid_roi",
        "partial_roi_target_unresolved",
        "partial_roi_quality_review",
    }:
        return "review"
    raise ValueError("Unknown crop review verdict: " + str(verdict))


def supplied_crop(box_text, size):
    """Independently reconstruct the historical reproduction crop coordinates."""
    lines = box_text.splitlines()
    fields = lines[0].strip().split(" ") if lines else []
    try:
        if len(fields) != 5:
            raise ValueError("Unparseable supplied box")
        x, y, width, height = (int(float(value)) for value in fields[:4])
    except (ValueError, OverflowError):
        return (0, 0, *size), "full_image_fallback"
    image_width, image_height = size
    x, y = int(x * (image_width / 224)), int(y * (image_height / 224))
    width = int(width * (image_width / 224))
    height = int(height * (image_height / 224))
    left, top = max(0, x), max(0, y)
    right = image_width if left + width > image_width else x + width
    bottom = image_height if top + height > image_height else y + height
    if width <= 0 or height <= 0 or left >= right or top >= bottom:
        raise ValueError("Empty or invalid supplied crop")
    mode = (
        "clipped_box"
        if (left, top, right, bottom) != (x, y, x + width, y + height)
        else "supplied_box"
    )
    return (left, top, right, bottom), mode


def box_iou(left, right):
    x1, y1, x2, y2 = left
    a1, b1, a2, b2 = right
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union > 0 else 0.0


def size_band(edge):
    for limit in SIZE_BANDS:
        if edge < limit:
            return f"lt_{limit}"
    return "ge_224"


def safe_input(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the dataset")
    return path


def detector_boxes(image, detector_model):
    """Reuse the project's pinned CPU OpenCV locator for diagnostic proposals."""
    key = str(detector_model.resolve())
    if getattr(_detector_state, "key", None) != key:
        _detector_state.engine = YuNetFaceDetector(detector_model)
        _detector_state.key = key
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return [
        (x, y, x + width, y + height)
        for x, y, width, height in _detector_state.engine.detect(bgr)
    ]


def inspect_crop(root, record, *, detector_model=None):
    """Verify bytes, decode one source image and preserve all diagnostic values."""
    name = record["relative_path"]
    result = {
        "relative_path": name,
        "split": record["audit_split"],
        "binary_label": int(record["binary_label"]),
        "source_type": int(record["source_type"]),
        "image_width": None,
        "image_height": None,
        "crop_box": None,
        "crop_width": None,
        "crop_height": None,
        "native_min_edge": None,
        "size_band": None,
        "upscale_max": None,
        "crop_mode": None,
        "laplacian_variance": None,
        "detector_boxes": None,
        "best_detector_iou": None,
        "detector_max_area_ratio": None,
        "detector_candidates": None,
        "proposed_disposition": "detector_not_run",
        "review_flags": [],
        "error": None,
    }
    image_path = safe_input(root, name)
    box_path = image_path.with_name(image_path.stem + "_BB.txt")
    try:
        image_bytes, box_bytes = image_path.read_bytes(), box_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != record["sha256"]:
            raise ValueError("Source image checksum mismatch")
        if hashlib.sha256(box_bytes).hexdigest() != record["bbox_sha256"]:
            raise ValueError("Source bbox checksum mismatch")
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGB")
        box, mode = supplied_crop(box_bytes.decode("utf-8"), image.size)
        crop = image.crop(box)
        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        edge = min(crop.size)
        result.update(
            image_width=image.width,
            image_height=image.height,
            crop_box=list(box),
            crop_width=crop.width,
            crop_height=crop.height,
            native_min_edge=edge,
            size_band=size_band(edge),
            upscale_max=max(224 / crop.width, 224 / crop.height),
            crop_mode=mode,
            laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        )
        if mode != "supplied_box":
            result["review_flags"].append(mode)
        if detector_model is not None:
            proposals = detector_boxes(image, detector_model)
            overlap = max(
                (box_iou(box, proposal) for proposal in proposals), default=0.0
            )
            area = crop.width * crop.height
            area_ratio = max(
                ((b[2] - b[0]) * (b[3] - b[1]) / area for b in proposals), default=0.0
            )
            result.update(
                detector_boxes=[list(b) for b in proposals],
                best_detector_iou=overlap,
                detector_max_area_ratio=area_ratio,
                detector_candidates=len(proposals),
            )
            # Face counts describe locator behaviour, not sample validity.
            # A missed spoof face or an identifiable target among several faces
            # can still supply a valid training ROI after content review.
            result["proposed_disposition"] = (
                "review_multiple_faces"
                if len(proposals) > 1
                else "review_no_detection"
                if not proposals
                else "review_single_face_crop"
            )
            if not proposals:
                result["review_flags"].append("detector_no_face")
            elif overlap == 0:
                result["review_flags"].append("detector_no_overlap")
            if len(proposals) > 1:
                result["review_flags"].append("multiple_detector_faces")
            # These ratios prioritize review only. The largest detected face
            # can be irrelevant to a spoof presentation and never auto-wins.
            if any(
                (proposal[2] - proposal[0]) * (proposal[3] - proposal[1]) / area >= 4
                and box_iou(box, proposal) < 0.5
                for proposal in proposals
            ):
                result["review_flags"].append("larger_disagreeing_face")
        if not math.isfinite(result["laplacian_variance"]):
            raise ValueError("Nonfinite source quality statistic")
    except (OSError, ValueError, cv2.error) as failure:
        result["error"] = f"{type(failure).__name__}: {failure}"
    return result


def input_records(root, inventory):
    """Bind the exact original experimental memberships; never move a sample."""
    wanted, bindings = {}, {}
    for split, filename in SPLIT_ANNOTATIONS.items():
        path = root / filename
        content = path.read_bytes()
        mapping = json.loads(content)
        bindings[split] = hashlib.sha256(content).hexdigest()
        for name, target in mapping.items():
            if name in wanted:
                raise ValueError("A source image belongs to more than one split")
            wanted[name] = (split, target[43], target[40])
    records = []
    with inventory.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["relative_path"] not in wanted:
                continue
            split, label, source_type = wanted.pop(row["relative_path"])
            if (
                int(row["binary_label"]) != label
                or int(row["source_type"]) != source_type
            ):
                raise ValueError("Inventory and experimental annotations disagree")
            row["audit_split"] = split
            records.append(row)
    if wanted:
        raise ValueError("Experimental annotations are absent from the inventory")
    records.sort(
        key=lambda row: (
            list(SPLIT_ANNOTATIONS).index(row["audit_split"]),
            row["relative_path"],
        )
    )
    return records, bindings
