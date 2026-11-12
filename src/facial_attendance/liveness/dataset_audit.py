"""Audit every source image and combine integrity, split and label findings."""

from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath

from PIL import Image

from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import bbox_path, read_face
from facial_attendance.liveness.image_inventory import build_inventory
from facial_attendance.liveness.data_contract import (
    FAMILY_NAMES,
    SPLITS,
    LivenessSample,
)
from facial_attendance.liveness.split_rules import check_split_integrity


def _audit_image(root: Path, sample: LivenessSample):
    digest = box_digest = ""
    path = None
    error = None

    try:
        path = local_path(root, sample.relative_path)
        digest = file_sha256(path)
        box = local_path(root, str(bbox_path(PurePosixPath(sample.relative_path))))
        box_digest = file_sha256(box)
        read_face(path, box_file=box)
    except (OSError, ValueError) as failure:
        error = {
            "relative_path": sample.relative_path,
            "split": sample.split,
            "source_type": sample.source_type,
            "error": str(failure),
            "raw_image_decode": "UNVERIFIED",
        }

        if path is not None:
            # A bad box must not prevent checking the original image's pixels.
            # This is integrity checking, never a full-image training fallback.
            try:
                with Image.open(path) as original:
                    original.load()
                error["raw_image_decode"] = "PASS"
            except (OSError, ValueError) as decode_error:
                error["raw_image_decode"] = "FAIL"
                error["raw_image_decode_error"] = str(decode_error)

    return replace(sample, sha256=digest, bbox_sha256=box_digest), error


def _audit_images(root: Path, records, workers: int):
    if workers == 1:
        for sample in records:
            yield _audit_image(root, sample)
        return

    # Two pending jobs per worker overlap file I/O without creating half a
    # million Futures. Consume in input order to keep inventory hashes stable.
    source = iter(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(2 * workers):
            sample = next(source, None)
            if sample is not None:
                pending.append(pool.submit(_audit_image, root, sample))

        while pending:
            result = pending.popleft().result()
            sample = next(source, None)
            if sample is not None:
                pending.append(pool.submit(_audit_image, root, sample))
            yield result


def audit_dataset(
    root: Path,
    data_description_file: Path,
    *,
    progress: Callable | None = None,
    workers: int = 1,
) -> tuple[tuple[LivenessSample, ...], dict]:
    """Check all annotated images without changing selection or split assignments."""
    root = root.resolve()
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")

    inventory = build_inventory(root, data_description_file)
    data_description = inventory.source
    policy = data_description.get("validation_from_train")
    relation_name = data_description.get("relationships")
    duplicate_subjects = (
        inventory.assignments.duplicate_train_subjects
        if inventory.assignments
        else set()
    )

    errors = []
    checked = []
    for sample, error in _audit_images(root, inventory.records, workers):
        checked.append(sample)
        if error is not None:
            errors.append(error)

        if progress is not None and len(checked) % 10000 == 0:
            progress(len(checked), sample.split, len(errors))

    records = checked

    if duplicate_subjects:
        evaluation_hashes = {
            row.sha256
            for row in records
            if row.split in {"validation", "test"} and row.sha256
        }
        subjects_with_duplicate_images = {
            row.subject_id
            for row in records
            if row.split == "quarantine"
            and row.subject_id in duplicate_subjects
            and row.sha256 in evaluation_hashes
        }
        if subjects_with_duplicate_images != duplicate_subjects:
            raise ValueError(
                f"No identical-image evidence for quarantined training subjects: {sorted(duplicate_subjects - subjects_with_duplicate_images)}"
            )

    report = audit_splits(records)
    selected_paths = {row.relative_path for row in records if not row.exclusion_reason}
    blocking_errors = [
        error for error in errors if error["relative_path"] in selected_paths
    ]
    report.update(
        {
            "source": data_description,
            "data_description_sha256": file_sha256(data_description_file),
            "annotations": inventory.annotations,
            "image_errors": errors,
            "blocking_image_errors": blocking_errors,
            "excluded_image_errors": [
                error
                for error in errors
                if error["relative_path"] not in selected_paths
            ],
            "image_integrity": {
                "raw_images_decoded": len(records)
                - sum(e.get("raw_image_decode") != "PASS" for e in errors),
                "valid_face_inputs": len(records) - len(errors),
                "invalid_face_inputs": len(errors),
            },
            "decode_workers": workers,
            "folder_label_disagreements": inventory.folder_label_disagreements,
            "label_authority": "Rich JSON attributes 40/43; folder names do not override labels",
        }
    )

    if policy:
        report["validation_from_train"] = {
            "method": policy["method"],
            "validation_subject_file_sha256": file_sha256(
                local_path(root, policy["validation_subject_file"])
            ),
            "validation_subjects": sorted(inventory.assignments.validation_subjects),
            "quarantined_train_subjects": sorted(
                inventory.assignments.quarantined_train_subjects
            ),
            "original_annotation_sha256": {
                split: file_sha256(local_path(root, path))
                for split, path in policy["source_annotations"].items()
            },
        }
    if relation_name:
        report["relationships_sha256"] = file_sha256(local_path(root, relation_name))
    if data_description.get("quality_findings"):
        report["quality_findings_sha256"] = file_sha256(
            local_path(root, data_description["quality_findings"])
        )

    report["accepted"] = (
        not blocking_errors
        and not report["overlaps"]
        and not report["missing_families"]
    )

    return tuple(records), report


def audit_splits(records) -> dict:
    """Check all supplied rows, including excluded attacks, before filtering."""
    records = tuple(records)
    checks = check_split_integrity(records)
    selected = [row for row in records if not row.exclusion_reason]
    counts = checks.family_counts

    return {
        "overlaps": checks.overlaps,
        "missing_families": checks.missing_families,
        "inventory_rows": len(records),
        "quarantined_rows": sum(row.split == "quarantine" for row in records),
        "counts": {
            split: {
                "selected": sum(row.split == split for row in selected),
                "excluded": sum(
                    row.split == split and bool(row.exclusion_reason) for row in records
                ),
                "subjects": len(
                    {row.subject_id for row in selected if row.split == split}
                ),
                "families": {family: counts[split, family] for family in FAMILY_NAMES},
                "source_types": dict(
                    Counter(
                        str(row.source_type) for row in records if row.split == split
                    )
                ),
            }
            for split in SPLITS
        },
        "source_media_known": sum(bool(row.source_media_id) for row in records),
        "source_media_unknown": sum(not row.source_media_id for row in records),
        "known_common_media_groups": sum(
            count > 1
            for count in Counter(
                row.source_media_id for row in records if row.source_media_id
            ).values()
        ),
        "session_known": sum(bool(row.session_id) for row in records),
        "limitations": [
            "SHA-256 detects identical files, not recompressed or near-duplicate images.",
            "Media/session overlap checks cover supplied IDs only; blanks mean unknown.",
            "Annotations and boxes do not independently verify visual labels or face count.",
        ],
    }
