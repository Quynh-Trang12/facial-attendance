"""Build annotation inventories and validate/read/write their CSV records."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from facial_attendance.liveness.dataset_files import file_sha256, local_path, read_json
from facial_attendance.liveness.data_contract import (
    EXCLUDED,
    SOURCE,
    SPLITS,
    LivenessSample,
    label_family,
)
from facial_attendance.liveness.split_rules import (
    SubjectSplits,
    check_split_integrity,
    read_subject_splits,
)


@dataclass(frozen=True)
class Inventory:
    records: tuple[LivenessSample, ...]
    source: dict
    annotations: dict
    folder_label_disagreements: list[str]
    assignments: SubjectSplits | None


def read_relationships(path: Path | None) -> dict:
    """Use documented media/session identities; never derive them from filenames."""
    if path is None:
        return {}

    records = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", "source_media_id", "session_id"]:
            raise ValueError(
                "Relationships need relative_path,source_media_id,session_id"
            )

        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed relationship row")

            name = row["relative_path"].replace("\\", "/")
            if name in records:
                raise ValueError(f"Duplicate relationship row: {name}")
            records[name] = (row["source_media_id"].strip(), row["session_id"].strip())

    return records


def build_inventory(
    root: Path,
    data_description_file: Path,
) -> Inventory:
    """Read documented project splits and check original split assignments.

    The data description names each annotation file and records how project
    validation is derived from original training data. Subject IDs follow
    Data/<original-split>/<subject-id>/<live-or-spoof>/<image>.
    """
    root = root.resolve()
    data_description = read_json(data_description_file)
    if not isinstance(data_description, dict):
        raise ValueError("Data description must be a JSON object")
    for field in ("dataset_archive", "received_on", "split_evidence", "selection"):
        if (
            not isinstance(data_description.get(field), str)
            or not data_description[field].strip()
        ):
            raise ValueError(f"Data description requires {field}")
    if data_description.get("source_url") != SOURCE:
        raise ValueError("Expected the LIV-01 CelebA-Spoof source_url")

    split_annotations = data_description.get("split_annotations", {})
    if not isinstance(split_annotations, dict) or set(split_annotations) != set(SPLITS):
        raise ValueError(
            "Project train, validation and test annotation files are required; split assignments are never inferred"
        )
    if any(
        not isinstance(name, str) or not name for name in split_annotations.values()
    ):
        raise ValueError("Annotation filenames must be nonempty strings")

    relation_name = data_description.get("relationships")
    if relation_name is not None and (
        not isinstance(relation_name, str) or not relation_name
    ):
        raise ValueError("relationships must name a CSV file, or be omitted")
    relationships = read_relationships(
        local_path(root, relation_name) if relation_name else None
    )

    policy = data_description.get("validation_from_train")
    assignments = read_subject_splits(root, policy) if policy else None
    original_annotations = assignments.original_annotations if assignments else {}
    duplicate_subjects = (
        set(policy.get("duplicate_train_subjects", [])) if policy else set()
    )

    records, annotation_hashes = [], {}
    folder_disagreements = []
    seen = set()
    inventory_annotations = dict(split_annotations)
    if data_description.get("quarantine_annotations"):
        if not policy:
            raise ValueError(
                "Quarantine annotations require the original subject split policy"
            )
        inventory_annotations["quarantine"] = data_description["quarantine_annotations"]

    for split, annotation_name in inventory_annotations.items():
        annotation = local_path(root, annotation_name)
        annotation_hashes[split] = {
            "path": annotation_name,
            "sha256": file_sha256(annotation),
        }

        labels = read_json(annotation)
        if not isinstance(labels, dict) or not labels and split != "quarantine":
            raise ValueError(f"Empty or invalid annotations: {annotation}")

        for raw_name, attributes in sorted(labels.items()):
            name = raw_name.replace("\\", "/")
            parts = PurePosixPath(name).parts

            if (
                len(parts) != 5
                or parts[0] != "Data"
                or not parts[2]
                or parts[3] not in {"live", "spoof"}
            ):
                raise ValueError(f"Unrecognised original image path layout: {name}")
            if name in seen:
                raise ValueError(f"Repeated image across annotation entries: {name}")
            seen.add(name)

            if (
                not isinstance(attributes, list)
                or len(attributes) != 44
                or type(attributes[40]) is not int
                or type(attributes[43]) is not int
            ):
                raise ValueError(f"Expected 44 attributes with integer labels: {name}")
            source_type, binary = attributes[40], attributes[43]
            family = label_family(source_type, binary, name)
            if parts[3] != ("live" if binary == 0 else "spoof"):
                folder_disagreements.append(name)

            original_split = parts[1]
            if policy:
                expected = assignments.project_split(original_split, parts[2])
                if (
                    original_split not in original_annotations
                    or original_annotations[original_split].get(name) != attributes
                    or split != expected
                ):
                    raise ValueError(
                        f"Original annotation/subject split policy violated: {name}"
                    )

            media, session = relationships.get(name, ("", ""))
            records.append(
                LivenessSample(
                    name,
                    parts[2],
                    split,
                    source_type,
                    binary,
                    family,
                    "",
                    "",
                    media,
                    session,
                    (
                        "training_subject_has_duplicate_in_evaluation"
                        if split == "quarantine" and parts[2] in duplicate_subjects
                        else "original_train_subject_overlaps_test"
                        if split == "quarantine"
                        else "outside_LIV-01_selected_types"
                        if source_type in EXCLUDED
                        else ""
                    ),
                    original_split,
                )
            )

    if data_description.get("selection_scope") == "full_dataset":
        expected_paths = set().union(
            *(set(labels) for labels in original_annotations.values())
        )
        if seen != expected_paths:
            raise ValueError(
                f"Full-dataset inventory mismatch: {len(expected_paths - seen)} missing, {len(seen - expected_paths)} extra"
            )

    unknown = set(relationships) - seen
    if unknown:
        raise ValueError(
            f"Relationship paths absent from annotations: {sorted(unknown)[:5]}"
        )

    return Inventory(
        tuple(records),
        data_description,
        annotation_hashes,
        folder_disagreements,
        assignments,
    )


def write_inventory(path: Path, records) -> str:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(LivenessSample.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)

    return file_sha256(path)


def read_inventory(path: Path) -> tuple[LivenessSample, ...]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(LivenessSample.__dataclass_fields__):
            raise ValueError("Unexpected liveness inventory columns")

        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed inventory row")
            for key in ("source_type", "binary_label"):
                row[key] = int(row[key])

            sample = LivenessSample(**row)
            parts = PurePosixPath(sample.relative_path.replace("\\", "/")).parts
            family = label_family(
                sample.source_type, sample.binary_label, sample.relative_path
            )
            excluded = sample.source_type in EXCLUDED
            quarantined = sample.split == "quarantine"
            expected_exclusion = (
                sample.exclusion_reason
                if quarantined
                else "outside_LIV-01_selected_types"
                if excluded
                else ""
            )

            if (
                len(parts) != 5
                or parts[0] != "Data"
                or ".." in parts
                or ":" in sample.relative_path
                or parts[2] != sample.subject_id
                or parts[3] not in {"live", "spoof"}
                or sample.original_split != parts[1]
                or sample.original_split not in SPLITS
                or sample.original_split == "test"
                and sample.split != "test"
                or sample.original_split == "validation"
                and sample.split != "validation"
                or sample.original_split == "train"
                and sample.split not in {"train", "validation", "quarantine"}
                or quarantined
                and sample.original_split != "train"
                or quarantined
                and sample.exclusion_reason
                not in {
                    "original_train_subject_overlaps_test",
                    "training_subject_has_duplicate_in_evaluation",
                }
                or sample.split not in (*SPLITS, "quarantine")
                or not sample.subject_id
                or sample.attack_family != family
                or sample.exclusion_reason != expected_exclusion
                or any(
                    len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    for digest in (sample.sha256, sample.bbox_sha256)
                )
            ):
                raise ValueError(f"Invalid inventory record: {sample.relative_path}")

            rows.append(sample)

    if not rows or len({r.relative_path for r in rows}) != len(rows):
        raise ValueError("Empty inventory or repeated image path")

    checks = check_split_integrity(rows)
    if checks.overlaps or checks.missing_families:
        raise ValueError("Inventory fails split/family checks")

    return tuple(rows)
