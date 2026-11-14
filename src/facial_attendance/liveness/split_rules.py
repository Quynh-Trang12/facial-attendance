"""Validate subject assignments and detect overlap between dataset splits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from facial_attendance.liveness.dataset_files import local_path, read_json
from facial_attendance.liveness.data_contract import FAMILY_NAMES, SPLITS


@dataclass(frozen=True)
class SubjectSplits:
    original_annotations: dict[str, dict]
    overlapping_subjects: set[str]
    validation_subjects: set[str]
    duplicate_train_subjects: set[str]

    @property
    def quarantined_train_subjects(self) -> set[str]:
        return self.overlapping_subjects | self.duplicate_train_subjects

    def project_split(self, original_split: str, subject_id: str) -> str:
        if original_split == "test":
            return "test"

        if (
            subject_id in self.overlapping_subjects
            or subject_id in self.duplicate_train_subjects
        ):
            return "quarantine"

        if subject_id in self.validation_subjects:
            return "validation"

        return "train"


def read_original_annotations(root: Path, paths: dict | None = None) -> dict:
    paths = (
        paths
        if paths is not None
        else {
            split: f"metas/intra_test/{split}_label.json" for split in ("train", "test")
        }
    )
    if not isinstance(paths, dict) or set(paths) != {"train", "test"}:
        raise ValueError(
            "Validation must be derived from original training annotations only"
        )

    return {
        split: read_json(local_path(root, paths[split])) for split in ("train", "test")
    }


def original_subject_ids(annotations: dict) -> dict[str, set[str]]:
    return {
        split: {PurePosixPath(name).parts[2] for name in labels}
        for split, labels in annotations.items()
    }


def validate_subject_splits(
    original_annotations: dict,
    validation_subjects: list[str],
    overlapping_subjects,
    duplicate_train_subjects=(),
) -> SubjectSplits:
    """Share subject constraints between preparation and inventory validation."""
    subject_ids_by_split = original_subject_ids(original_annotations)
    overlap = subject_ids_by_split["train"] & subject_ids_by_split["test"]
    if set(overlapping_subjects) != overlap:
        raise ValueError(
            "Quarantine must identify every original train/test subject overlap"
        )

    if (
        not isinstance(validation_subjects, list)
        or not validation_subjects
        or not all(isinstance(value, str) for value in validation_subjects)
        or len(set(validation_subjects)) != len(validation_subjects)
    ):
        raise ValueError(
            "Validation subject file requires unique validation subject IDs"
        )
    validation_subjects = set(validation_subjects)
    if not validation_subjects < (subject_ids_by_split["train"] - overlap):
        raise ValueError(
            "Validation subjects must be a strict subset of non-overlapping original train subjects"
        )

    duplicate_train_subjects = set(duplicate_train_subjects)
    if not duplicate_train_subjects <= (
        subject_ids_by_split["train"] - overlap - validation_subjects
    ):
        raise ValueError(
            "Duplicate quarantine may only exclude training subjects; validation/test are fixed"
        )

    return SubjectSplits(
        original_annotations, overlap, validation_subjects, duplicate_train_subjects
    )


def read_subject_splits(root: Path, policy: dict) -> SubjectSplits:
    if not isinstance(policy, dict) or not isinstance(
        policy.get("source_annotations"), dict
    ):
        raise ValueError(
            "Validation policy requires source_annotations and a validation subject file"
        )

    original_annotations = read_original_annotations(root, policy["source_annotations"])
    assignment = read_json(local_path(root, policy["validation_subject_file"]))

    return validate_subject_splits(
        original_annotations,
        assignment.get("validation_subjects"),
        policy.get("quarantined_train_subjects", []),
        policy.get("duplicate_train_subjects", []),
    )


@dataclass(frozen=True)
class SplitChecks:
    overlaps: list[dict]
    family_counts: Counter

    @property
    def missing_families(self) -> list[str]:
        return [
            f"{split}/{family}"
            for split in SPLITS
            for family in FAMILY_NAMES
            if self.family_counts[split, family] == 0
        ]


def check_split_integrity(records) -> SplitChecks:
    """Check all assigned records, including attacks excluded from training."""
    records = tuple(records)
    assigned_records = tuple(row for row in records if row.split in SPLITS)

    overlaps = []
    for field in ("subject_id", "sha256", "source_media_id", "session_id"):
        groups = defaultdict(set)
        for row in assigned_records:
            value = getattr(row, field)
            if value:
                groups[value].add(row.split)

        overlaps.extend(
            {"field": field, "value": value, "splits": sorted(splits)}
            for value, splits in sorted(groups.items())
            if len(splits) > 1
        )

    family_counts = Counter(
        (row.split, row.attack_family) for row in records if not row.exclusion_reason
    )

    return SplitChecks(overlaps, family_counts)
