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
    QUALITY_REQUIREMENTS,
    TASK_REQUIREMENTS,
    EXPERIMENT_REQUIREMENTS,
    EXPERIMENT_IMPLEMENTATION_FILES,
    SELECTION_REASONS,
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


def read_quality_report(root: Path, path: Path, inventory_sha256: str) -> dict:
    """Verify the receipt's input and evidence bindings before using decisions."""
    report = read_json(path)
    if not set(QUALITY_REQUIREMENTS) <= set(report.get("requirements", {})):
        raise ValueError("Quality report is missing required checks")
    accepted = all(value == "PASS" for value in report["requirements"].values())
    failed = any(value.startswith("FAIL") for value in report["requirements"].values())
    result = "PASS" if accepted else "FAIL" if failed else "INCOMPLETE"
    if report.get("accepted") is not accepted or report.get("result") != result:
        raise ValueError("Quality result disagrees with its requirement checks")
    if (Path(report["data_root"]).resolve() != root.resolve()
            or report["inventory_sha256"] != inventory_sha256):
        raise ValueError("Quality report belongs to another inventory/data root")
    if not report.get("source_bindings") or not report.get("evidence_bindings"):
        raise ValueError("Quality report has no source/evidence bindings")
    for category in ("source_bindings", "evidence_bindings", "implementation_sha256"):
        if category == "implementation_sha256" and not report.get(category):
            raise ValueError("Quality report has no implementation binding")
        for filename, digest in report[category].items():
            if file_sha256(Path(filename)) != digest:
                raise ValueError(f"Quality {category} changed: {filename}")
    if file_sha256(Path(report["selection_file"])) != report["selection_sha256"]:
        raise ValueError("Quality selection changed since assessment")
    return report


def read_selection(path: Path, records) -> set[str]:
    """Apply only explicit exclusions, retaining all original inventory rows."""
    known = {r.relative_path for r in records if not r.exclusion_reason}
    excluded = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", "exclusion_reason"]:
            raise ValueError("Unexpected quality selection columns")
        for row in reader:
            name = row["relative_path"]
            if (name not in known or name in excluded or row["exclusion_reason"] not in
                    SELECTION_REASONS):
                raise ValueError(f"Invalid quality selection: {name}")
            excluded.add(name)
    return excluded


def read_experimental_report(root: Path, path: Path, inventory_sha256: str) -> dict:
    """Authorize a bound training experiment without claiming full visual truth."""
    report = read_json(path)
    checks = report.get("readiness_checks", {})
    if (report.get("report_type") != "experimental_readiness" or report.get("schema_version") != 1
            or not set(EXPERIMENT_REQUIREMENTS) <= set(checks)
            or any(value != "PASS" for value in checks.values())
            or report.get("experimental_ready") is not True or report.get("result") != "PASS"
            or report.get("pipeline_completed") is not True):
        raise ValueError("Experimental readiness checks have not passed")
    if (Path(report["data_root"]).resolve() != root.resolve()
            or report.get("inventory_sha256") != inventory_sha256):
        raise ValueError("Experimental readiness belongs to another inventory/data root")
    for category in ("source_bindings", "evidence_bindings", "implementation_sha256"):
        if not report.get(category):
            raise ValueError(f"Experimental readiness has no {category}")
        for filename, digest in report[category].items():
            if file_sha256(Path(filename)) != digest:
                raise ValueError(f"Experimental readiness {category} changed: {filename}")
    expected = {str((Path(__file__).parent/name).resolve()) for name in EXPERIMENT_IMPLEMENTATION_FILES}
    expected.add(str((Path(__file__).parent.parent/'recognition/data_loader.py').resolve()))
    if not expected <= set(report['implementation_sha256']):
        raise ValueError('Experimental readiness implementation coverage is incomplete')
    if file_sha256(Path(report['selection_file'])) != report['selection_sha256']:
        raise ValueError('Experimental selection changed since assessment')
    policy = report.get('candidate_policy', {})
    if policy.get('search_complete') is not True or policy.get('remaining_actionable_pairs') != 0:
        raise ValueError('Experimental candidate policy is unresolved')
    counts = report.get('cleaned_subset_counts', {})
    if (set(counts) != set(SPLITS) or any(type(n) is not int or n <= 0 for n in counts.values())
            or type(report.get('selected_rows')) is not int or sum(counts.values()) != report['selected_rows']):
        raise ValueError('Experimental selected-row counts are invalid')
    return report


def read_audited_inventory(root: Path, inventory: Path, audit_file: Path):
    """Load accepted evidence only; standalone CSVs are diagnostic inputs."""
    if not audit_file.is_file():
        raise ValueError("LIV-03 audit report required; standalone inventory is diagnostic only")
    audit = read_json(audit_file)
    if audit.get('report_type') == 'experimental_readiness':
        experiment = read_experimental_report(root, audit_file, file_sha256(inventory))
        records = read_inventory(inventory)
        excluded = read_selection(Path(experiment['selection_file']), records)
        counts = {split: sum(r.split == split and not r.exclusion_reason and r.relative_path not in excluded
                             for r in records) for split in SPLITS}
        if counts != experiment['cleaned_subset_counts']:
            raise ValueError('Experimental selected-row coverage differs')
        return records, excluded
    requirements = audit.get("requirements", {})
    if (audit.get("accepted") is not True or audit.get("result") != "PASS"
            or not audit.get("pipeline_completed") or not set(TASK_REQUIREMENTS) <= set(requirements)
            or any(value != "PASS" for value in requirements.values())):
        raise ValueError("LIV-03 evidence is incomplete; inventory is not accepted for model use")
    digest = file_sha256(inventory)
    if digest != audit["inventory_sha256"]:
        raise ValueError("Inventory changed since audit")
    binding = audit.get("quality_report")
    if not binding or file_sha256(Path(binding["path"])) != binding["sha256"]:
        raise ValueError("Missing or changed quality assessment")
    quality = read_quality_report(root, Path(binding["path"]), digest)
    if (quality.get("accepted") is not True or quality.get("result") != "PASS"
            or not quality.get("requirements")
            or any(value != "PASS" for value in quality["requirements"].values())):
        raise ValueError("Quality evidence is incomplete")
    records = read_inventory(inventory)
    excluded = read_selection(Path(quality["selection_file"]), records)
    if sum(not r.exclusion_reason and r.relative_path not in excluded for r in records) != quality["selected_rows"]:
        raise ValueError("Quality selected-row coverage differs")
    return records, excluded


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
