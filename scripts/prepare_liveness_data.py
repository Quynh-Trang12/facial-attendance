"""Extract the complete CelebA-Spoof dataset and apply fixed subject splits."""

import argparse
import bisect
from collections import Counter
import csv
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.liveness.dataset_files import file_sha256, read_json
from facial_attendance.liveness.data_contract import FAMILIES, SOURCE
from facial_attendance.liveness.split_rules import (
    original_subject_ids,
    read_original_annotations,
    validate_subject_splits,
)


class ArchiveParts(io.RawIOBase):
    """Seekable view over ordinary sequential .001, .002, ... archive chunks."""

    def __init__(self, paths):
        super().__init__()
        self.paths = tuple(paths)
        self.offsets, self.position = [0], 0
        self._handle, self._part = None, None

        for path in self.paths:
            self.offsets.append(self.offsets[-1] + path.stat().st_size)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")

        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.offsets[-1] + offset
        )
        if position < 0:
            raise ValueError("Negative seek")

        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.offsets[-1] - self.position

        chunks = []
        while size > 0 and self.position < self.offsets[-1]:
            part = bisect.bisect_right(self.offsets, self.position) - 1
            if part != self._part:
                if self._handle is not None:
                    self._handle.close()
                self._handle, self._part = self.paths[part].open("rb"), part

            self._handle.seek(self.position - self.offsets[part])
            data = self._handle.read(min(size, self.offsets[part + 1] - self.position))
            if not data:
                raise OSError(f"Truncated archive part: {self.paths[part]}")

            chunks.append(data)
            self.position += len(data)
            size -= len(data)

        return b"".join(chunks)

    def close(self):
        if self._handle is not None:
            self._handle.close()
        super().close()


def extract_full(parts_dir: Path, output: Path):
    """Extract and CRC-check every entry, without creating a merged ZIP copy."""
    parts = [parts_dir / f"CelebA_Spoof.zip.{number:03}" for number in range(1, 75)]
    for number, path in enumerate(parts, 1):
        if path.stat().st_size != (453369048 if number == 74 else 1048576000):
            raise ValueError(f"Unexpected archive part size: {path}")

    with ArchiveParts(parts) as stream, zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate archive entry names")

        payload = sum(entry.file_size for entry in entries)
        # Small bounding-box files consume filesystem blocks, not just their
        # short text payload. Reserve another 2 GiB for inventories and audit output.
        estimate = sum(
            ((e.file_size + 4095) // 4096) * 4096 if not e.is_dir() else 4096
            for e in entries
        )
        free = shutil.disk_usage(output.parent).free
        if free < estimate + 2 * 2**30:
            raise ValueError(
                f"Need approximately {estimate + 2 * 2**30} free bytes; available {free}"
            )

        for entry in entries:
            relative = PurePosixPath(entry.filename)
            if (
                not relative.parts
                or relative.parts[0] != "CelebA_Spoof"
                or ".." in relative.parts
                or ":" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive entry: {entry.filename}")

        output.mkdir(parents=True, exist_ok=False)
        files = 0

        for entry in entries:
            relative = PurePosixPath(entry.filename)
            target = output.joinpath(*relative.parts[1:])
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            files += 1
            if files % 50000 == 0:
                print(f"Full extraction: {files} files, CRC checked", flush=True)

        report = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "archive_parts": [
                {"name": p.name, "bytes": p.stat().st_size} for p in parts
            ],
            "archive_entries": len(entries),
            "extracted_files": files,
            "uncompressed_payload_bytes": payload,
            "crc_all_files": "PASS",
            "selection": "All archive entries; no image sampling",
            "script_sha256": file_sha256(Path(__file__)),
        }

        (output / "extraction.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)


def configure_validation(output: Path, folds: int):
    """Record the agreed validation subject selection using full source-train annotations."""
    # Only split preparation needs scikit-learn; multipart inspection stays
    # usable with Python's standard library and the source-check environment.
    import numpy as np
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    if folds != 10:
        raise ValueError("The agreed validation policy uses 10 folds")

    original_annotations = read_original_annotations(output)
    subject_ids_by_split = original_subject_ids(original_annotations)
    overlapping_subjects = subject_ids_by_split["train"] & subject_ids_by_split["test"]
    items = sorted(
        (n, a)
        for n, a in original_annotations["train"].items()
        if a[40] in FAMILIES and PurePosixPath(n).parts[2] not in overlapping_subjects
    )

    y = np.array([a[40] for _, a in items])
    groups = np.array([PurePosixPath(n).parts[2] for n, _ in items])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    train, val = next(splitter.split(np.zeros(len(items)), y, groups))
    validation_subjects = set(groups[val])

    method = f"StratifiedGroupKFold(n_splits={folds}, shuffle=True, random_state=42), first fold; stratify source type, group subject"
    assignment = {
        "method": method,
        "sklearn_version": sklearn.__version__,
        "validation_subjects": sorted(validation_subjects),
        "quarantined_train_subjects": sorted(overlapping_subjects),
        "full_source_train_counts": {
            "images": len(train),
            "subjects": len(set(groups[train])),
            "source_types": dict(Counter(str(y[i]) for i in train)),
        },
        "full_source_validation_counts": {
            "images": len(val),
            "subjects": len(validation_subjects),
            "source_types": dict(Counter(str(y[i]) for i in val)),
        },
    }

    # Existing evidence is never overwritten; each policy needs its own prepared root.
    with (output / "validation-subjects.json").open("x", encoding="utf-8") as handle:
        json.dump(assignment, handle, indent=2)
        handle.write("\n")

    configure_dataset(output, output / "validation-subjects.json")


def configure_dataset(
    output: Path, validation_subject_file: Path, *, duplicate_subjects=None
):
    """Apply the already fixed subject list to every original annotated image."""
    assignment = read_json(validation_subject_file)

    original_annotations = read_original_annotations(output)
    assignments = validate_subject_splits(
        original_annotations,
        assignment["validation_subjects"],
        assignment["quarantined_train_subjects"],
        duplicate_subjects or (),
    )
    overlapping_subjects = assignments.overlapping_subjects
    reviewed = duplicate_subjects is not None
    duplicate_subjects = assignments.duplicate_train_subjects

    project_annotations = {s: {} for s in ("train", "validation", "test", "quarantine")}
    for original_split, labels in original_annotations.items():
        for name, attributes in labels.items():
            subject = PurePosixPath(name).parts[2]
            split = assignments.project_split(original_split, subject)
            project_annotations[split][name] = attributes

    destination = output / "validation-subjects.json"
    if destination.exists():
        if destination.read_bytes() != validation_subject_file.read_bytes():
            raise ValueError(
                "Existing validation subject file differs from the agreed split assignment"
            )
    else:
        with destination.open("xb") as handle:
            handle.write(validation_subject_file.read_bytes())

    annotation_paths = {}
    for split, labels in project_annotations.items():
        filename = f"project_{split}.json"
        if reviewed and split in {"validation", "test"}:
            if read_json(output / filename) != labels:
                raise ValueError(
                    f"Reviewed preparation would change {split} split assignment"
                )
            annotation_paths[split] = filename
            continue
        filename = ("reviewed_" if reviewed else "") + filename
        annotation_paths[split] = filename
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(labels, handle, sort_keys=True)
            handle.write("\n")

    data_description = {
        "source_url": SOURCE,
        "dataset_archive": "CelebA_Spoof.zip.001-.074, full extracted dataset",
        "received_on": datetime.now(timezone.utc).date().isoformat(),
        "split_evidence": "Original README and metas/intra_test train/test JSON; fixed project-derived validation subject file",
        "selection": "Every original image inventoried and decoded, including excluded attacks and quarantined source-train subjects; no sampling",
        "selection_scope": "full_dataset",
        "split_annotations": {
            s: annotation_paths[s] for s in ("train", "validation", "test")
        },
        "quarantine_annotations": annotation_paths["quarantine"],
        "validation_from_train": {
            "method": assignment["method"],
            "validation_subject_file": "validation-subjects.json",
            "source_annotations": {
                s: f"metas/intra_test/{s}_label.json" for s in original_annotations
            },
            "quarantined_train_subjects": sorted(overlapping_subjects),
            "duplicate_train_subjects": sorted(duplicate_subjects),
        },
    }
    if reviewed:
        data_description["quality_findings"] = "quality-findings.json"

    with (
        output
        / ("reviewed-data-description.json" if reviewed else "data-description.json")
    ).open("x", encoding="utf-8") as handle:
        json.dump(data_description, handle, indent=2)
        handle.write("\n")

    print(
        json.dumps({s: len(labels) for s, labels in project_annotations.items()}),
        flush=True,
    )


def resolve_audit(output: Path, audit_file: Path):
    """Quarantine training subjects proven to share identical images with evaluation."""
    audit = read_json(audit_file)
    if Path(audit["data_root"]).resolve() != output.resolve():
        raise ValueError("Audit belongs to a different dataset root")
    if (
        file_sha256(output / "data-description.json")
        != audit["data_description_sha256"]
    ):
        raise ValueError("Original data description changed since audit")

    overlaps = audit["overlaps"]
    if not overlaps or any(
        o["field"] != "sha256"
        or "train" not in o["splits"]
        or not set(o["splits"]) & {"validation", "test"}
        for o in overlaps
    ):
        raise ValueError(
            "This resolution only handles proven train/evaluation byte duplicates"
        )

    hashes = {o["value"] for o in overlaps}
    inventory = audit_file.parent / "rejected-inventory.csv"
    subjects, rows = set(), []
    with inventory.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sha256"] in hashes:
                rows.append(row)
                if row["split"] == "train":
                    subjects.add(row["subject_id"])

    findings = {
        "original_audit_sha256": file_sha256(audit_file),
        "original_inventory_sha256": file_sha256(inventory),
        "duplicate_groups": len(hashes),
        "duplicate_records": rows,
        "duplicate_train_subjects": sorted(subjects),
        "image_errors": audit["image_errors"],
        "resolution": "Quarantine all images of affected training subjects; preserve fixed validation and original test; source files unchanged",
    }

    with (output / "quality-findings.json").open("x", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2)
        handle.write("\n")

    configure_dataset(
        output, output / "validation-subjects.json", duplicate_subjects=subjects
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--full",
        action="store_true",
        help="Extract all entries with CRC checks; no sampling",
    )
    stages.add_argument(
        "--configure-validation",
        type=int,
        choices=(10,),
        help="Reproduce the agreed first-fold validation subject selection from original training annotations",
    )
    stages.add_argument(
        "--validation-subject-file",
        type=Path,
        help="Apply an existing fixed subject assignment to every source image",
    )
    stages.add_argument(
        "--resolve-audit",
        type=Path,
        help="Quarantine training subjects with proven byte duplicates in an earlier full audit",
    )
    args = parser.parse_args()
    if args.resolve_audit:
        resolve_audit(args.output_dir, args.resolve_audit)
    elif args.validation_subject_file:
        configure_dataset(args.output_dir, args.validation_subject_file)
    elif args.configure_validation:
        configure_validation(args.output_dir, args.configure_validation)
    elif args.parts_dir:
        extract_full(args.parts_dir, args.output_dir)
    else:
        parser.error("--parts-dir is required for full extraction")
