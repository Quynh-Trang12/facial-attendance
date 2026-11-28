"""Freeze confirmed crop exclusions and lazy cache row maps for all three splits.

Unresolved reviewed cases remain explicit. A retained row is not a declaration
that its ROI has been semantically cleared by the unfinished full review.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from facial_attendance.liveness.audit.crop_content_scan import (  # noqa: E402
    SPLIT_ANNOTATIONS,
    input_records,
    reviewed_crop_disposition,
)
from facial_attendance.liveness.dataset_files import file_sha256  # noqa: E402
from facial_attendance.liveness.datasets.reviewed_cache import row_order_sha256


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, rows):
    fields = [
        "relative_path",
        "split",
        "original_split",
        "source_type",
        "attack_family",
        "binary_label",
        "decision",
        "reason_code",
        "reason",
        "supplied_bbox_pixels",
        "crop_width",
        "crop_height",
        "source_sha256",
        "bbox_sha256",
        "review_method",
        "evidence",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(project, reviews, policy_path, output):
    root = project / "data/raw/liveness/celeba-spoof-full"
    inventory = root / "audits/current-schema-2026-11-12/inventory.csv"
    audit_root = project / "evidence/liveness/crop-review/full-source-20261126"
    audit = json.loads((audit_root / "summary.json").read_text())
    if not audit["completed"] or audit["errors"]:
        raise ValueError("The complete source audit must finish without input errors")
    records, annotations = input_records(root, inventory)
    by_path = {row["relative_path"]: row for row in records}
    if (
        file_sha256(inventory) != audit["inputs"]["inventory_sha256"]
        or annotations != audit["inputs"]["annotation_sha256"]
    ):
        raise ValueError("Source assignment differs from the completed audit")

    policy = json.loads(policy_path.read_text())
    if policy.get("schema") != "liveness-crop-eligibility-policy-v3":
        raise ValueError(
            "Use the content-review policy without detector-count exclusion"
        )

    decisions = {}
    for review_path in reviews:
        packet = json.loads(review_path.read_text())
        for item in packet["candidates"]:
            name = item["relative_path"]
            if name not in by_path:
                raise ValueError(
                    "Review image is outside the experimental split assignments"
                )
            evidence = (review_path.parent / item["evidence"]).resolve()
            if not evidence.is_relative_to(project.resolve()) or not evidence.is_file():
                raise ValueError("Review evidence is missing or outside the project")
            if (
                item.get("evidence_sha256")
                and file_sha256(evidence) != item["evidence_sha256"]
            ):
                raise ValueError("Rendered review evidence changed")
            state = reviewed_crop_disposition(item)
            if not all(
                isinstance(item.get(field), str) and item[field].strip()
                for field in ("finding", "review_method")
            ):
                raise ValueError("Every reviewed case requires a finding and method")
            candidate = {
                **item,
                "state": state,
                "evidence_path": str(evidence.relative_to(project.resolve())),
            }
            previous = decisions.get(name)
            if previous:
                if {previous["state"], state} == {"retain", "quarantine"}:
                    raise ValueError("Contradictory confirmed content-review decisions")
                if state == "review" and previous["state"] != "review":
                    continue
            decisions[name] = candidate

    with closing(sqlite3.connect(audit_root / "crops.sqlite")) as db:
        details = {
            name: json.loads(
                db.execute(
                    "SELECT result FROM samples WHERE path=?", (name,)
                ).fetchone()[0]
            )
            for name in decisions
        }

    reviewed_rows = []
    for name, item in sorted(decisions.items()):
        record, crop = by_path[name], details[name]
        image = root / name
        bbox = image.with_name(image.stem + "_BB.txt")
        if (
            file_sha256(image) != record["sha256"]
            or file_sha256(bbox) != record["bbox_sha256"]
        ):
            raise ValueError("Reviewed source image or bbox changed")
        reason = item["finding"]
        if item["state"] == "quarantine":
            reason += (
                f" Supplied ROI: {crop['crop_width']} × {crop['crop_height']} source pixels, "
                "resized to 224 × 224; exclusion is based on the recorded visual content finding, not size or detector count alone."
            )
        reviewed_rows.append(
            {
                "relative_path": name,
                "split": record["audit_split"],
                "original_split": record["original_split"],
                "source_type": record["source_type"],
                "attack_family": record["attack_family"],
                "binary_label": record["binary_label"],
                "decision": item["state"],
                "reason_code": item.get("reason_code", item["decision"]),
                "reason": reason,
                "supplied_bbox_pixels": json.dumps(crop["crop_box"]),
                "crop_width": crop["crop_width"],
                "crop_height": crop["crop_height"],
                "source_sha256": record["sha256"],
                "bbox_sha256": record["bbox_sha256"],
                "review_method": item["review_method"],
                "evidence": item["evidence_path"],
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    quarantine = [row for row in reviewed_rows if row["decision"] == "quarantine"]
    pending = [row for row in reviewed_rows if row["decision"] == "review"]
    write_csv(output / "quarantine.csv", quarantine)
    write_csv(output / "pending-review.csv", pending)
    write_csv(output / "reviewed.csv", reviewed_rows)
    write_json(output / "policy.json", policy)

    inventory = {
        "schema": "liveness-reviewed-cache-view-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Frozen confirmed exclusions from prioritized review batch; not full dataset clearance",
        "inventory_sha256": file_sha256(inventory),
        "source_audit_summary_sha256": file_sha256(audit_root / "summary.json"),
        "review_bindings": {
            str(path.relative_to(project)): file_sha256(path) for path in reviews
        },
        "policy_sha256": file_sha256(output / "policy.json"),
        "quarantine_sha256": file_sha256(output / "quarantine.csv"),
        "pending_review_sha256": file_sha256(output / "pending-review.csv"),
        "base_pixels_modified": False,
        "labels_modified": False,
        "split_assignments_moved": False,
        "pending_review_handling": "Not labelled as confirmed defects; normal training access blocked for an affected split unless diagnostic access is explicit",
        "splits": {},
    }

    rejected_names = {r["relative_path"] for r in quarantine}
    pending_names = {r["relative_path"] for r in pending}
    breakdown = []
    for split in SPLIT_ANNOTATIONS:
        original = [r for r in records if r["audit_split"] == split]
        names = [r["relative_path"] for r in original]
        keep = np.array(
            [i for i, name in enumerate(names) if name not in rejected_names],
            dtype=np.int64,
        )
        removed = np.array(
            [i for i, name in enumerate(names) if name in rejected_names],
            dtype=np.int64,
        )
        unresolved = np.array(
            [i for i, name in enumerate(names) if name in pending_names], dtype=np.int64
        )
        for label, indices in (
            ("keep", keep),
            ("quarantine", removed),
            ("pending", unresolved),
        ):
            np.save(
                output / f"{split}-{label}-indices.npy", indices, allow_pickle=False
            )
        identities = (
            SimpleNamespace(
                relative_path=r["relative_path"],
                binary_label=r["binary_label"],
                source_type=r["source_type"],
            )
            for r in original
        )
        inventory["splits"][split] = {
            "annotations_sha256": annotations[split],
            "base_samples": len(original),
            "base_row_order_sha256": row_order_sha256(identities),
            "retained_samples": len(keep),
            "quarantined_samples": len(removed),
            "pending_review": len(unresolved),
            "keep_indices": f"{split}-keep-indices.npy",
            "keep_indices_sha256": file_sha256(output / f"{split}-keep-indices.npy"),
        }
        for source_type in sorted({int(r["source_type"]) for r in original}):
            group = [r for r in original if int(r["source_type"]) == source_type]
            count = Counter(
                "quarantine"
                if r["relative_path"] in rejected_names
                else "review"
                if r["relative_path"] in pending_names
                else "reviewed_retained"
                if r["relative_path"] in decisions
                else "not_reviewed_in_batch"
                for r in group
            )
            breakdown.append(
                {
                    "split": split,
                    "source_type": source_type,
                    "base": len(group),
                    "quarantined": count["quarantine"],
                    "pending_review": count["review"],
                    "retained": len(group) - count["quarantine"],
                    "reviewed_retained": count["reviewed_retained"],
                    "not_reviewed_in_batch": count["not_reviewed_in_batch"],
                }
            )

    inventory["counts_by_split_and_source_type"] = breakdown
    write_json(output / "cache-view.json", inventory)
    write_json(
        output / "summary.json",
        {
            "reviewed_images": len(reviewed_rows),
            "quarantined": len(quarantine),
            "pending_review": len(pending),
            "counts_by_split_and_source_type": breakdown,
            "scope": inventory["scope"],
            "base_pixels_modified": False,
        },
    )
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    inventory = build(
        args.project.resolve(),
        [p.resolve() for p in args.reviews],
        args.policy.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                split: {
                    k: v
                    for k, v in record.items()
                    if k
                    in (
                        "base_samples",
                        "retained_samples",
                        "quarantined_samples",
                        "pending_review",
                    )
                }
                for split, record in inventory["splits"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
