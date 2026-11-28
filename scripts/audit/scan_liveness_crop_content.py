"""Inspect source crops across fixed train/validation/test memberships.

Diagnostics never change source boxes, labels, checkpoints or training caches.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import cv2

from facial_attendance.liveness.audit.crop_content_scan import (  # noqa: E402
    SIZE_BANDS,
    SPLIT_ANNOTATIONS,
    input_records,
    inspect_crop,
)
from facial_attendance.liveness.dataset_files import file_sha256  # noqa: E402


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def summary(db, inventory, started):
    count = db.execute("SELECT count(*) FROM samples").fetchone()[0]
    return {
        "schema": "liveness-source-crop-audit-v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "inputs": inventory,
        "completed_rows": count,
        "split_counts": dict(
            db.execute("SELECT split,count(*) FROM samples GROUP BY split")
        ),
        "errors": dict(
            db.execute(
                "SELECT split,count(*) FROM samples WHERE error IS NOT NULL GROUP BY split"
            )
        ),
        "native_min_edge_bands": list(
            db.execute(
                "SELECT split,json_extract(result,'$.size_band'),count(*) FROM samples GROUP BY 1,2"
            )
        ),
        "review_flags": list(
            db.execute(
                "SELECT split,flag.value,count(*) FROM samples,json_each(samples.flags) AS flag GROUP BY 1,2"
            )
        ),
        "elapsed_seconds_this_execution": time.monotonic() - started,
        "completed": count == inventory["requested_rows"],
        "semantic_review": "Unverified: automated diagnostics do not establish intended target identity",
        "cache_pixel_parity": "Not checked: original source images and boxes only",
        "eligibility_changes": 0,
    }


def run(args):
    started = time.monotonic()
    root = args.data_root.resolve()
    inventory = root / "audits/current-schema-2026-11-12/inventory.csv"
    records, annotation_hashes = input_records(root, inventory)
    scope_counts = {
        split: sum(r["audit_split"] == split for r in records)
        for split in SPLIT_ANNOTATIONS
    }
    if args.workers < 1 or (
        args.sample_per_split is not None and args.sample_per_split < 1
    ):
        raise ValueError("Workers and optional sample count must be positive")
    if args.sample_per_split:
        chosen = set(args.include)
        for split in SPLIT_ANNOTATIONS:
            candidates = [r for r in records if r["audit_split"] == split]
            candidates.sort(
                key=lambda r: hashlib.sha256(r["relative_path"].encode()).hexdigest()
            )
            chosen.update(
                r["relative_path"] for r in candidates[: args.sample_per_split]
            )
        records = [r for r in records if r["relative_path"] in chosen]
        if chosen - {r["relative_path"] for r in records}:
            raise ValueError("Included example is absent from the memberships")
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 1024**3:
        raise ValueError(
            "Need at least 1 GiB free for audit records; no cache copies are made"
        )
    detector = args.detector.resolve() if args.detector else None
    inventory = {
        "root": str(root),
        "inventory_sha256": file_sha256(inventory),
        "annotation_sha256": annotation_hashes,
        "full_scope_counts": scope_counts,
        "requested_rows": len(records),
        "sample_per_split": args.sample_per_split,
        "included_paths": sorted(args.include),
        "detector_sha256": file_sha256(detector) if detector else None,
        "detector_runtime": "Existing pinned YuNet/OpenCV CPU locator"
        if detector
        else "Not run",
        "size_bands_pixels": list(SIZE_BANDS),
        "size_policy": "Descriptive bands only; no minimum acceptance threshold chosen",
        "source_sha256": {
            "entry_point": file_sha256(Path(__file__)),
            "audit_module": file_sha256(
                Path(__file__).parents[2]
                / "src/facial_attendance/liveness/audit/crop_content_scan.py"
            ),
        },
    }
    inventory_path = args.output / "inputs.json"
    if inventory_path.exists():
        if json.loads(inventory_path.read_text()) != inventory:
            raise ValueError(
                "Existing audit belongs to different inputs, scope or source"
            )
    else:
        write_json(inventory_path, inventory)
    # Each CPU detector is thread-local. Disable nested OpenCV parallelism
    # so the explicit worker count controls diagnostic resource use.
    cv2.setNumThreads(1)
    with sqlite3.connect(args.output / "crops.sqlite") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS samples(path TEXT PRIMARY KEY, split TEXT, label INTEGER, source_type INTEGER, min_edge INTEGER, laplacian REAL, flags TEXT, error TEXT, result TEXT)"
        )
        existing = {r[0] for r in db.execute("SELECT path FROM samples")}
        if existing - {r["relative_path"] for r in records}:
            raise ValueError("Audit database includes rows outside the bound scope")
        pending_records = iter(r for r in records if r["relative_path"] not in existing)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = deque()

            def submit():
                record = next(pending_records, None)
                if record is not None:
                    pending.append(
                        pool.submit(inspect_crop, root, record, detector_model=detector)
                    )

            for _ in range(args.workers * 2):
                submit()
            last_report = time.monotonic()
            while pending:
                result = pending.popleft().result()
                db.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        result["relative_path"],
                        result["split"],
                        result["binary_label"],
                        result["source_type"],
                        result["native_min_edge"],
                        result["laplacian_variance"],
                        json.dumps(result["review_flags"]),
                        result["error"],
                        json.dumps(result, allow_nan=False),
                    ),
                )
                submit()
                if time.monotonic() - last_report >= 10:
                    db.commit()
                    report = summary(db, inventory, started)
                    write_json(args.output / "summary.json", report)
                    print(
                        json.dumps(
                            {
                                "checked": report["completed_rows"],
                                "total": len(records),
                                "elapsed_seconds": round(
                                    report["elapsed_seconds_this_execution"], 1
                                ),
                            }
                        ),
                        flush=True,
                    )
                    last_report = time.monotonic()
        db.commit()
        report = summary(db, inventory, started)
        write_json(args.output / "summary.json", report)

    print(json.dumps(report, indent=2), flush=True)
    return 1 if report["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--detector",
        type=Path,
        help="Optional pinned YuNet model; diagnostic CPU inference",
    )
    parser.add_argument(
        "--sample-per-split",
        type=int,
        help="Explicit preflight sample, not a full audit",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Known example included in preflight",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
