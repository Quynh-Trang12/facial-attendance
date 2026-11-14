"""Scan unchanged source images and derive traceable exact-duplicate exclusions."""

from collections import Counter, deque
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import sqlite3

from PIL import Image

from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import bbox_path, face_box
from facial_attendance.liveness.image_fingerprints import pixel_sha256, perceptual_hash
from facial_attendance.liveness.image_inventory import build_inventory, read_inventory


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def source_records(root, inventory, description):
    """Bind every inventory label/split/relationship to its current source."""
    records = read_inventory(inventory)
    source = build_inventory(root, description)
    expected = {r.relative_path: r for r in source.records}
    if len(records) != len(expected):
        raise ValueError("Inventory/source row coverage differs")
    for row in records:
        actual = asdict(row)
        actual["sha256"] = actual["bbox_sha256"] = ""
        if row.relative_path not in expected or actual != asdict(expected[row.relative_path]):
            raise ValueError(f"Inventory differs from source: {row.relative_path}")
    return records


def fingerprint(root, row):
    pixel = crop = full_hash = crop_hash = error = ""
    retained = 0.0
    image_path = local_path(root, row.relative_path)
    box_path = local_path(root, str(bbox_path(Path(row.relative_path))))
    if file_sha256(image_path) != row.sha256 or file_sha256(box_path) != row.bbox_sha256:
        raise ValueError(f"Image/box changed since audit: {row.relative_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
            image = image.convert("RGB")
            pixel, full_hash = pixel_sha256(image), perceptual_hash(image)
            box, retained = face_box(box_path, image.size)
            face = image.crop(box)
            crop, crop_hash = pixel_sha256(face), perceptual_hash(face)
    except (OSError, ValueError) as failure:
        error = str(failure)
    return (row.relative_path, row.split, row.source_type, row.binary_label,
            row.exclusion_reason, row.sha256, pixel, crop, full_hash, crop_hash,
            retained, error)


def scan(root, inventory, description, output, *, workers=4):
    """Verify source binding and fingerprint all rows, including excluded attacks."""
    if workers < 1:
        raise ValueError("workers must be positive")
    records = source_records(root, inventory, description)
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise ValueError("Need at least 1 GiB free for quality index and reports")
    output.mkdir(exist_ok=False, parents=True)
    database = output / "fingerprints.sqlite"
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE images(path TEXT PRIMARY KEY, split TEXT, source_type INTEGER, label INTEGER, exclusion TEXT, file_hash TEXT, pixel_hash TEXT, crop_hash TEXT, full_phash TEXT, crop_phash TEXT, retained REAL, error TEXT)")
        source = iter(records)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for _ in range(workers * 2):
                row = next(source, None)
                if row is not None:
                    pending.append(pool.submit(fingerprint, root, row))
            count = 0
            while pending:
                result = pending.popleft().result()
                db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", result)
                count += 1
                row = next(source, None)
                if row is not None:
                    pending.append(pool.submit(fingerprint, root, row))
                if count % 10000 == 0:
                    db.commit()
                    print(f"Fingerprinted {count}/{len(records)}", flush=True)
        db.commit()
        for field in ("file_hash", "pixel_hash", "crop_hash"):
            db.execute(f"CREATE INDEX idx_{field} ON images({field})")
        db.commit()
    source = build_inventory(root, description)
    bindings = {str(Path(description).resolve()): file_sha256(description)}
    for item in source.annotations.values():
        bindings[str(local_path(root, item["path"]))] = item["sha256"]
    for key in ("relationships", "quality_findings"):
        if source.source.get(key):
            path = local_path(root, source.source[key])
            bindings[str(path)] = file_sha256(path)
    if source.source.get("validation_from_train"):
        policy = source.source["validation_from_train"]
        for name in (*policy["source_annotations"].values(), policy["validation_subject_file"]):
            path = local_path(root, name)
            bindings[str(path)] = file_sha256(path)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "data_root": str(root.resolve()), "inventory_sha256": file_sha256(inventory),
              "inventory_rows": len(records), "source_bindings": bindings,
              "fingerprints_sha256": file_sha256(database),
              "implementation_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(output / "scan.json", report)
    return report


def exact_duplicates(output):
    """Keep test, then validation, then train; never rewrite original assignments.

    Exclusions define a cleaned subset, not a replacement benchmark. Full-image
    pixel equality is safe to resolve automatically when labels/types agree.
    Crop-only equality is sent for review: different media can show one face.
    """
    decisions, conflicts, group_count, crop_groups = {}, 0, 0, 0
    priority = {"test": 0, "validation": 1, "train": 2}
    with closing(sqlite3.connect(output / "fingerprints.sqlite")) as db, db:
        with (output / "exact-groups.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "digest", "relative_path", "split", "source_type", "decision"])
            for kind in ("file_hash", "pixel_hash", "crop_hash"):
                groups = db.execute(f"SELECT {kind} FROM images WHERE exclusion='' AND {kind}!='' GROUP BY {kind} HAVING COUNT(*)>1")
                for (digest,) in groups:
                    rows = db.execute(f"SELECT path,split,source_type,label FROM images WHERE exclusion='' AND {kind}=? ORDER BY path", (digest,)).fetchall()
                    if kind == "crop_hash":
                        crop_groups += 1
                        for path, split, code, _ in rows:
                            writer.writerow([kind, digest, path, split, code, "review_crop_context"])
                        continue
                    group_count += 1
                    conflict = len({(r[2], r[3]) for r in rows}) > 1
                    conflicts += conflict
                    keep = min(rows, key=lambda r: (priority[r[1]], r[0]))[0]
                    for path, split, code, _ in rows:
                        reason = "duplicate_label_conflict" if conflict else "exact_duplicate" if path != keep else ""
                        if reason:
                            decisions[path] = reason
                        writer.writerow([kind, digest, path, split, code, reason or "representative"])
        with (output / "selection.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["relative_path", "exclusion_reason"])
            writer.writerows(sorted(decisions.items()))
        remaining = {split: db.execute("SELECT COUNT(*) FROM images WHERE split=? AND exclusion=''", (split,)).fetchone()[0] for split in priority}
        for path in decisions:
            split = db.execute("SELECT split FROM images WHERE path=?", (path,)).fetchone()[0]
            remaining[split] -= 1
    report = {"exact_duplicate_groups_by_fingerprint": group_count,
              "label_conflict_groups_by_fingerprint": conflicts,
              "crop_only_review_groups": crop_groups,
              "additional_exclusions": len(decisions), "cleaned_subset_counts": remaining,
              "selection_sha256": file_sha256(output / "selection.csv"),
              "groups_sha256": file_sha256(output / "exact-groups.csv"),
              "policy": "Lexicographic representative; prefer test over validation over train; conflicts excluded pending review; original inventory/splits unchanged",
              "result": "PASS" if not conflicts else "UNVERIFIED"}
    write_json(output / "exact.json", report)
    return report
