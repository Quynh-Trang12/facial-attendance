"""Scan unchanged source images and derive traceable exact-duplicate exclusions."""

import csv
import json
import shutil
import sqlite3
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from facial_attendance.liveness.audit.image_fingerprints import (
    complete_hamming_pairs,
    perceptual_hash,
    pixel_sha256,
)
from facial_attendance.liveness.dataset_files import (
    file_sha256,
    local_path,
    read_json,
    source_path,
    source_root,
)
from facial_attendance.liveness.datasets.face_crop import bbox_path, face_box
from facial_attendance.liveness.datasets.image_inventory import (
    build_inventory,
    read_inventory,
)


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
        if row.relative_path not in expected or actual != asdict(
            expected[row.relative_path]
        ):
            raise ValueError(f"Inventory differs from source: {row.relative_path}")
    return records


def fingerprint(root, row):
    pixel = crop = full_hash = crop_hash = error = ""
    retained = 0.0
    image_path = local_path(root, row.relative_path)
    box_path = local_path(root, str(bbox_path(Path(row.relative_path))))
    if (
        file_sha256(image_path) != row.sha256
        or file_sha256(box_path) != row.bbox_sha256
    ):
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
    return (
        row.relative_path,
        row.split,
        row.source_type,
        row.binary_label,
        row.exclusion_reason,
        row.sha256,
        pixel,
        crop,
        full_hash,
        crop_hash,
        retained,
        error,
    )


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
        db.execute(
            "CREATE TABLE images(path TEXT PRIMARY KEY, split TEXT, source_type INTEGER, label INTEGER, exclusion TEXT, file_hash TEXT, pixel_hash TEXT, crop_hash TEXT, full_phash TEXT, crop_phash TEXT, retained REAL, error TEXT)"
        )
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
                db.execute(
                    "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", result
                )
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
        for name in (
            *policy["source_annotations"].values(),
            policy["validation_subject_file"],
        ):
            path = local_path(root, name)
            bindings[str(path)] = file_sha256(path)
    report = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "data_root": str(root.resolve()),
        "inventory_sha256": file_sha256(inventory),
        "inventory_rows": len(records),
        "source_bindings": bindings,
        "fingerprints_sha256": file_sha256(database),
        "implementation_sha256": {
            p.relative_to(source_root()).as_posix(): file_sha256(p)
            for p in sorted(source_root().rglob("*.py"))
        },
    }
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
        with (output / "exact-groups.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["kind", "digest", "relative_path", "split", "source_type", "decision"]
            )
            for kind in ("file_hash", "pixel_hash", "crop_hash"):
                groups = db.execute(
                    f"SELECT {kind} FROM images WHERE exclusion='' AND {kind}!='' GROUP BY {kind} HAVING COUNT(*)>1"
                )
                for (digest,) in groups:
                    rows = db.execute(
                        f"SELECT path,split,source_type,label FROM images WHERE exclusion='' AND {kind}=? ORDER BY path",
                        (digest,),
                    ).fetchall()
                    if kind == "crop_hash":
                        crop_groups += 1
                        for path, split, code, _ in rows:
                            writer.writerow(
                                [kind, digest, path, split, code, "review_crop_context"]
                            )
                        continue
                    group_count += 1
                    conflict = len({(r[2], r[3]) for r in rows}) > 1
                    conflicts += conflict
                    keep = min(rows, key=lambda r: (priority[r[1]], r[0]))[0]
                    for path, split, code, _ in rows:
                        reason = (
                            "duplicate_label_conflict"
                            if conflict
                            else "exact_duplicate"
                            if path != keep
                            else ""
                        )
                        if reason:
                            decisions[path] = reason
                        writer.writerow(
                            [
                                kind,
                                digest,
                                path,
                                split,
                                code,
                                reason or "representative",
                            ]
                        )
        with (output / "selection.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["relative_path", "exclusion_reason"])
            writer.writerows(sorted(decisions.items()))
        remaining = {
            split: db.execute(
                "SELECT COUNT(*) FROM images WHERE split=? AND exclusion=''", (split,)
            ).fetchone()[0]
            for split in priority
        }
        for path in decisions:
            split = db.execute(
                "SELECT split FROM images WHERE path=?", (path,)
            ).fetchone()[0]
            remaining[split] -= 1
    report = {
        "exact_duplicate_groups_by_fingerprint": group_count,
        "label_conflict_groups_by_fingerprint": conflicts,
        "crop_only_review_groups": crop_groups,
        "additional_exclusions": len(decisions),
        "cleaned_subset_counts": remaining,
        "selection_sha256": file_sha256(output / "selection.csv"),
        "groups_sha256": file_sha256(output / "exact-groups.csv"),
        "policy": "Lexicographic representative; prefer test over validation over train; conflicts excluded pending review; original inventory/splits unchanged",
        "result": "PASS" if not conflicts else "UNVERIFIED",
    }
    write_json(output / "exact.json", report)
    return report


def complete_near_search(source, target):
    """Reuse verified fingerprints; preserve the prior scan and review records.

    Members plus signature edges represent all image pairs, including equal
    signatures. Neither equality nor an edge is a duplicate verdict. Keeping
    this graph compact avoids quadratic CSV expansion for dense signatures.
    This supplemental evidence does not authorize the training loader.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    scan = read_json(source / "scan.json")
    exact = read_json(source / "exact.json")
    database = source / "fingerprints.sqlite"
    if file_sha256(database) != scan["fingerprints_sha256"]:
        raise ValueError("Fingerprint index changed since scan")
    if file_sha256(source / "selection.csv") != exact["selection_sha256"]:
        raise ValueError("Exact selection changed since scan")
    for path, digest in scan["source_bindings"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Source changed since scan: {path}")
    with (source / "selection.csv").open(newline="", encoding="utf-8") as handle:
        excluded = {r["relative_path"] for r in csv.DictReader(handle)}
    if shutil.disk_usage(target.parent).free < 1024**3:
        raise ValueError("Need at least 1 GiB free for compact candidate graph")
    target.mkdir(exist_ok=False)
    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "result": "UNVERIFIED",
        "search_complete": False,
        "radius": 6,
        "source_scan": str(source / "scan.json"),
        "inventory_sha256": scan["inventory_sha256"],
        "input_bindings": {
            str(source / name): file_sha256(source / name)
            for name in (
                "scan.json",
                "exact.json",
                "selection.csv",
                "fingerprints.sqlite",
            )
        },
        "source_bindings": scan["source_bindings"],
        "views": {},
        "algorithm": "Exact four-block multi-index; all 16-bit blocks at Hamming distance <=1; full 64-bit verification; radius 6",
        "limitation": "Complete only within the fixed pHash radius for full and supplied face crops. Natural-pair recall, crop robustness, duplicate adjudication, source relationships and visual labels remain unverified.",
    }
    with (
        closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(target / "candidates.sqlite")) as db,
        db,
    ):
        db.execute(
            "CREATE TABLE members(view TEXT, signature TEXT, path TEXT, split TEXT, source_type INTEGER, label INTEGER, PRIMARY KEY(view,path))"
        )
        db.execute(
            "CREATE TABLE edges(view TEXT, left_hash TEXT, right_hash TEXT, hamming INTEGER, PRIMARY KEY(view,left_hash,right_hash))"
        )
        for view in ("full_phash", "crop_phash"):
            started = time.monotonic()
            rows = []
            for path, split, code, label, signature, error in original.execute(
                f"SELECT path,split,source_type,label,{view},error FROM images WHERE exclusion='' ORDER BY path"
            ):
                if path in excluded:
                    continue
                if error:
                    raise ValueError(f"Selected row has an input error: {path}")
                rows.append((view, signature, path, split, code, label))
            if any(not row[1] for row in rows):
                raise ValueError("Selected row missing a perceptual fingerprint")
            db.executemany("INSERT INTO members VALUES (?,?,?,?,?,?)", rows)
            counts = Counter(row[1] for row in rows)
            edges, implied = 0, sum(n * (n - 1) // 2 for n in counts.values())
            for a, b, distance in complete_hamming_pairs(counts):
                db.execute("INSERT INTO edges VALUES (?,?,?,?)", (view, a, b, distance))
                edges += 1
                implied += counts[a] * counts[b]
                if edges % 100000 == 0:
                    db.commit()
                    print(f"{view}: {edges} signature edges", flush=True)
            db.commit()
            result["views"][view] = {
                "selected_rows": len(rows),
                "distinct_signatures": len(counts),
                "signature_edges": edges,
                "equal_signature_groups": sum(n > 1 for n in counts.values()),
                "implied_image_pairs": implied,
                "seconds": time.monotonic() - started,
            }
            print(f"{view}: complete, {implied} image pairs", flush=True)
        db.execute("CREATE INDEX member_signature ON members(view,signature)")
        db.commit()
    result["search_complete"] = True
    result["graph_sha256"] = file_sha256(target / "candidates.sqlite")
    result["implementation_sha256"] = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__).resolve(),
            source_path("audit/image_fingerprints.py").resolve(),
        )
    }
    write_json(target / "near-complete.json", result)
    return result
