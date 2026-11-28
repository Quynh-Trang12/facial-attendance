import csv
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "sad-happy-neutral-geometry-v2-20261121"
ZIP_SHA = "f94320ce10a3403722d4b693c0ca0f32786ad41f0683a31bc1a619790e3cd039"
MANIFEST_SHA = "54ac9d2ea037dbb2d879acd9d960f11161bfc353b89f5ca8a0151f221d2ac4ec"
COUNTS = {
    "train": {"sad": 3581, "happy": 5541, "neutral": 3781},
    "val": {"sad": 890, "happy": 1384, "neutral": 943},
    "test": {"sad": 1186, "happy": 1749, "neutral": 1198},
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def validate_dataset(root):
    root = Path(root)
    assert sha_file(root / "manifest.csv") == MANIFEST_SHA, "Dataset manifest differs."
    info = json.loads((root / "dataset-info.json").read_text(encoding="utf-8"))
    assert info["version"] == VERSION and info["model_ids"] == {
        "sad": 0,
        "happy": 1,
        "neutral": 2,
    }
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    retained = [r for r in rows if r["status"] == "retained"]
    assert len(rows) == 20716 and len(retained) == 20253
    assert len({r["pixel_sha256"] for r in retained}) == 20253, "Exact content overlap."
    actual = {
        s: {
            label: sum(
                r["output_split"] == s and r["original_label"] == label
                for r in retained
            )
            for label in ["sad", "happy", "neutral"]
        }
        for s in COUNTS
    }
    assert actual == COUNTS and info["retained_counts"] == COUNTS
    assert {p.relative_to(root).as_posix() for p in root.rglob("*.jpg")} == {
        r["output_path"] for r in retained
    }, "Image inventory differs."

    def verify(r):
        relative = Path(r["output_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert (
            int(r["model_id"])
            == {"sad": 0, "happy": 1, "neutral": 2}[r["original_label"]]
        )
        assert sha_file(root / relative) == r["file_sha256"], r["output_path"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, retained))
    print(
        "PASS: 20,253 files, labels, split counts and hashes verified. Test files checked as bytes only; no test evaluation."
    )
    return info


def prepare_dataset(drive_zip, cache_root):
    drive_zip = Path(drive_zip)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    assert drive_zip.is_file(), (
        f"ZIP not found: {drive_zip}. Check the mounted Google account and Drive folder."
    )
    archive = cache_root / (VERSION + ".zip")
    if not archive.exists() or sha_file(archive) != ZIP_SHA:
        shutil.copyfile(drive_zip, archive)
    assert sha_file(archive) == ZIP_SHA, (
        "ZIP checksum differs; use the approved cleaned archive."
    )
    destination = cache_root / VERSION
    if not destination.exists():
        staging = cache_root / (VERSION + "-extracting")
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                relative = Path(item.filename)
                assert not relative.is_absolute() and ".." not in relative.parts, (
                    "Unsafe ZIP path."
                )
            assert z.testzip() is None, "Damaged ZIP."
            z.extractall(staging)
        validate_dataset(staging)
        staging.rename(destination)
    else:
        validate_dataset(destination)
    return destination
