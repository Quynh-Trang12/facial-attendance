"""Audit every face image for label, duplicate and leakage problems; never train here."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.config import load_paths
from facial_attendance.recognition.audit import audit_dataset
from facial_attendance.recognition.data_loader import FaceLayout
from facial_attendance.recognition.download import COMPETITION, download_face_data
from facial_attendance.recognition.exclusions import DEFAULT_EXCLUSIONS, apply_exclusions
from facial_attendance.recognition.sampling import (
    IDENTITIES_PER_BATCH,
    IMAGES_PER_IDENTITY,
)

PROGRESS_EVERY = 20_000


def parse_arguments(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--download",
        action="store_true",
        help="Use the assigned Kaggle competition API",
    )
    inputs.add_argument(
        "--data-root", type=Path, help="Override the configured dataset location"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/project.toml")
    parser.add_argument(
        "--cache-dir", type=Path, help="Override the configured Kaggle cache"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the manual spot-check sample"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help="Images the loader drops; the audit checks what is left",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Audit the data as published, unfiltered"
    )
    parser.add_argument(
        "--write-exclusions",
        type=Path,
        help="With --raw, generate an exclusion list from this audit's hashes",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Write one CSV row per image with its hash and keep/exclude decision",
    )
    parser.add_argument(
        "--triplet-batch",
        type=int,
        nargs=2,
        metavar=("P", "K"),
        default=(IDENTITIES_PER_BATCH, IMAGES_PER_IDENTITY),
        help="Identities per batch and images per identity for the P×K check",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output file; earlier audit reports are evidence")
    if args.inventory is not None and args.inventory.exists():
        parser.error("Choose a new inventory file")
    if args.write_exclusions is not None and (
        not args.raw or args.write_exclusions.exists()
    ):
        parser.error("--write-exclusions needs --raw and a new file")
    return args


def resolve_layout(args):
    """All default storage locations come from one config, not duplicated literals."""
    paths = load_paths(args.config, root=ROOT)
    if args.download:
        layout = download_face_data(cache_dir=args.cache_dir or paths.kaggle_cache)
    else:
        layout = FaceLayout.discover(args.data_root or paths.face_data)
    if not args.raw:
        layout = apply_exclusions(layout, args.exclusions)
    return layout, paths


def display(path: Path) -> str:
    # Repository-relative where possible, so reports carry no personal home path.
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def show_progress(done: int, total: int):
    if done % PROGRESS_EVERY == 0 or done == total:
        print(f"audited {done:,}/{total:,} files", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        kagglehub_version = version("kagglehub")
    except PackageNotFoundError:
        # Local audits never call Kaggle, so a missing client must not lose the report.
        kagglehub_version = None
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FAC-02",
        "status": "FAIL",
        "competition": COMPETITION,
        "source": "kaggle_api" if args.download else "local_files",
        "kagglehub_version": kagglehub_version,
    }
    code = 1
    try:
        layout, paths = resolve_layout(args)
        report["configuration"] = {
            "file": display(args.config),
            "face_data": display(paths.face_data),
            "kaggle_cache": display(paths.kaggle_cache),
        }
        report["dataset_root"] = display(layout.root)
        report.update(
            audit_dataset(
                layout,
                seed=args.seed,
                progress=show_progress,
                write_exclusions=args.write_exclusions,
                inventory=args.inventory,
                triplet_batch=tuple(args.triplet_batch),
            )
        )
        code = 0 if report["status"] == "PASS" else 1
    except Exception as error:
        report.update(
            status="FAIL",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )

    # Preserve both successful and failed observations; never relabel an old run.
    files = [
        Path(__file__).resolve(),
        ROOT / "src/facial_attendance/config.py",
        ROOT / "src/facial_attendance/recognition/audit.py",
        ROOT / "src/facial_attendance/recognition/data_loader.py",
        ROOT / "src/facial_attendance/recognition/download.py",
        ROOT / "src/facial_attendance/recognition/exclusions.py",
        ROOT / "src/facial_attendance/recognition/sampling.py",
        args.config.resolve(),
        (args.write_exclusions or args.exclusions).resolve(),
    ]
    report["source_sha256"] = {
        display(p): sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    # The full report holds capped example lists; the console needs only the verdict.
    summary = {
        k: report[k] for k in ("status", "failures", "warnings", "error") if k in report
    }
    print(json.dumps(summary | {"report": str(args.output)}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
