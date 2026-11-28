"""Audit the frozen post-review selection without changing data or model artifacts."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import csv
import json
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.liveness.audit.crop_content_scan import SPLIT_ANNOTATIONS  # noqa: E402
from facial_attendance.liveness.dataset_files import file_sha256  # noqa: E402
from facial_attendance.liveness.datasets.reviewed_cache import (  # noqa: E402 — configure the repository path before local imports.
    ReviewedCacheView,
    bound_file,
)
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages  # noqa: E402 — configure the repository path before local imports.


def release_file(project, release, name):
    record = release["files"][name]
    return bound_file(project, record["delivery_path"], record["sha256"])


def scan(view, workers):
    """Bound outstanding decodes so a full audit does not queue the whole split."""

    def check(index):
        image, _ = view[index]
        image.close()

    indices = iter(range(len(view)))
    checked = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers * 2):
            index = next(indices, None)
            if index is not None:
                pending.append(pool.submit(check, index))
        while pending:
            pending.popleft().result()
            checked += 1
            if checked % 10000 == 0:
                print(
                    f"{view.split}: verified {checked:,}/{len(view):,} retained crops",
                    flush=True,
                )
            index = next(indices, None)
            if index is not None:
                pending.append(pool.submit(check, index))
    return checked


def preview(view, output):
    # Cover every available source type; these examples do not clear unseen ROIs.
    selected = {}
    for index, row in enumerate(view.samples):
        selected.setdefault(row.source_type, index)
    examples = []
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3.3), squeeze=False)
    try:
        for axis in axes.flat:
            axis.axis("off")
        for axis, (source_type, index) in zip(axes.flat, sorted(selected.items())):
            row = view.samples[index]
            image, target = view[index]
            resized = image.resize((224, 224), Image.Resampling.BILINEAR)
            axis.imshow(resized)
            axis.set_title(
                f"{'Live' if row.binary_label == 0 else 'Spoof'} · source type {source_type}\n{row.relative_path}",
                fontsize=8,
                wrap=True,
            )
            examples.append(
                {
                    "relative_path": row.relative_path,
                    "base_index": view.base_index(index),
                    "binary_label": row.binary_label,
                    "source_type": source_type,
                    "source_sha256": row.sha256,
                    "bbox_sha256": row.bbox_sha256,
                }
            )
            if target["binary"] != row.binary_label:
                raise ValueError("Preview target is misaligned")
            resized.close()
            image.close()
        fig.suptitle(
            f"{view.split.title()}: retained source crops after confirmed exclusions"
        )
        fig.tight_layout()
        fig.savefig(output / f"{view.split}.png", dpi=120)
    finally:
        plt.close(fig)
    return examples


def run(project, output, *, inspect=False, workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    project = project.resolve()
    release = json.loads((project / "configs/liveness-release.json").read_text())
    view_path = release_file(project, release, release["crop_review"]["cache_view"])
    frozen = json.loads(view_path.read_text())
    quarantine = bound_file(
        view_path.parent, "quarantine.csv", frozen["quarantine_sha256"]
    )
    bound_file(view_path.parent, "pending-review.csv", frozen["pending_review_sha256"])
    with quarantine.open(encoding="utf-8", newline="") as stream:
        exclusions = list(csv.DictReader(stream))
    excluded = {row["relative_path"]: row for row in exclusions}
    if len(excluded) != len(exclusions):
        raise ValueError("Duplicate quarantine identities")
    root = project / "data/raw/liveness/celeba-spoof-full"
    prefix = root.relative_to(project).as_posix() + "/"
    inventory = release_file(
        project, release, prefix + "audits/current-schema-2026-11-12/inventory.csv"
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "liveness-reviewed-selection-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "inspect" if inspect else "audit",
        "completed": False,
        "cache_view_sha256": file_sha256(view_path),
        "quarantine_sha256": file_sha256(quarantine),
        "inventory_sha256": file_sha256(inventory),
        "splits": {},
        "limits": [
            "Preview examples do not establish semantic correctness of unseen crops or labels.",
            "Source-video independence remains unverified due to unavailable source-video IDs.",
            "This run does not train, evaluate or modify checkpoints, frozen exclusions or split assignments.",
            "Inspect checks metadata and preview inputs only; audit additionally checks every retained image and box.",
        ],
    }
    seen = set()
    subjects = {}
    matched_exclusions = set()
    try:
        for split, name in SPLIT_ANNOTATIONS.items():
            annotation = release_file(project, release, prefix + name)
            source = ReproductionImages(root, inventory, annotation)
            view = ReviewedCacheView(source, view_path.parent, split)
            # Check the actual complement, not just its size: equal-size substitutions
            # would silently reinstate an excluded crop and remove a valid one.
            keep = set(map(int, view.indices))
            removed = {
                row.relative_path: row
                for i, row in enumerate(source.samples)
                if i not in keep
            }
            expected = {name for name, row in excluded.items() if row["split"] == split}
            if (
                set(removed) != expected
                or len(removed) != frozen["splits"][split]["quarantined_samples"]
            ):
                raise ValueError(
                    "Retained rows disagree with the frozen quarantine complement"
                )
            for name, row in removed.items():
                decision = excluded[name]
                if (
                    decision["decision"] != "quarantine"
                    or int(decision["binary_label"]) != row.binary_label
                    or int(decision["source_type"]) != row.source_type
                    or decision["source_sha256"] != row.sha256
                    or decision["bbox_sha256"] != row.bbox_sha256
                ):
                    raise ValueError("Quarantine labels or source bindings disagree")
            matched_exclusions.update(expected)
            names = {row.relative_path for row in source.samples}
            if seen & names:
                raise ValueError("Images occur in multiple frozen splits")
            seen.update(names)
            subjects[split] = {row.subject_id for row in source.samples}
            if split == "validation" and subjects["train"] & subjects[split]:
                raise ValueError("Frozen train/validation subjects overlap")
            checked = 0 if inspect else scan(view, workers)
            examples = preview(view, output)
            report["splits"][split] = {
                "annotations_sha256": file_sha256(annotation),
                "base_samples": len(source),
                "retained_samples": len(view),
                "quarantined_samples": len(removed),
                "pending_review": frozen["splits"][split]["pending_review"],
                "integrity_checked_rows": checked,
                "preview_checked_rows": len(examples),
                "counts_by_binary_label": dict(
                    Counter(row.binary_label for row in view.samples)
                ),
                "counts_by_source_type": dict(
                    Counter(row.source_type for row in view.samples)
                ),
                "examples": examples,
            }
            print(
                f"{split}: {len(view):,} retained; {len(removed):,} excluded",
                flush=True,
            )
            del view, source
        if matched_exclusions != set(excluded):
            raise ValueError("Quarantine contains unknown split assignments")
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Check frozen metadata and preview inputs without a full image scan",
    )
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    run(args.project, args.output, inspect=args.inspect, workers=args.workers)


if __name__ == "__main__":
    main()
