"""Audit an acquired CelebA-Spoof dataset and save traceable sample batches."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw

from facial_attendance.config import load_paths
from facial_attendance.liveness.dataset_audit import audit_dataset
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import bbox_path, read_face
from facial_attendance.liveness.image_inventory import write_inventory
from facial_attendance.liveness.data_contract import SPLITS
from facial_attendance.liveness.audit_report import (
    AuditReport,
    BatchEvidence,
    write_summary,
)
from facial_attendance.recognition.data_loader import image_transform


def sample_batch(
    dataset, output: Path, batch_size: int, image_size: int
) -> BatchEvidence:
    # This boundary keeps CLI help and annotation failures usable without Torch,
    # matching the project's Pillow-only source-check job.
    import torch
    from torch.utils.data import DataLoader, Subset

    # Include each selected source type that exists, then fill in inventory order.
    indices, types = [], set()
    for index, row in enumerate(dataset.samples):
        if row.source_type not in types:
            indices.append(index)
            types.add(row.source_type)
    if batch_size < len(indices):
        raise ValueError(f"batch-size must cover {len(indices)} source types")
    indices += [i for i in range(len(dataset)) if i not in indices][
        : batch_size - len(indices)
    ]

    images, targets = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
        )
    )

    if (
        images.shape != (len(indices), 3, image_size, image_size)
        or images.dtype != torch.float32
        or targets.dtype != torch.int64
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError(f"Invalid batch in {dataset.split}")

    records = []
    sheet = Image.new("RGB", (image_size * len(indices), image_size + 64), "white")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        row = dataset.samples[index]
        face = read_face(
            local_path(dataset.root, row.relative_path),
            box_file=local_path(dataset.root, str(bbox_path(Path(row.relative_path)))),
        )
        expected = dataset.transform(face)
        if int(targets[position]) != row.binary_label or not torch.equal(
            images[position], expected
        ):
            raise ValueError(f"Image/target alignment failed: {row.relative_path}")

        sheet.paste(
            face.resize((image_size, image_size), Image.Resampling.BILINEAR),
            (position * image_size, 0),
        )
        draw.text(
            (position * image_size + 3, image_size + 3),
            f"index={index} target={row.binary_label}\n{row.split} type={row.source_type}\n{row.attack_family}",
            fill="black",
        )
        records.append({"dataset_index": index, **asdict(row)})

    stem = output / dataset.split
    sheet.save(stem.with_suffix(".png"))
    torch.save(
        {
            "images": images,
            "targets": targets,
            "relative_paths": [r["relative_path"] for r in records],
        },
        stem.with_suffix(".pt"),
    )

    saved = torch.load(stem.with_suffix(".pt"), weights_only=True)
    if (
        not torch.equal(saved["images"], images)
        or not torch.equal(saved["targets"], targets)
        or saved["relative_paths"] != [r["relative_path"] for r in records]
    ):
        raise ValueError(f"Saved batch readback failed: {dataset.split}")

    return BatchEvidence(
        saved_batch_readback="PASS",
        shape=list(images.shape),
        image_dtype=str(images.dtype),
        target_dtype=str(targets.dtype),
        range=[images.min().item(), images.max().item()],
        alignment="PASS",
        rows=records,
        tensor_sha256=file_sha256(stem.with_suffix(".pt")),
        preview_sha256=file_sha256(stem.with_suffix(".png")),
    )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/project.toml")
    parser.add_argument("--data-root", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--summarize-audit",
        type=Path,
        help="Export a summary from an existing audit.json without rerunning the audit",
    )
    inputs.add_argument(
        "--data-description",
        type=Path,
        help="Source details and project split assignment JSON; see docs/liveness-loader.md",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        help="New summary file; defaults to summary.json beside audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit new output directory; overrides the route-based default",
    )

    parser.add_argument(
        "--execution-route",
        choices=("local", "colab"),
        default="local",
        help="Execution route for default evidence/runs/LIV-03/<route>/<UTC timestamp> output",
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU image audit workers; inventory order is unchanged",
    )

    return parser.parse_args(argv)


def audit_output_directory(args):
    """Keep implicit outputs separated by execution route; honor explicit destinations."""
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (ROOT / "evidence/runs/LIV-03" / args.execution_route / run_name).resolve()


def main(argv=None):
    args = parse_arguments(argv)
    if args.summarize_audit:
        write_summary(
            args.summarize_audit,
            args.summary_output or args.summarize_audit.parent / "summary.json",
        )
        return 0

    if args.image_size < 1 or args.batch_size < 1 or args.workers < 1:
        raise ValueError("image-size, batch-size and workers must be positive")

    root = (
        (args.data_root or load_paths(args.config, root=ROOT).liveness_data)
        .expanduser()
        .resolve()
    )
    output = audit_output_directory(args)

    # Previews, tensors and per-image records stay outside tracked evidence.
    # Do not overwrite evidence from an earlier run.
    summary_output = args.summary_output or output / "summary.json"
    if summary_output.exists():
        raise FileExistsError(summary_output)
    if args.summary_output and not summary_output.parent.is_dir():
        raise FileNotFoundError(summary_output.parent)
    output.mkdir(parents=True, exist_ok=False)

    report = AuditReport.start(root, args.image_size, args.workers)

    try:
        report.record_extraction()
        records, audit = audit_dataset(
            root,
            args.data_description,
            workers=args.workers,
            progress=lambda n, split, errors: print(
                f"Audited {n} images; split={split}; image errors={errors}", flush=True
            ),
        )
        report.audit = audit
        if not audit["accepted"]:
            write_inventory(output / "rejected-inventory.csv", records)
            raise ValueError(
                "Data audit failed; inspect overlaps, missing_families and image_errors"
            )

        # A candidate becomes a loadable accepted inventory only after batch checks.
        inventory = output / "candidate-inventory.csv"
        report.inventory_sha256 = write_inventory(inventory, records)

        transform = image_transform(args.image_size)
        report.batches = {}
        for split in SPLITS:
            dataset = LivenessImages(root, inventory, split, transform=transform)
            report.batches[split] = sample_batch(
                dataset, output, args.batch_size, args.image_size
            )

        inventory.rename(output / "inventory.csv")
        report.complete()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        report.fail(error)

    report.save(output, summary_output, Path(__file__))

    print(f"{report.result}: {output / 'audit.json'}")
    if report.error:
        print(report.error, file=sys.stderr)
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
