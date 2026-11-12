"""Inspect configured/API face data in readable stages; never train a model here."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.config import load_paths
from facial_attendance.recognition.data_loader import (
    ClassificationFaces,
    FaceLayout,
    PairFaces,
    VerificationPairs,
    image_transform,
    sample_training_pairs,
)
from facial_attendance.recognition.download import COMPETITION, download_face_data
from facial_attendance.recognition.exclusions import DEFAULT_EXCLUSIONS, apply_exclusions
from facial_attendance.recognition.sampling import (
    IDENTITIES_PER_BATCH,
    IMAGES_PER_IDENTITY,
    IdentityBatchSampler,
)


def first_batch(dataset, batch_size: int):
    from torch.utils.data import DataLoader

    # Preserve official trial order. Zero workers also avoids notebook/Windows
    # process-spawning surprises; images are decoded on CPU before model transfer.
    return next(
        iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))
    )


def inspect_classification(splits: dict, *, batch_size: int, image_size: int) -> dict:
    """Check one actual decoded batch from each split, keeping their roles separate."""
    import torch

    observations = {}
    for split, dataset in splits.items():
        images, targets = first_batch(dataset, batch_size)
        valid_targets = ((targets >= 0) & (targets < len(dataset.class_to_idx))).all()
        if (
            images.shape[1:] != (3, image_size, image_size)
            or images.dtype != torch.float32
            or not torch.isfinite(images).all()
            or targets.dtype != torch.int64
            or not valid_targets
        ):
            raise RuntimeError(f"Invalid classification batch in {split}")
        observations[split] = {
            "images": len(dataset),
            "images_dropped_by_exclusions": dataset.excluded_count,
            "identities_present": len({s.identity for s in dataset.samples}),
            "batch_shape": list(images.shape),
            "targets": targets.tolist(),
            "dtype": str(images.dtype),
            "finite": True,
        }
    return observations


def inspect_pair_batch(dataset, *, batch_size: int, image_size: int) -> dict:
    """Use the same tensor/label contract for official and sampled pairs."""
    import torch

    left, right, labels = first_batch(dataset, batch_size)
    if (
        left.shape != right.shape
        or left.shape[1:] != (3, image_size, image_size)
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or not torch.isfinite(left).all()
        or not torch.isfinite(right).all()
        or labels.dtype != torch.int64
        or not set(labels.tolist()) <= {0, 1}
    ):
        raise RuntimeError("Invalid pair batch")
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "labels": labels.tolist(),
    }


def inspect_sampling(
    train, *, transform, batch_size: int, image_size: int, seed: int
) -> tuple:
    """Validate generated labels against training identities, not model predictions."""
    sampled = sample_training_pairs(train, count_per_label=3, seed=seed)
    identities = {sample.path: sample.identity for sample in train.samples}
    for pair in sampled:
        same_identity = identities[pair.left] == identities[pair.right]
        if same_identity != bool(pair.label) or pair.left == pair.right:
            raise RuntimeError("Sampled pair violates its identity label")
    # Repeating a seeded draw should not depend on another module's global RNG.
    if sampled != sample_training_pairs(train, count_per_label=3, seed=seed):
        raise RuntimeError("Pair sampling is not reproducible")
    batch = inspect_pair_batch(
        PairFaces(sampled, transform=transform),
        batch_size=batch_size,
        image_size=image_size,
    )
    return sampled, batch


def inspect_triplet_batch(train, *, image_size: int, seed: int, shape) -> dict:
    """Decode the first real P×K batch and check identity relationships in it."""
    import torch
    from torch.utils.data import DataLoader

    identities_per_batch, images_per_identity = shape
    sampler = IdentityBatchSampler(
        train,
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        seed=seed,
    )
    indices = next(iter(sampler))
    images, targets = next(iter(DataLoader(train, batch_sampler=sampler, num_workers=0)))
    samples = [train.samples[i] for i in indices]
    per_target = Counter(targets.tolist())
    if (
        tuple(images.shape) != (sampler.batch_size, 3, image_size, image_size)
        or images.dtype != torch.float32
        or not torch.isfinite(images).all()
        # Same seed and epoch, so the loader decoded exactly these indices.
        or targets.tolist() != [s.target for s in samples]
        or len(per_target) != identities_per_batch
        or set(per_target.values()) != {images_per_identity}
        # Distinct files: every anchor–positive pair compares two different images.
        or len({s.path for s in samples}) != len(samples)
        or any(s.path in train.layout.excluded for s in samples)
    ):
        raise RuntimeError("Invalid P×K triplet batch")
    return {
        "seed": seed,
        "identities_per_batch": identities_per_batch,
        "images_per_identity": images_per_identity,
        "batches_per_epoch": len(sampler),
        "batch_shape": list(images.shape),
        "identities": sorted({s.identity for s in samples})[:5],
        "first_identity_paths": [
            str(s.path.relative_to(train.layout.root))
            for s in samples[:images_per_identity]
        ],
    }


def pair_preview(pairs, root: Path) -> list[dict]:
    # Relative paths make reviewer reports comparable across machines/cache roots.
    return [
        {
            "left": str(p.left.relative_to(root)),
            "right": str(p.right.relative_to(root)),
            "label": p.label,
        }
        for p in pairs
    ]


def inspect_data(
    layout: FaceLayout,
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    triplet_batch=(IDENTITIES_PER_BATCH, IMAGES_PER_IDENTITY),
) -> dict:
    """Orchestrate indexing, batch decoding and sampling as distinct checks."""
    transform = image_transform(image_size)
    train = ClassificationFaces(layout, "train_data", transform=transform)
    # Learn identity-to-target encoding once from train. Re-fitting it per split
    # could assign different integer labels to the same person.
    splits = {"train_data": train}
    for split in ("val_data", "test_data"):
        splits[split] = ClassificationFaces(
            layout, split, class_to_idx=train.class_to_idx, transform=transform
        )
    trials = VerificationPairs(layout, transform=transform)

    classification = inspect_classification(
        splits, batch_size=batch_size, image_size=image_size
    )
    verification = inspect_pair_batch(
        trials, batch_size=batch_size, image_size=image_size
    )
    sampled, training_batch = inspect_sampling(
        train,
        transform=transform,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
    )
    triplets = inspect_triplet_batch(
        train, image_size=image_size, seed=seed, shape=triplet_batch
    )
    return {
        "dataset_root": str(layout.root),
        "images_excluded_by_list": len(layout.excluded),
        "data_loading_device": "cpu",
        "class_count": len(train.class_to_idx),
        "identity_mapping_sha256": sha256(
            json.dumps(train.class_to_idx, sort_keys=True).encode()
        ).hexdigest(),
        "identity_mapping_preview": dict(list(train.class_to_idx.items())[:5]),
        "classification": classification,
        "verification": {
            "trials": len(trials),
            "label_counts": dict(Counter(p.label for p in trials.pairs)),
            "left_batch_shape": verification["left_shape"],
            "right_batch_shape": verification["right_shape"],
            "batch_labels": verification["labels"],
            "pairs_file_sha256": sha256(layout.pairs_file.read_bytes()).hexdigest(),
            "trial_preview": pair_preview(trials.pairs[:3], layout.root),
        },
        "sampled_training_pairs": pair_preview(sampled, layout.root),
        "sampling_seed": seed,
        "training_pair_batch": training_batch,
        "triplet_batch": triplets,
        "checks": {
            "shared_identity_mapping": True,
            "classification_batches": True,
            "verification_paths_and_labels": True,
            "verification_batch": True,
            "sampling_identity_rules": True,
            "sampling_reproducible": True,
            "sampled_pair_batch": True,
            "triplet_batch_identity_rules": True,
        },
        "limits": "First decoded batches only; the full image, duplicate and leakage audit is scripts/audit_face_data.py; no model evaluation.",
    }


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Inspection resize, not final model preprocessing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help="Images the loader drops before any check",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Check the data as published, unfiltered"
    )
    parser.add_argument(
        "--triplet-batch",
        type=int,
        nargs=2,
        metavar=("P", "K"),
        default=(IDENTITIES_PER_BATCH, IMAGES_PER_IDENTITY),
        help="Identities per batch and images per identity",
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.batch_size < 1 or args.image_size < 1:
        parser.error("Choose a new output file and positive batch/image sizes")
    return args


def resolve_layout(args):
    """All default storage locations come from one config, not duplicated literals."""
    paths = load_paths(args.config, root=ROOT)
    if args.download:
        layout = download_face_data(cache_dir=args.cache_dir or paths.kaggle_cache)
    else:
        layout = FaceLayout.discover(args.data_root or paths.face_data)
    return layout, paths


def select_images(layout: FaceLayout, args) -> FaceLayout:
    """Training checks see the filtered data unless the raw data is asked for by name."""
    return layout if args.raw else apply_exclusions(layout, args.exclusions)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FAC-01.2",
        "status": "FAIL",
        "competition": COMPETITION,
        "source": "kaggle_api" if args.download else "local_files",
        "kagglehub_version": version("kagglehub"),
    }
    code = 1
    try:
        layout, paths = resolve_layout(args)
        layout = select_images(layout, args)
        report["configuration"] = {
            "file": str(args.config.resolve()),
            "face_data": str(paths.face_data),
            "kaggle_cache": str(paths.kaggle_cache),
        }
        report.update(
            inspect_data(
                layout,
                batch_size=args.batch_size,
                image_size=args.image_size,
                seed=args.seed,
                triplet_batch=tuple(args.triplet_batch),
            )
        )
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )

    # Preserve both successful and failed observations; never relabel an old run.
    files = [
        Path(__file__).resolve(),
        ROOT / "src/facial_attendance/config.py",
        ROOT / "src/facial_attendance/recognition/data_loader.py",
        ROOT / "src/facial_attendance/recognition/download.py",
        ROOT / "src/facial_attendance/recognition/exclusions.py",
        ROOT / "src/facial_attendance/recognition/sampling.py",
        args.config.resolve(),
        *([] if args.raw else [args.exclusions.resolve()]),
    ]
    report["source_sha256"] = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(
            p.read_bytes()
        ).hexdigest()
        for p in files
        if p.is_file()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
