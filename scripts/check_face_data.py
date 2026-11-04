"""Download/load the assigned face data and record actual classification/pair batches."""

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
    ClassificationFaces, FaceLayout, PairFaces, VerificationPairs, image_transform, sample_training_pairs,
)
from facial_attendance.recognition.download import COMPETITION, download_face_data


def inspect_data(layout: FaceLayout, *, batch_size: int, image_size: int, seed: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    transform = image_transform(image_size)
    train = ClassificationFaces(layout, "train_data", transform=transform)
    splits = {"train_data": train}
    for split in ("val_data", "test_data"):
        splits[split] = ClassificationFaces(layout, split, class_to_idx=train.class_to_idx, transform=transform)
    trials = VerificationPairs(layout, transform=transform)
    classification = {}
    for split, dataset in splits.items():
        images, targets = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
        if (images.shape[1:] != (3, image_size, image_size) or not torch.isfinite(images).all()
                or images.dtype != torch.float32 or targets.dtype != torch.int64):
            raise RuntimeError(f"Invalid classification batch in {split}")
        classification[split] = {
            "images": len(dataset), "identities_present": len({s.identity for s in dataset.samples}),
            "batch_shape": list(images.shape), "targets": targets.tolist(),
            "dtype": str(images.dtype), "finite": True,
        }
    left, right, labels = next(iter(DataLoader(trials, batch_size=batch_size, shuffle=False, num_workers=0)))
    if (left.shape != right.shape or left.shape[1:] != (3, image_size, image_size)
            or not torch.isfinite(left).all() or not torch.isfinite(right).all()
            or not set(labels.tolist()) <= {0, 1}):
        raise RuntimeError("Invalid verification batch")
    sampled = sample_training_pairs(train, count_per_label=3, seed=seed)
    path_to_identity = {sample.path: sample.identity for sample in train.samples}
    for pair in sampled:
        if ((path_to_identity[pair.left] == path_to_identity[pair.right]) != bool(pair.label)
                or pair.left == pair.right):
            raise RuntimeError("Sampled pair violates its identity label")
    if sampled != sample_training_pairs(train, count_per_label=3, seed=seed):
        raise RuntimeError("Pair sampling is not reproducible")
    sampled_loader = DataLoader(PairFaces(sampled, transform=transform), batch_size=batch_size,
                                shuffle=False, num_workers=0)
    train_left, train_right, train_labels = next(iter(sampled_loader))
    if (train_left.shape != train_right.shape or train_left.shape[1:] != (3, image_size, image_size)
            or not torch.isfinite(train_left).all() or not torch.isfinite(train_right).all()
            or train_labels.dtype != torch.int64):
        raise RuntimeError("Invalid sampled training-pair batch")
    return {
        "dataset_root": str(layout.root), "class_count": len(train.class_to_idx),
        "identity_mapping_sha256": sha256(json.dumps(train.class_to_idx, sort_keys=True).encode()).hexdigest(),
        "identity_mapping_preview": dict(list(train.class_to_idx.items())[:5]),
        "classification": classification,
        "verification": {
            "trials": len(trials), "label_counts": dict(Counter(pair.label for pair in trials.pairs)),
            "left_batch_shape": list(left.shape), "right_batch_shape": list(right.shape),
            "batch_labels": labels.tolist(),
            "pairs_file_sha256": sha256(layout.pairs_file.read_bytes()).hexdigest(),
            "trial_preview": [
                {"left": str(p.left.relative_to(layout.root)), "right": str(p.right.relative_to(layout.root)),
                 "label": p.label} for p in trials.pairs[:3]
            ],
        },
        "sampled_training_pairs": [
            {"left": str(p.left.relative_to(layout.root)), "right": str(p.right.relative_to(layout.root)),
             "label": p.label} for p in sampled
        ],
        "sampling_seed": seed,
        "training_pair_batch": {"left_shape": list(train_left.shape), "right_shape": list(train_right.shape),
                                "labels": train_labels.tolist()},
        "checks": {"shared_identity_mapping": True, "classification_batches": True,
                   "verification_paths_and_labels": True, "verification_batch": True,
                   "sampling_identity_rules": True, "sampling_reproducible": True,
                   "sampled_pair_batch": True},
        "limits": "Decoded first batch per split and verification; not a full image/duplicate/leakage audit or model evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--download", action="store_true", help="Use kagglehub.competition_download")
    inputs.add_argument("--data-root", type=Path, help="Existing extracted course dataset")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.kagglehub")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=112, help="Inspection resize, not final model preprocessing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or args.image_size < 1:
        parser.error("Choose a new output file and positive batch/image sizes")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "task": "FAC-01.2", "status": "FAIL",
        "competition": COMPETITION, "source": "kaggle_api" if args.download else "local_files",
        "kagglehub_version": version("kagglehub"),
    }
    code = 1
    try:
        if args.download:
            layout = download_face_data(cache_dir=args.cache_dir)
        else:
            data_root = args.data_root or load_paths(ROOT / "configs/project.toml", root=ROOT).face_data
            layout = FaceLayout.discover(data_root)
        report.update(inspect_data(layout, batch_size=args.batch_size, image_size=args.image_size, seed=args.seed))
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), ROOT / "src/facial_attendance/recognition/data_loader.py",
                  ROOT / "src/facial_attendance/recognition/download.py"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
