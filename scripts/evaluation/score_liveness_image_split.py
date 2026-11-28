"""Reload a checkpoint and score the selected dataset split with its saved threshold.

Write per-image predictions and aggregate error metrics. Full-source test runs
also write descriptive benchmark threshold sweeps; none modifies the checkpoint.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import os

import numpy as np
import torch

# A historical checkpoint must use its verified producing modules before any
# liveness imports. Otherwise a corrected crop would silently change its test.
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--source-snapshot", type=Path)
bootstrap.add_argument("--checkpoint", type=Path)
source_args, _ = bootstrap.parse_known_args()
if source_args.source_snapshot is not None:
    if source_args.checkpoint is None:
        bootstrap.error("--source-snapshot requires --checkpoint")
    checkpoint_metadata = torch.load(
        source_args.checkpoint, map_location="cpu", weights_only=True
    )["meta"]
    bindings = checkpoint_metadata.get("source_sha256", {})
    if not bindings:
        raise ValueError("Checkpoint has no producing source hashes")
    for name, digest in bindings.items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or ":" in name
            or not name.endswith(".py")
        ):
            raise ValueError("Invalid producing source filename")
        producing = (source_args.source_snapshot / name).resolve()
        if not producing.is_relative_to(source_args.source_snapshot.resolve()):
            raise ValueError("Producing source path escapes its snapshot")
        with producing.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Producing source checksum mismatch: " + name)
    import facial_attendance.liveness

    facial_attendance.liveness.__path__ = [str(source_args.source_snapshot.resolve())]

from liveness_training_health import (  # noqa: E402 — configure the repository first.
    InferenceProgress,
    runtime_loader_workers,
)

if source_args.source_snapshot is not None:
    # Resolve each component from its verified producing layout. This supports
    # both historical flat trees and earlier package layouts without mixing code.
    from importlib import import_module

    def producing(*candidates):
        for relative in candidates:
            if (source_args.source_snapshot / (relative + ".py")).is_file():
                return import_module(
                    "facial_attendance.liveness." + relative.replace("/", ".")
                )
        raise ValueError(
            "Producing snapshot is missing a required component: " + candidates[0]
        )

    model_module = producing("models/binary_model", "binary_model")
    config_module = producing("experiments/binary_experiment", "binary_training")
    prediction_module = producing("training/validation", "binary_training")
    source_module = producing(
        "datasets/source_dataset", "source_dataset", "reproduction_dataset"
    )
    selected_module = producing(
        "datasets/selected_dataset", "selected_dataset", "data_loader"
    )
    metrics_module = producing("evaluation/evaluation_metrics", "evaluation_metrics")
    files_module = producing("dataset_files")
    face_transform, load_checkpoint = (
        model_module.face_transform,
        model_module.load_checkpoint,
    )
    TrainingConfig, predict_dataset = (
        config_module.TrainingConfig,
        prediction_module.predict_dataset,
    )
    save_predictions = prediction_module.save_predictions
    write_json = (
        files_module.write_json
        if hasattr(files_module, "write_json")
        else producing("binary_training").write_json
    )
    file_sha256 = files_module.file_sha256
    LivenessImages = selected_module.LivenessImages
    BinaryReproductionImages, ReproductionImages = (
        source_module.BinaryReproductionImages,
        source_module.ReproductionImages,
    )
    binary_metrics, family_metrics = (
        metrics_module.binary_metrics,
        metrics_module.family_metrics,
    )
    published_benchmark_metrics = metrics_module.published_benchmark_metrics
else:
    from facial_attendance.liveness.dataset_files import (
        file_sha256,  # noqa: E402 — repository path is configured before local imports.
        write_json,
    )
    from facial_attendance.liveness.datasets.selected_dataset import LivenessImages
    from facial_attendance.liveness.datasets.source_dataset import (
        BinaryReproductionImages,
        ReproductionImages,
    )
    from facial_attendance.liveness.evaluation.evaluation_metrics import (  # noqa: E402 — repository path is configured before local imports.
        binary_metrics,
        family_metrics,
        published_benchmark_metrics,
    )
    from facial_attendance.liveness.experiments.binary_experiment import TrainingConfig
    from facial_attendance.liveness.models.binary_model import (
        face_transform,
        load_checkpoint,
    )
    from facial_attendance.liveness.training.validation import (
        predict_dataset,
        save_predictions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Verified producing liveness modules for historical checkpoints",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Full-source reproduction evaluation; mutually exclusive with --readiness",
    )
    parser.add_argument(
        "--decoded-cache",
        type=Path,
        help="Verified predecoded cache for the selected split",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to the checkpoint validation batch size",
    )
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) / "evaluation"
        if "GPU_JOB_OUTPUT" in os.environ
        else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")
    if (args.batch_size is not None and args.batch_size < 1) or (
        args.workers is not None and args.workers < 0
    ):
        parser.error("Invalid batch-size/workers")
    args.workers = runtime_loader_workers(args.workers)
    model, meta = load_checkpoint(args.checkpoint, "cuda")
    if file_sha256(args.inventory) != meta["data"]["inventory_sha256"]:
        raise ValueError("Evaluation inventory differs from the frozen data release")
    if bool(args.readiness) == bool(args.annotations):
        parser.error(
            "Choose exactly one of --readiness (audited project subset) or --annotations (full-source reproduction)"
        )
    transform = face_transform(architecture=meta["architecture"])
    if args.annotations:
        expected = (
            meta["data"]["test_annotations_sha256"]
            if args.split == "test"
            else meta["data"]["validation"]["annotations_sha256"]
        )
        if file_sha256(args.annotations) != expected:
            raise ValueError(
                "Reproduction evaluation annotation split assignment changed"
            )
        source = ReproductionImages(args.data_root, args.inventory, args.annotations)
        recorded_crop = meta["data"]["train"].get("crop_recipe")
        if source.bindings().get("crop_recipe") != recorded_crop:
            raise ValueError(
                "Checkpoint crop recipe differs; use --source-snapshot with its producing source"
            )
        if args.decoded_cache is not None:
            # Historical source snapshots may not include the optional cache adapter.
            from facial_attendance.liveness.datasets.decoded_cache import DecodedFaces

            source = DecodedFaces(source, args.decoded_cache)
        dataset = BinaryReproductionImages(source, transform)
    else:
        if args.decoded_cache is not None:
            parser.error("--decoded-cache requires --annotations")
        if file_sha256(args.readiness) != meta["data"]["readiness_sha256"]:
            raise ValueError(
                "Evaluation readiness differs from the frozen data release"
            )
        dataset = LivenessImages(
            args.data_root,
            args.inventory,
            args.split,
            transform=transform,
            audit_file=args.readiness,
        )

    config = TrainingConfig(
        batch_size=args.batch_size or meta["training_config"].get("batch_size", 128),
        workers=args.workers,
    )
    labels, scores, loss, seconds = predict_dataset(
        model,
        dataset,
        config,
        autocast=meta.get("inference_precision", "float32") == "float16",
        on_batch=InferenceProgress(len(dataset), config.batch_size),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_predictions(output / "predictions.csv", dataset, labels, scores)

    result = {
        "split": args.split,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_version": meta["model_version"],
        "inference_precision": meta.get("inference_precision", "float32"),
        "threshold_source": meta["threshold_source"],
        "metrics": binary_metrics(labels, scores, meta["threshold"]),
        "unweighted_BCE": loss,
        "by_attack": family_metrics(
            labels,
            scores,
            [r.attack_family for r in dataset.samples],
            meta["threshold"],
        ),
        "evaluation_seconds": seconds,
        "by_source_type": {
            str(source_type): binary_metrics(
                labels[
                    np.asarray([r.source_type == source_type for r in dataset.samples])
                ],
                scores[
                    np.asarray([r.source_type == source_type for r in dataset.samples])
                ],
                meta["threshold"],
            )
            for source_type in sorted({r.source_type for r in dataset.samples})
        },
        "scope": "Frozen checkpoint and threshold; no model/threshold selection",
    }
    if args.annotations and args.split == "test":
        result["published_benchmark"] = published_benchmark_metrics(labels, scores)
    write_json(output / "evaluation.json", result)
    print(result)


if __name__ == "__main__":
    main()
