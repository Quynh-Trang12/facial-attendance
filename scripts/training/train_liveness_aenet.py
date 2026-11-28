"""Train an AENet supervision configuration for 15 epochs or continue to 25."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import os
from dataclasses import asdict

from facial_attendance.liveness.experiments.aenet_experiment import (
    SUPERVISION_VARIANTS,
    FullTrainingConfig,
    run_full_training,
)


def main(argv=None):
    """Parse arguments and call the experiment with an explicit output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "inventory",
        "train-annotations",
        "validation-annotations",
        "geometry-cache",
        "imagenet-weights",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(SUPERVISION_VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--retain-epoch-checkpoints", action="store_true")
    parser.add_argument("--large-batch-size", type=int)
    parser.add_argument("--large-batch-start-epoch", type=int, default=1)
    parser.add_argument("--gpu-augmentation", action="store_true")
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--reviewed-cache-view", type=Path)
    default = FullTrainingConfig()
    for name in ("epochs", "batch_size", "workers", "seed"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(default, name)
        )
    for name in (
        "learning_rate",
        "momentum",
        "weight_decay",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(default, name)
        )

    args = parser.parse_args(argv)

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) if "GPU_JOB_OUTPUT" in os.environ else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")

    config = FullTrainingConfig(
        **{name: getattr(args, name) for name in asdict(default)}
    )

    result = run_full_training(
        args.data_root,
        args.inventory,
        args.train_annotations,
        args.validation_annotations,
        args.geometry_cache,
        args.imagenet_weights,
        output,
        config,
        arm=args.arm,
        preflight=args.preflight,
        resume=args.resume,
        decoded_cache=args.decoded_cache,
        reviewed_cache_view=args.reviewed_cache_view,
        screening=args.screening,
        retain_epoch_checkpoints=args.retain_epoch_checkpoints,
    )

    display = dict(result)
    if display.get("epoch_checkpoint_directory"):
        display["epoch_checkpoint_directory"] = Path(
            display["epoch_checkpoint_directory"]
        ).name

    print(json.dumps(display), flush=True)


if __name__ == "__main__":
    main()
