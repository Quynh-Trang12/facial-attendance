"""Run the matched binary-control and semantic training experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import os
from dataclasses import asdict

from facial_attendance.liveness.experiments.multitask_experiment import (
    MultitaskTrainingConfig,
    run_multitask_training,
)


def main(argv=None):
    """Parse arguments and call the experiment with an explicit output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "inventory",
        "readiness",
        "train-annotations",
        "validation-annotations",
        "imagenet-weights",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    defaults = MultitaskTrainingConfig()
    for name in (
        "epochs",
        "batch_size",
        "workers",
        "patience",
        "seed",
        "epoch_samples",
        "validation_samples",
        "max_seconds",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in ("learning_rate", "momentum", "weight_decay"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )

    args = parser.parse_args(argv)

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) if "GPU_JOB_OUTPUT" in os.environ else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")

    config = MultitaskTrainingConfig(
        **{name: getattr(args, name) for name in asdict(defaults)}
    )
    run_multitask_training(
        args.data_root,
        args.inventory,
        args.readiness,
        args.train_annotations,
        args.validation_annotations,
        args.imagenet_weights,
        output,
        config,
    )


if __name__ == "__main__":
    main()
