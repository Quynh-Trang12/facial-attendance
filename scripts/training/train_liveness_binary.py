"""Train the bounded binary baseline from an audited image inventory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import os

from facial_attendance.liveness.experiments.binary_experiment import (
    TrainingConfig,
    run_training,
)


def main(argv=None):
    """Parse arguments and call the experiment with an explicit output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--epoch-samples",
        type=int,
        help="Training samples per logical epoch; no replacement within an epoch",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        help="Freeze loss weight across data-policy comparisons",
    )
    parser.add_argument(
        "--epoch-live-samples",
        type=int,
        help="Preserve baseline live/spoof counts in a matched epoch",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)

    args = parser.parse_args(argv)

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) if "GPU_JOB_OUTPUT" in os.environ else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        epoch_samples=args.epoch_samples,
        pos_weight=args.pos_weight,
        epoch_live_samples=args.epoch_live_samples,
    )
    run_training(
        args.data_root,
        args.inventory,
        args.readiness,
        output,
        config,
        smoke=args.smoke,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
