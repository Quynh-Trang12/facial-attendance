"""Train the metric-learning (triplet loss) face embedding, or resume a run after a disconnect.

New run:    python scripts/train_metric.py --download --output-root /content/drive/MyDrive/fac-04-runs
Resume:     python scripts/train_metric.py --download --resume /content/drive/MyDrive/fac-04-runs/<run_id>
Warm start: python scripts/train_metric.py --download --output-root ... --warm-start <FAC-03 best.pt>
A warm start is a separate run, used only if training from scratch collapses.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.config import load_paths
from facial_attendance.recognition.data_loader import FaceLayout
from facial_attendance.recognition.download import download_face_data
from facial_attendance.recognition.exclusions import DEFAULT_EXCLUSIONS, apply_exclusions
from facial_attendance.recognition.metric_training import load_metric_config, train_metric

RUNTIME_ONLY = {"num_workers", "eval_batch_size"}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download", action="store_true", help="Use the Kaggle competition API")
    source.add_argument("--data-root", type=Path, help="Override the configured dataset location")
    parser.add_argument("--paths-config", type=Path, default=ROOT / "configs/project.toml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/face-metric.toml")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output-root", type=Path, help="Parent of new run directories (default: paths.checkpoints/FAC-04)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path, help="Run directory to continue from its last.pt")
    start.add_argument("--warm-start", type=Path, help="Classification best.pt whose backbone starts a new run")
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda when available)")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    smoke = parser.add_argument_group("smoke runs (recorded as changes; not a reported result)")
    smoke.add_argument("--epochs", type=int)
    smoke.add_argument("--max-steps-per-epoch", type=int)
    smoke.add_argument("--identities-per-batch", type=int)
    smoke.add_argument("--images-per-identity", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import torch

    args = parse_arguments(argv)
    paths = load_paths(args.paths_config, root=ROOT)
    if args.download:
        layout = download_face_data(cache_dir=paths.kaggle_cache)
    else:
        layout = FaceLayout.discover(args.data_root or paths.face_data)
    layout = apply_exclusions(layout, args.exclusions)

    overrides = {
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "identities_per_batch": args.identities_per_batch,
        "images_per_identity": args.images_per_identity,
        "num_workers": args.num_workers,
        "eval_batch_size": args.eval_batch_size,
    }
    config = load_metric_config(args.config, **overrides)
    changes = [
        {"setting": name, "value": value, "reason": "command-line override"}
        for name, value in overrides.items()
        if value is not None and name not in RUNTIME_ONLY
    ]
    if args.warm_start:
        changes.append({
            "setting": "initialisation",
            "value": str(args.warm_start),
            "reason": "warm start from the classification backbone, allowed when triplet training from scratch collapses",
        })
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_metric(
        config,
        layout,
        output_root=args.output_root or paths.checkpoints / "FAC-04",
        exclusions_file=args.exclusions,
        device=device,
        resume=args.resume,
        warm_start=args.warm_start,
        changes=changes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
