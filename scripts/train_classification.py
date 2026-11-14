"""Train the classification-based face embedding, or resume a run after a disconnect.

New run:  python scripts/train_classification.py --download --output-root /content/drive/MyDrive/fac-03-runs
Resume:   python scripts/train_classification.py --download --resume /content/drive/MyDrive/fac-03-runs/<run_id>
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
from facial_attendance.recognition.training import load_config, train


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download", action="store_true", help="Use the Kaggle competition API")
    source.add_argument("--data-root", type=Path, help="Override the configured dataset location")
    parser.add_argument("--paths-config", type=Path, default=ROOT / "configs/project.toml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/face-classification.toml")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output-root", type=Path, help="Parent of new run directories (default: paths.checkpoints/FAC-03)")
    parser.add_argument("--resume", type=Path, help="Run directory to continue from its last.pt")
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda when available)")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    smoke = parser.add_argument_group("smoke runs (recorded as changes; not a reported result)")
    smoke.add_argument("--epochs", type=int)
    smoke.add_argument("--max-steps-per-epoch", type=int)
    smoke.add_argument("--batch-size", type=int)
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
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "eval_batch_size": args.eval_batch_size,
    }
    config = load_config(args.config, **overrides)
    changes = [
        {"setting": name, "value": value, "reason": "command-line override"}
        for name, value in overrides.items()
        if value is not None and name not in {"num_workers", "eval_batch_size"}
    ]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train(
        config,
        layout,
        output_root=args.output_root or paths.checkpoints / "FAC-03",
        exclusions_file=args.exclusions,
        device=device,
        resume=args.resume,
        changes=changes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
