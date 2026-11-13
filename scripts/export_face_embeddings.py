"""Export a checkpoint's inputs for the shared evaluator, without retraining.

Writes into a new directory: raw embeddings for the official trials and for a run's
validation pairs (NPZ: paths, raw_embeddings), and top-1 predictions on val_data and
test_data (id,true,predicted CSV plus face_labels.json).
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.config import load_paths
from facial_attendance.recognition.data_loader import FaceLayout
from facial_attendance.recognition.download import download_face_data
from facial_attendance.recognition.embeddings import export_predictions, export_trial_embeddings
from facial_attendance.recognition.exclusions import DEFAULT_EXCLUSIONS, apply_exclusions
from facial_attendance.recognition.model import load_checkpoint


def main(argv=None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download", action="store_true")
    source.add_argument("--data-root", type=Path)
    parser.add_argument("--paths-config", type=Path, default=ROOT / "configs/project.toml")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-pairs", type=Path, help="The run's val-pairs.txt")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory; earlier exports are kept")

    paths = load_paths(args.paths_config, root=ROOT)
    layout = (
        download_face_data(cache_dir=paths.kaggle_cache)
        if args.download
        else FaceLayout.discover(args.data_root or paths.face_data)
    )
    layout = apply_exclusions(layout, args.exclusions)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(args.checkpoint, device)
    summary = {"model_id": meta["model_id"], "model_version": meta["model_version"]}
    summary["official_trials"] = export_trial_embeddings(
        model, meta, layout, layout.pairs_file, args.output / "official-embeddings.npz", device=device
    )
    if args.val_pairs:
        summary["val_pairs"] = export_trial_embeddings(
            model, meta, layout, args.val_pairs, args.output / "val-pairs-embeddings.npz", device=device
        )
    summary["classification"] = [
        export_predictions(model, meta, layout, split, args.output, device=device)
        for split in ("val_data", "test_data")
    ]
    (args.output / "export-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
