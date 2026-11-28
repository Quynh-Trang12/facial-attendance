"""Write the selected cosine threshold into a copy of the selected face checkpoint.

The embedding schema keeps the threshold in the checkpoint's meta, so a threshold is
never applied to another model. Training leaves it empty; the comparison chooses it on
the validation pairs and records it in selection.json. This script copies best.pt to a
new file with `threshold` and `threshold_source` filled in. The weights, and so
`model_version`, stay the same, and the source checkpoint is not changed.

    python scripts/set_face_threshold.py --checkpoint <run>/best.pt \
        --selection evidence/FAC-05/selection.json --output <run>/deploy.pt
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.comparison import VAL_PAIRS_SHA256, file_sha256  # noqa: E402
from facial_attendance.recognition.model import load_checkpoint, save_checkpoint  # noqa: E402


def deploy(checkpoint: Path, selection_file: Path, output: Path) -> dict:
    """Copy the checkpoint with the selected threshold; refuse any other model."""
    if output.exists():
        raise ValueError(f"{output} already exists; choose a new file")
    selection = json.loads(selection_file.read_text())
    decision = selection["decision"]
    if decision["reject_all"] or decision["threshold"] is None:
        raise ValueError("selection.json has no usable threshold")
    model, meta = load_checkpoint(checkpoint)
    for key, want in (("model_id", decision["selected_model_id"]),
                      ("model_version", decision["selected_model_version"]),
                      ("approach", decision["selected_approach"])):
        if meta.get(key) != want:
            raise ValueError(f"Checkpoint {key} is {meta.get(key)!r}; selection.json selected {want!r}")
    if meta.get("threshold") not in (None, decision["threshold"]):
        raise ValueError(f"Checkpoint already holds a different threshold {meta['threshold']}")
    source = (f"FAC-05 selection.json sha256 {file_sha256(selection_file)}: "
              f"{decision['threshold_rule']} on validation pairs {VAL_PAIRS_SHA256[:12]} "
              f"(FAR {decision['validation_far']:.4f}, TAR {decision['validation_tar']:.4f}); "
              "accept when cosine >= threshold")
    save_checkpoint(output, model, {**meta, "threshold": decision["threshold"], "threshold_source": source})
    _, written = load_checkpoint(output)
    if written["model_version"] != meta["model_version"] or written["threshold"] != decision["threshold"]:
        raise ValueError("The written checkpoint does not reload with the same weights and threshold")
    return {"output": str(output), "sha256": file_sha256(output),
            **{k: written[k] for k in ("model_id", "model_version", "approach", "threshold", "threshold_source")}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The selected run's best.pt")
    parser.add_argument("--selection", type=Path, required=True, help="Committed selection.json")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint file")
    args = parser.parse_args(argv)
    print(json.dumps(deploy(args.checkpoint, args.selection, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
