"""Score two face images with a saved checkpoint: cosine similarity and Euclidean distance.

python scripts/score_face_pair.py --checkpoint best.pt left.jpg right.jpg
Higher cosine (lower distance) means more likely the same person. No threshold is
applied here; FAC-05 chooses it on the validation pairs.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.embeddings import score_images
from facial_attendance.recognition.model import load_checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    model, meta = load_checkpoint(args.checkpoint)
    result = {"model_id": meta["model_id"], "model_version": meta["model_version"]}
    result |= score_images(model, meta, args.left, args.right)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
