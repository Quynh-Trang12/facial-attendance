"""Verify the external final checkpoint and one synthetic inference."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from facial_attendance.emotion.final import EmotionPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = EmotionPredictor(args.checkpoint).predict_rgb(
        Image.new("L", (48, 48), 128)
    )
    assert (
        result["status"] == "ok"
        and abs(sum(result["probabilities"].values()) - 1) < 1e-5
    )
    print(
        json.dumps(
            {
                "check": "PASS",
                "scope": "Checkpoint integrity and synthetic inference only; not accuracy.",
                "mapping": {"0": "Sad", "1": "Happy", "2": "Neutral"},
            },
            indent=2,
        )
    )
