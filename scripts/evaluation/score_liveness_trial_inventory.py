"""Evaluate recorded live, printed-photo and screen presentations using a saved model.

Each recording is one trial. Sample up to 64 frames, take the median available
spoof score and compare it with the checkpoint's saved validation threshold.
A recording with no scored frames is uncertain, not an accepted live trial.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager

from liveness_camera_inputs import count_frames, recordings
from liveness_training_health import InferenceProgress

from facial_attendance.liveness.evaluation.presentation_evaluation import (
    evaluate_trials,
)


@contextmanager
def trial_metadata_view(root):
    """Adapt flat MP4 inputs to the historical scorer without changing its maths."""
    root = Path(root)
    if any(root.glob("*/trial.json")):
        yield root
        return
    clips = recordings(root)
    with tempfile.TemporaryDirectory(prefix="liveness-camera-") as temporary:
        staged = Path(temporary)
        for trial_id, (video, checksum, subject, kind) in clips.items():
            folder = staged / trial_id
            folder.mkdir()
            # A real copy keeps the scorer's existing path-containment checks valid.
            shutil.copyfile(video, folder / "recording.mp4")
            metadata = {
                "schema": "physical-liveness-trial-v1",
                "trial_id": trial_id,
                "subject_id": subject,
                "session_id": trial_id,
                "session_source": "Clip identifier; session independence remains unverified",
                "kind": kind,
                "binary_label": int(kind != "live"),
                "video": "recording.mp4",
                "video_sha256": checksum,
                "source_recording": str(video.resolve()),
                "frames": count_frames(video),
            }
            (folder / "trial.json").write_text(json.dumps(metadata) + "\n")
        yield staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--brightness-factor", type=float, default=1.0)
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.0,
        help="Fraction of detected width/height added on each side",
    )

    args = parser.parse_args()

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) / "physical-tests"
        if "GPU_JOB_OUTPUT" in os.environ
        else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")
    with trial_metadata_view(args.trials_root) as trial_root:
        total_frames = 0
        for path in trial_root.glob("*/trial.json"):
            frames = json.loads(path.read_text())["frames"]
            every = max(1, math.ceil(frames / 64))
            total_frames += min(64, math.ceil(frames / every))
        progress = InferenceProgress(total_frames, 1, unit="sampled frames")
        input_records = []
        for path in sorted(trial_root.glob("*/trial.json")):
            content = path.read_bytes()
            input_records.append(
                {
                    "trial_id": json.loads(content)["trial_id"],
                    "metadata_sha256": hashlib.sha256(content).hexdigest(),
                    "original_metadata_utf8": content.decode("utf-8"),
                }
            )
        report = evaluate_trials(
            args.checkpoint,
            trial_root,
            output,
            device=args.device,
            detector_model=args.detector_model,
            on_frame=progress,
            brightness_factor=args.brightness_factor,
            crop_margin=args.crop_margin,
        )
        with (output / "recording-inputs.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_root": str(args.trials_root.resolve()),
                    "recordings": input_records,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(report)


if __name__ == "__main__":
    main()
