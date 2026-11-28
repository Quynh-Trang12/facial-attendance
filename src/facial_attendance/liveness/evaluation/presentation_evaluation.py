"""Evaluate declared physical presentations without treating frames as trials."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    local_path,
    read_json,
    source_path,
)
from facial_attendance.liveness.evaluation.evaluation_metrics import binary_metrics
from facial_attendance.liveness.inference.liveness_adapter import LivenessAdapter
from facial_attendance.liveness.models.face_detector import YUNET_SHA256


def score_recording(adapter, video, meta, *, maximum_frames=64, on_frame=None):
    """Score a recording with validated labels and frame count."""
    if maximum_frames < 1:
        raise ValueError("maximum-frames must be positive")

    capture = cv2.VideoCapture(str(video))
    frames = []
    every = max(1, int(np.ceil(meta["frames"] / maximum_frames)))
    trial_scores, sampled = [], 0

    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {video}")
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0 and sampled < maximum_frames:
                start = time.perf_counter()
                if on_frame is not None:
                    on_frame()
                prediction = adapter.predict_frame(frame)
                frames.append(
                    {
                        "trial_id": meta["trial_id"],
                        "frame_index": index,
                        "elapsed_seconds": time.perf_counter() - start,
                        **prediction,
                    }
                )
                sampled += 1
                if prediction["spoof_score"] is not None:
                    trial_scores.append(prediction["spoof_score"])
            index += 1
    finally:
        capture.release()

    if index != meta["frames"]:
        raise ValueError("Decoded frame coverage differs from recording metadata")

    score = float(np.median(trial_scores)) if trial_scores else None
    status = (
        "uncertain"
        if score is None
        else "spoof"
        if score >= adapter.predictor.meta["threshold"]
        else "live"
    )

    return {
        "trial_id": meta["trial_id"],
        "kind": meta["kind"],
        "binary_label": meta["binary_label"],
        "subject_id": meta["subject_id"],
        "session_id": meta["session_id"],
        "sampled_frames": sampled,
        "scored_frames": len(trial_scores),
        "median_spoof_score": score,
        "status": status,
        "allow_attendance": status == "live",
        "recording_sha256": meta["video_sha256"],
    }, frames


def evaluate_trials(
    checkpoint,
    trials_root,
    output,
    *,
    device="cuda",
    maximum_frames=64,
    detector_model=None,
    on_frame=None,
    brightness_factor=1.0,
    crop_margin=0.0,
):
    if maximum_frames < 1:
        raise ValueError("maximum-frames must be positive")
    trial_files = sorted(Path(trials_root).glob("*/trial.json"))
    if not trial_files:
        raise ValueError(
            "No operator-recorded physical trials; dataset images are not a substitute"
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    adapter = LivenessAdapter(
        checkpoint,
        device,
        detector_model=detector_model,
        brightness_factor=brightness_factor,
        crop_margin=crop_margin,
    )

    results, frames, inputs = [], [], {}
    for metadata_file in trial_files:
        meta = read_json(metadata_file)
        if (
            meta.get("schema") != "physical-liveness-trial-v1"
            or meta.get("kind") not in ("live", "print", "screen")
            or meta.get("binary_label") != int(meta["kind"] != "live")
        ):
            raise ValueError("Invalid physical trial declaration")
        video = local_path(metadata_file.parent, meta["video"])
        if file_sha256(video) != meta["video_sha256"]:
            raise ValueError("Physical recording changed since capture")
        inputs[str(metadata_file.resolve())] = file_sha256(metadata_file)
        inputs[str(video.resolve())] = meta["video_sha256"]
        result, trial_frames = score_recording(
            adapter, video, meta, maximum_frames=maximum_frames, on_frame=on_frame
        )
        results.append(result)
        frames.extend(trial_frames)

    scored = [r for r in results if r["median_spoof_score"] is not None]
    metrics = (
        binary_metrics(
            [r["binary_label"] for r in scored],
            [r["median_spoof_score"] for r in scored],
            adapter.predictor.meta["threshold"],
        )
        if scored
        else None
    )

    summary = {
        "trials": results,
        "presentation_metrics_on_scored_trials": metrics,
        "frame_prediction_counts": [
            {
                "trial_id": trial["trial_id"],
                **{
                    status: sum(
                        frame["trial_id"] == trial["trial_id"]
                        and frame["status"] == status
                        for frame in frames
                    )
                    for status in ("live", "spoof", "uncertain")
                },
            }
            for trial in results
        ],
        "uncertain_presentations": sum(r["status"] == "uncertain" for r in results),
        "operational_errors_including_uncertain": {
            "live_not_accepted": sum(
                r["kind"] == "live" and not r["allow_attendance"] for r in results
            ),
            "attack_accepted": sum(
                r["kind"] != "live" and r["allow_attendance"] for r in results
            ),
        },
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_version": adapter.predictor.meta["model_version"],
        "threshold": adapter.predictor.meta["threshold"],
        "preprocessing": {
            "brightness_factor": brightness_factor,
            "crop_margin_per_side": crop_margin,
            "order": (
                "YuNet on original frame; expand and clamp box; "
                "brighten RGB crop; checkpoint transform"
            ),
            "selection_limit": (
                "Physical-trial preprocessing adjusted after prior trial exposure; "
                "not an independent holdout"
            ),
        },
        "input_bindings": inputs,
        "source_sha256": {
            str(p.resolve()): file_sha256(p)
            for p in (
                Path(__file__),
                source_path("inference/liveness_adapter.py"),
                source_path("models/binary_model.py"),
                source_path("evaluation/evaluation_metrics.py"),
                source_path("models/face_detector.py"),
            )
        },
        "opencv": cv2.__version__,
        "face_detector": "YuNet 2023mar; auxiliary locator only",
        "detector_sha256": YUNET_SHA256,
        "method": (
            "At most 64 uniformly sampled frames per presentation; "
            "median valid spoof score uses the frozen validation threshold. "
            "Missing face is uncertain and cannot allow attendance."
        ),
        "limits": (
            "Operator-declared labels and subject/session tags; "
            "small physical presentation set, not independent frames. "
            "YuNet webcam crops differ from supplied RetinaFace dataset boxes. "
            "No physical threshold tuning."
        ),
    }

    for name, value in (
        ("frame-outputs.json", frames),
        ("physical-evaluation.json", summary),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return summary
