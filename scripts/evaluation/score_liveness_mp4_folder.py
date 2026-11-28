"""Validate and score MP4 camera trials directly from one folder."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import json
import tempfile

import cv2
import torch
from liveness_camera_inputs import count_frames, recordings

import facial_attendance.liveness.evaluation.presentation_evaluation as evaluator
from facial_attendance.liveness.dataset_files import file_sha256, source_root
from facial_attendance.liveness.inference.liveness_adapter import LivenessAdapter


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        temp = Path(stream.name)
    temp.replace(path)


def cache_key(bindings):
    return hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()


def preserve_previous_summary(cache):
    """Keep exact prior summary bytes before updating the current view."""
    latest = cache / "latest.json"
    if not latest.is_file():
        return
    content = latest.read_bytes()
    destination = cache / "summaries" / (hashlib.sha256(content).hexdigest() + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("Stored camera summary differs from its content checksum")
    else:
        with destination.open("xb") as handle:
            handle.write(content)


def cached_trial(cache, keys, identity, trial_id, checksum):
    for key in keys:
        receipt = cache / key / f"{identity}.json"
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            result = saved["result"]
            if (saved["key"], result["trial_id"], result["recording_sha256"]) != (
                key,
                trial_id,
                checksum,
            ):
                raise ValueError(f"Invalid cached result for {trial_id}")
            return saved
    return None


# Only these modules affect frame scoring. Training/audit edits must not invalidate
# prediction caches; the helper and entry point have separate bindings below.
INFERENCE_SOURCES = (
    "models/__init__.py",
    "inference/__init__.py",
    "evaluation/__init__.py",
    "models/binary_model.py",
    "dataset_files.py",
    "models/face_detector.py",
    "inference/liveness_adapter.py",
    "inference/temporal_liveness.py",
    "evaluation/presentation_evaluation.py",
    "evaluation/evaluation_metrics.py",
)


def evaluate_batch(args, clips, compatible_results):
    checkpoint_sha = file_sha256(args.checkpoint)
    source = source_root()
    bindings = dict(
        checkpoint=checkpoint_sha,
        detector=file_sha256(args.detector),
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        source={name: file_sha256(source / name) for name in INFERENCE_SOURCES},
        batch=file_sha256(Path(__file__)),
        input_helper=file_sha256(
            Path(__file__).resolve().parents[1] / "liveness_camera_inputs.py"
        ),
    )
    device = getattr(args, "device", "cuda")
    if device != "cuda":
        # Historical receipts describe CUDA inference; CPU scores have their own key.
        bindings["device"] = device
    key = cache_key(bindings)
    compatible_keys = [key]
    for result_version in compatible_results:
        if result_version.get("device", "cuda") != device:
            continue
        # Legacy keys are valid only for the reviewed input-helper implementation.
        if (
            result_version.get("accepted_input_helper_sha256")
            != bindings["input_helper"]
        ):
            continue
        if (
            result_version.get("accepted_source_sha256", bindings["source"])
            != bindings["source"]
        ):
            continue
        if (
            result_version.get("accepted_batch_sha256", bindings["batch"])
            != bindings["batch"]
        ):
            continue
        # A complete producing fingerprint must keep its original path keys.
        # Merging today's relocated paths would create a cache key that never existed.
        previous_source = (
            result_version["source"]
            if result_version.get("complete_source_bindings", False)
            else {**bindings["source"], **result_version["source"]}
        )
        previous = {
            **bindings,
            "source": previous_source,
            "batch": result_version["batch"],
        }
        if result_version.get("input_helper") is None:
            previous.pop("input_helper")
        else:
            previous["input_helper"] = result_version["input_helper"]
        compatible_keys.append(cache_key(previous))
    directory = args.cache / key
    directory.mkdir(parents=True, exist_ok=True)
    adapter = None
    rows, thresholds, inferred = [], set(), 0
    for trial_id, (video, checksum, subject, kind) in clips.items():
        identity = hashlib.sha256(f"{trial_id}:{checksum}".encode()).hexdigest()
        saved = cached_trial(args.cache, compatible_keys, identity, trial_id, checksum)
        if saved is None:
            metadata = dict(
                trial_id=trial_id,
                subject_id=subject,
                session_id=trial_id,
                kind=kind,
                binary_label=int(kind != "live"),
                frames=count_frames(video),
                video_sha256=checksum,
            )
            if adapter is None:
                adapter = LivenessAdapter(
                    args.checkpoint, device, detector_model=args.detector
                )
            result, frames = evaluator.score_recording(adapter, video, metadata)
            if file_sha256(video) != checksum:
                raise ValueError(f"Camera recording changed while scoring: {video}")
            saved = dict(
                key=key,
                result=result,
                threshold=adapter.predictor.meta["threshold"],
                frames=frames,
            )
            write_json(directory / f"{identity}.json", saved)
            inferred += 1
            print(f"Evaluated {trial_id}", flush=True)
        else:
            print(f"Reused {trial_id}", flush=True)
        rows.append(saved["result"])
        thresholds.add(saved["threshold"])
    (threshold,) = thresholds
    preserve_previous_summary(args.cache)
    write_json(
        args.cache / "latest.json",
        dict(
            checkpoint_sha256=checkpoint_sha,
            threshold=threshold,
            trials=rows,
            reused=len(rows) - inferred,
            inferred=inferred,
            bindings=bindings,
        ),
    )
    print(
        f"{len(rows)} camera trials: {inferred} evaluated, "
        f"{len(rows) - inferred} reused"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "trials", "detector", "cache", "inventory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    args = parser.parse_args(argv)
    try:
        clips = recordings(args.trials)
    except ValueError as error:
        parser.error(str(error))
    inventory = json.loads(args.inventory.read_text())
    compatible_results = inventory["evaluation"]["camera_evaluator"][
        "compatible_results"
    ]
    evaluate_batch(args, clips, compatible_results)


if __name__ == "__main__":
    main()
