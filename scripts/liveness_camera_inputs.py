"""Camera filename validation and recording inventory shared by CLI entry points."""

import hashlib
import json
import os
import re
import subprocess
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

NAME = re.compile(r"(.+)-(\d{8})[-_](\d{6})-(live|screen|print)-(bright|dim)")


def recordings(root):
    videos = sorted(
        path
        for path in root.glob("*")
        if path.suffix.lower() == ".mp4" and path.is_file()
    )
    if not videos:
        raise ValueError("No MP4 camera recordings found")

    named = {}
    invalid = []
    for video in videos:
        match = NAME.fullmatch(video.stem)
        if match is None:
            invalid.append(str(video.relative_to(root)))
            continue
        subject, day, time, kind, light = match.groups()
        captured = f"{day}_{time}"
        try:
            datetime.strptime(captured, "%Y%m%d_%H%M%S")
        except ValueError:
            invalid.append(str(video.relative_to(root)))
            continue
        trial_id = f"{subject}-{captured}-{kind}-{light}"
        if trial_id in named:
            raise ValueError(
                f"Duplicate trial ID {trial_id}: {named[trial_id][0]} and {video}"
            )
        named[trial_id] = (video, subject, kind)
    if invalid:
        raise ValueError(
            "Invalid camera filename or timestamp; expected "
            "subject-YYYYMMDD_HHMMSS-(live|screen|print)-(bright|dim).mp4: "
            + ", ".join(invalid)
        )

    clips, checksums = {}, {}
    for trial_id, (video, subject, kind) in named.items():
        if not video.stat().st_size:
            raise ValueError(f"Empty camera recording: {video}")
        checksum = _file_digest(video)
        if checksum in checksums:
            raise ValueError(
                f"Duplicate video content: {checksums[checksum]} and {video}"
            )
        checksums[checksum] = video
        clips[trial_id] = (video, checksum, subject, kind)
    return clips


def count_frames(video):
    # Naming and inventory checks remain usable before OpenCV is installed.
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open camera recording: {video}")
        while capture.grab():
            frames += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decoded frames in camera recording: {video}")
    return frames


def _file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selected_checkpoint(run_root):
    """Find the validation winner by checksum, independent of its original host path."""
    choice = json.loads((run_root / "model-choice.json").read_text())
    arm = choice["chosen"]
    if Path(arm).name != arm or arm in ("", ".", ".."):
        raise ValueError("Invalid selected configuration name")
    expected = choice["arms"][arm]["checkpoint_sha256"]
    for directory in sorted(run_root.glob(arm + "*")):
        candidate = directory / "liveness-inference.pt"
        if candidate.is_file() and _file_digest(candidate) == expected:
            return arm, candidate, expected
    raise ValueError("Validation-selected checkpoint is missing or has changed")


def run_colab_inference(
    shared,
    project,
    checkpoint,
    checksum,
    camera_folder,
    output,
    release,
    batch_script,
    workflow,
):
    """Keep Colab source staging and CUDA setup out of the analytical notebook.

    Desktop inference uses the existing environment through the shared CLI. This
    route preserves the original isolated Colab CUDA environment and cache layout.
    """
    camera_work = project / "camera-work"
    camera_work.mkdir(parents=True, exist_ok=True)
    detector = "artifacts/detectors/face_detection_yunet_2023mar.onnx"
    required = {
        detector,
        ".python-version",
        "requirements.txt",
        "requirements-cuda.txt",
        "requirements.lock",
    }
    files = {
        name: dict(record)
        for name, record in release["files"].items()
        if name in required or name.startswith("src/")
    }
    for name, record in files.items():
        if (
            not (shared / record["delivery_path"]).is_file()
            and (shared / name).is_file()
        ):
            record["delivery_path"] = name
    inventory = camera_work / "camera-inputs.json"
    inventory.write_text(json.dumps({**release, "files": files}), encoding="utf-8")
    health = workflow.TrainingHealth(camera_work, "camera-preparation")
    args = SimpleNamespace(
        project_root=project,
        run_root=camera_work,
        source_root=shared,
        release_inventory=inventory,
        prepare_only=False,
    )
    print("Verifying camera inputs; details: source-verification.log", flush=True)
    with (
        (camera_work / "source-verification.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
    ):
        workflow.stage_verified_inputs(args, health)
    print(
        "Preparing the camera CUDA environment; details: gpu-environment.log",
        flush=True,
    )
    gpu_log = camera_work / "gpu-environment.log"
    try:
        with gpu_log.open("w", encoding="utf-8") as log, redirect_stdout(log):
            python = workflow.prepare_environment(args, health)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"Camera GPU preparation failed. Details: {gpu_log}"
        ) from None
    command = [
        str(python),
        "-u",
        str(batch_script),
        "--checkpoint",
        str(checkpoint),
        "--trials",
        str(camera_folder),
        "--detector",
        str(project / detector),
        "--cache",
        str(output / "camera-evaluation" / checksum),
        "--inventory",
        str(inventory),
    ]
    log_path = camera_work / "camera-inference.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                env={**os.environ, "PYTHONPATH": str(project / "src")},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = lines[-1] if lines else "Camera inference failed."
        raise RuntimeError(f"{detail} See {log_path}") from None
    print("Camera results saved.")
    return python
