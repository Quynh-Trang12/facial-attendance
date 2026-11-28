"""Record a labelled MP4 for camera inference; keep capture metadata in evidence."""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from liveness_camera_inputs import NAME

from facial_attendance.liveness.dataset_files import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/liveness/camera-trials")
    )
    parser.add_argument(
        "--trial-id", help="Optional full filename ID; otherwise generated in UTC"
    )
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--session-id", help="Operator-declared session when known")
    parser.add_argument("--kind", choices=("live", "print", "screen"), required=True)
    parser.add_argument("--illumination", choices=("bright", "dim"), required=True)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Operator sidecars; defaults to the metadata subfolder of --output-root",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.subject_id)
        or not 1 <= args.seconds <= 60
    ):
        parser.error("Use a simple subject ID and duration from 1 to 60 seconds")
    captured = datetime.now(UTC)
    trial_id = (
        args.trial_id
        or f"{args.subject_id}-{captured:%Y%m%d_%H%M%S}-{args.kind}-{args.illumination}"
    )
    match = NAME.fullmatch(trial_id)
    if match is None:
        parser.error("Trial ID must use subject-YYYYMMDD_HHMMSS-kind-lighting")
    subject, day, clock, kind, lighting = match.groups()
    try:
        datetime.strptime(day + clock, "%Y%m%d%H%M%S")
    except ValueError:
        parser.error("Trial ID contains an invalid capture timestamp")
    if (subject, kind, lighting) != (args.subject_id, args.kind, args.illumination):
        parser.error("Trial ID conflicts with the declared labels")
    output = args.output_root / (trial_id + ".mp4")
    metadata_root = args.metadata_root or args.output_root / "metadata"
    metadata_file = metadata_root / (trial_id + ".trial.json")
    if output.exists() or metadata_file.exists():
        raise FileExistsError(output)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Camera did not open; no physical trial was recorded")
    writer = None
    count, started = 0, None
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = float(camera.get(cv2.CAP_PROP_FPS))
        fps = fps if 1 <= fps <= 120 else 20.0
        print(
            f"Prepare physical {args.kind} presentation; recording starts in 3 seconds.",
            flush=True,
        )
        time.sleep(3)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve a new filename; never overwrite another recording.
        with output.open("xb"):
            pass
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Video writer did not open")
            writer.write(frame)
            count += 1
            if args.preview:
                cv2.imshow("Physical liveness trial", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()
    if not count:
        raise RuntimeError("Empty recording is not a test trial")
    metadata = {
        "schema": "physical-liveness-trial-v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "trial_id": trial_id,
        "subject_id": args.subject_id,
        "session_id": args.session_id or trial_id,
        "session_source": "Operator-declared"
        if args.session_id
        else "Clip identifier; session independence unverified",
        "illumination": args.illumination,
        "kind": args.kind,
        "binary_label": int(args.kind != "live"),
        "label_source": "Operator-declared physical presentation; not inferred from model or filename",
        "video": output.name,
        "video_sha256": file_sha256(output),
        "frames": count,
        "container_fps": fps,
        "recording_seconds": time.monotonic() - started,
        "camera_index": args.camera,
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
