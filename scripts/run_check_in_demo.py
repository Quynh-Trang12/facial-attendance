"""Webcam runner for the quality-gated multi-frame check-in (individual innovative feature).

  enroll     register a person: 5 good frames averaged into one template
  run        check in with --mode single (one frame, today's rule) or --mode fused
             (quality gate + multi-frame fusion), logging one row per attempt
  calibrate  record quality numbers of good or bad frames, no images
  limits     turn the calibration recordings into [check_in.limits]
  summarize  turn the attempt log into the before/after summary

Keys: SPACE starts an attempt or a capture, Q or Esc quits. The gallery, logs and
calibration recordings go to the git-ignored artifacts/check-in-demo/ folder; frames
are never written to disk.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.check_in_live import (  # noqa: E402
    ENROLL_FRAMES,
    CheckIn,
    Gallery,
    append_log,
    calibrate_limits,
    crop,
    faces_from_yunet,
    limits_toml,
    summarize,
)
from facial_attendance.recognition.probe_fusion import load_check_in_settings  # noqa: E402
from facial_attendance.recognition.quality import MEASURE_SIZE, check, measure, prompt  # noqa: E402

DEMO = ROOT / "artifacts/check-in-demo"
CONFIG = ROOT / "configs/check-in-quality.toml"
# Same file, folder and hash as the GUI's SYS-05 YuNet setup, so both share one model.
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_SIZE = 232589
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_FILENAME
YUNET_MODEL = ROOT / "artifacts/sys05-yunet" / YUNET_FILENAME
CALIBRATION_COLUMNS = ("sharpness", "brightness", "contrast", "size", "yaw", "roll")
GREEN, RED, AMBER, WHITE = (60, 180, 75), (60, 60, 220), (0, 170, 255), (255, 255, 255)


def detector_file(download: bool) -> Path:
    if not YUNET_MODEL.is_file():
        if not download:
            raise SystemExit(f"Missing {YUNET_MODEL}. Run again with --download-detector to fetch it.")
        from urllib.request import urlopen

        with urlopen(YUNET_URL, timeout=30) as response:
            data = response.read(YUNET_SIZE + 1)
        if len(data) != YUNET_SIZE or sha256(data).hexdigest() != YUNET_SHA256:
            raise SystemExit("The downloaded YuNet file has the wrong size or hash; nothing was saved.")
        YUNET_MODEL.parent.mkdir(parents=True, exist_ok=True)
        YUNET_MODEL.write_bytes(data)
    if sha256(YUNET_MODEL.read_bytes()).hexdigest() != YUNET_SHA256:
        raise SystemExit(f"{YUNET_MODEL} does not have the pinned hash; refusing it.")
    return YUNET_MODEL


class Detector:
    """OpenCV YuNet with the GUI's settings; keeps the five landmarks."""

    def __init__(self, path: Path):
        import cv2

        self.cv2, self.path, self.model = cv2, path, None

    def __call__(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        if self.model is None:
            self.model = self.cv2.FaceDetectorYN.create(str(self.path), "", (width, height), 0.9, 0.3, 5000)
        self.model.setInputSize((width, height))
        _, rows = self.model.detect(frame_bgr)
        return faces_from_yunet(rows, width, height)


def load_embedder(checkpoint: Path):
    """The deployed checkpoint, with the evaluation preprocessing used for the official AUC."""
    import torch
    from PIL import Image

    from facial_attendance.recognition.model import eval_transform, load_checkpoint

    model, meta = load_checkpoint(checkpoint, "cpu")
    if meta.get("threshold") is None:
        raise SystemExit("The checkpoint has no deployed threshold; use deploy.pt from FAC-05.")
    transform = eval_transform(meta["input_size"])

    @torch.no_grad()
    def embed(face_rgb):
        image = transform(Image.fromarray(face_rgb)).unsqueeze(0)
        return model.embed(image)[0].numpy().astype("float64")

    return embed, meta


class Source:
    """Camera frames on a monotonic clock, or video frames on the video's own clock."""

    def __init__(self, args):
        import cv2

        self.cv2 = cv2
        self.video = args.video is not None
        self.capture = cv2.VideoCapture(str(args.video) if self.video else args.camera)
        if not self.capture.isOpened():
            raise SystemExit("Could not open the camera or video. On macOS, allow camera access for your terminal.")

    def read(self):
        ok, frame_bgr = self.capture.read()
        if not ok:
            return None, None, None
        now = self.capture.get(self.cv2.CAP_PROP_POS_MSEC) / 1000 if self.video else time.monotonic()
        return frame_bgr, self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB), now

    def close(self):
        self.capture.release()


class Window:
    def __init__(self, title, show):
        import cv2

        self.cv2, self.title, self.show = cv2, title, show

    def draw(self, frame_bgr, faces, lines, kept=0, needed=0, colour=WHITE):
        if not self.show:
            return None
        cv2 = self.cv2
        for face in faces:
            x1, y1, x2, y2 = face.box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
        for row, text in enumerate(lines):
            cv2.putText(frame_bgr, text, (12, 30 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame_bgr, text, (12, 30 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        if needed:
            for index in range(needed):
                fill = GREEN if index < kept else (90, 90, 90)
                cv2.rectangle(frame_bgr, (12 + 34 * index, 30 + 30 * len(lines)),
                              (40 + 34 * index, 44 + 30 * len(lines)), fill, -1)
        cv2.imshow(self.title, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def quit(key):
        return key in (ord("q"), ord("Q"), 27)

    def close(self):
        if self.show:
            self.cv2.destroyAllWindows()


def run_enroll(args):
    embed, meta = load_embedder(args.checkpoint)
    gallery = Gallery.load_or_new(args.gallery, meta)
    limits = load_check_in_settings(args.config).limits
    detect, source, window = Detector(detector_file(args.download_detector)), Source(args), Window("Enroll", True)
    embeddings, capturing, last = [], False, 0.0
    try:
        while len(embeddings) < ENROLL_FRAMES:
            frame_bgr, frame_rgb, now = source.read()
            if frame_bgr is None:
                raise SystemExit("The video ended before enrollment finished.")
            faces = detect(frame_bgr)
            text = f"Enroll {args.name}: SPACE to capture"
            if capturing and len(faces) == 1 and now - last >= 0.3:
                face_crop = crop(frame_rgb, faces[0])
                quality = measure(face_crop, faces[0].landmarks)
                reasons = ("small",) if quality.size < MEASURE_SIZE else ()
                reasons = check(quality, limits) if limits else reasons
                if reasons:
                    text = prompt(reasons)
                else:
                    embeddings.append(embed(face_crop))
                    last = now
            elif capturing:
                text = "One face, please" if faces else "No face found"
            key = window.draw(frame_bgr, faces, [text, "Turn your head slightly between frames"],
                              len(embeddings), ENROLL_FRAMES)
            if key == ord(" "):
                capturing = True
            if window.quit(key):
                raise SystemExit("Enrollment cancelled; nothing saved.")
    finally:
        source.close()
        window.close()
    try:
        gallery.enroll(args.name, embeddings)
    except ValueError as error:
        raise SystemExit(f"Not registered: {error}")
    gallery.save(args.gallery)
    print(f"Registered {args.name}; the gallery now holds {', '.join(gallery.labels)}")
    return 0


def run_check_in(args):
    embed, meta = load_embedder(args.checkpoint)
    gallery = Gallery.load_or_new(args.gallery, meta)
    if not gallery.labels:
        raise SystemExit("The gallery is empty; register someone with the enroll step first.")
    settings = load_check_in_settings(args.config)
    try:
        check_in = CheckIn(args.mode, gallery, embed, settings=settings)
    except ValueError as error:
        raise SystemExit(str(error))
    detect, source = Detector(detector_file(args.download_detector)), Source(args)
    window = Window(f"Check-in ({args.mode})", not args.no_window)
    shown, shown_until, count = None, 0.0, 0
    try:
        while True:
            frame_bgr, frame_rgb, now = source.read()
            if frame_bgr is None:
                break
            if args.no_window and not check_in.active:
                check_in.start(now)
            faces = detect(frame_bgr)
            display = check_in.step(frame_rgb, faces, now)
            if display.decision:
                count += 1
                append_log(args.log, args.person, display.decision)
                decision = display.decision
                result = (f"{decision.label} ({decision.score:.2f})" if decision.accepted
                          else display.prompt if decision.score is None else f"Unknown ({decision.score:.2f})")
                shown, shown_until = (result, f"{1000 * decision.elapsed_s:.0f} ms, {decision.frames_used} frame(s)",
                                      GREEN if decision.accepted else RED), now + 2.0
                print(f"attempt {count}: {result}, {1000 * decision.elapsed_s:.0f} ms, end={decision.end}", flush=True)
            if shown and now < shown_until:
                lines, colour, kept, needed = [shown[0], shown[1]], shown[2], 0, 0
            else:
                lines, colour, kept, needed = [display.prompt], AMBER if check_in.active else WHITE, display.kept, display.needed
            key = window.draw(frame_bgr, faces, [f"{args.mode} | {args.person} | attempts {count}", *lines],
                              kept, needed, colour)
            if key == ord(" ") and not check_in.active:
                check_in.start(now)
                shown = None
            if window.quit(key):
                break
    finally:
        source.close()
        window.close()
    print(f"{count} attempts logged to {args.log}")
    return 0


def run_calibrate(args):
    detect, source, window = Detector(detector_file(args.download_detector)), Source(args), Window("Calibrate", True)
    output = args.output or DEMO / f"calibration-{args.label}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    new = not output.exists()
    recorded, recording, last = 0, False, 0.0
    with output.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(CALIBRATION_COLUMNS)
        try:
            while recorded < args.frames:
                frame_bgr, frame_rgb, now = source.read()
                if frame_bgr is None:
                    break
                faces = detect(frame_bgr)
                if recording and len(faces) == 1 and now - last >= 0.1:
                    quality = measure(crop(frame_rgb, faces[0]), faces[0].landmarks)
                    writer.writerow([quality.sharpness, quality.brightness, quality.contrast, quality.size,
                                     quality.yaw, quality.roll])
                    recorded, last = recorded + 1, now
                hint = ("normal, steady, well lit" if args.label == "good"
                        else "move, dim the light, turn your head more than 20 degrees")
                key = window.draw(frame_bgr, faces, [f"{args.label} frames {recorded}/{args.frames}",
                                                     "SPACE to record: " + hint])
                if key == ord(" "):
                    recording = True
                if window.quit(key):
                    break
        finally:
            source.close()
            window.close()
    print(f"Recorded {recorded} {args.label} frames to {output}")
    return 0


def read_rows(path: Path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_limits(args):
    result = calibrate_limits(read_rows(args.good), read_rows(args.bad), min_size=MEASURE_SIZE)
    result["good_file_sha256"] = sha256(args.good.read_bytes()).hexdigest()
    result["bad_file_sha256"] = sha256(args.bad.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.write_config:
        text = args.config.read_text()
        if load_check_in_settings(args.config).limits is not None:
            raise SystemExit(f"{args.config} already has limits; edit them by hand if they must change.")
        source = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
        args.config.write_text(text.rstrip("\n") + "\n" + limits_toml(result["limits"], str(source)))
        load_check_in_settings(args.config)
        print(f"Limits written to {args.config}")
    return 0


def run_summarize(args):
    rows = read_rows(args.log)
    result = {"log_rows": len(rows), "modes": summarize(rows, impostor=args.impostor)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="command", required=True)

    def camera_options(sub, checkpoint=True):
        source = sub.add_mutually_exclusive_group()
        source.add_argument("--camera", type=int, default=0)
        source.add_argument("--video", type=Path, help="a recorded clip instead of the camera")
        sub.add_argument("--download-detector", action="store_true", help="fetch the pinned YuNet file if missing")
        if checkpoint:
            sub.add_argument("--checkpoint", type=Path, required=True, help="deploy.pt from FAC-05")
            sub.add_argument("--gallery", type=Path, default=DEMO / "gallery.npz")
            sub.add_argument("--config", type=Path, default=CONFIG)

    enroll = steps.add_parser("enroll", help="register a person")
    camera_options(enroll)
    enroll.add_argument("--name", required=True)

    run = steps.add_parser("run", help="check in and log each attempt")
    camera_options(run)
    run.add_argument("--mode", choices=("single", "fused"), required=True)
    run.add_argument("--person", required=True, help='who is in front of the camera, or "impostor"')
    run.add_argument("--log", type=Path, default=DEMO / "attempts.csv")
    run.add_argument("--no-window", action="store_true", help="no display; start attempts automatically")

    calibrate = steps.add_parser("calibrate", help="record quality numbers of good or bad frames")
    camera_options(calibrate, checkpoint=False)
    calibrate.add_argument("--label", choices=("good", "bad"), required=True)
    calibrate.add_argument("--frames", type=int, default=60)
    calibrate.add_argument("--output", type=Path)

    limits = steps.add_parser("limits", help="calibrated limits from the recordings")
    limits.add_argument("--good", type=Path, default=DEMO / "calibration-good.csv")
    limits.add_argument("--bad", type=Path, default=DEMO / "calibration-bad.csv")
    limits.add_argument("--output", type=Path, default=ROOT / "evidence/quality-gated-check-in/calibration.json")
    limits.add_argument("--config", type=Path, default=CONFIG)
    limits.add_argument("--write-config", action="store_true")

    summary = steps.add_parser("summarize", help="before/after summary of the attempt log")
    summary.add_argument("--log", type=Path, default=DEMO / "attempts.csv")
    summary.add_argument("--impostor", default="impostor")
    summary.add_argument("--output", type=Path, default=ROOT / "evidence/quality-gated-check-in/live-ab.json")

    args = parser.parse_args(argv)
    handlers = {"enroll": run_enroll, "run": run_check_in, "calibrate": run_calibrate,
                "limits": run_limits, "summarize": run_summarize}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
