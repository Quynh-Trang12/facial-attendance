"""Webcam runner for the quality-gated multi-frame check-in (individual innovative feature).

  enroll     register a person from 5 good webcam frames, or from photos (--images);
             one template per crop rule (bare box, square 1.0, 1.2, 1.4) from the
             same frames, each in its own gallery file
  run        check in with --mode single (one frame, today's rule), --mode fused
             (quality gate + multi-frame fusion) or --mode both (both decide on the
             same frames, and every frame's quality and score under every crop rule
             is traced); --crop picks the gallery that decides, and both modes
             name a face only past max(tau, min_score) and the margin in
             configs/recognition.toml
  probe      score held-out photos of an enrolled person, or of someone never
             enrolled (--impostor), under every crop rule
  calibrate  record quality numbers of good or bad frames, no images
  limits     set [check_in.limits] from the validation images' quality range, and
             check them on the good and bad webcam recordings
  replay     replay recorded attempts through the recorded gate and the current
             limits: how many frames pass and how many attempts reach a decision
  summarize  turn the attempt log into the before/after summary
  diagnose   from a --mode both run and probes: does the gate help, and which crop
             rule separates the enrolled people best?

Keys: SPACE starts an attempt or a capture, Q or Esc quits. The galleries, logs and
calibration recordings go to the git-ignored artifacts/check-in-demo/ folder; frames
are never written to disk. Photos are read where they are and never copied.
"""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.check_in_live import (  # noqa: E402
    ENROLL_FRAMES,
    SCHEMA_RULE,
    STUDY_RULES,
    CheckIn,
    TRACE_COLUMNS,
    append_log,
    append_rows,
    check_limits,
    crop,
    crop_margin,
    dataset_limits,
    diagnose,
    enroll_study,
    faces_from_yunet,
    image_paths,
    limits_toml,
    load_study,
    photo_samples,
    replay_gate,
    study_embeddings,
    study_path,
    summarize,
    template_cosines,
    trace_row,
    with_limits,
)
from facial_attendance.recognition.decision import load_margin, load_min_score  # noqa: E402
from facial_attendance.recognition.probe_fusion import load_check_in_settings  # noqa: E402
from facial_attendance.recognition.quality import MEASURE_SIZE, check, measure, prompt  # noqa: E402

DEMO = ROOT / "artifacts/check-in-demo"
# The crop study keeps its galleries, logs and trace apart from earlier bare-box runs.
STUDY = DEMO / "crop-study"
CONFIG = ROOT / "configs/check-in-quality.toml"
DECISION_CONFIG = ROOT / "configs/recognition.toml"
EVIDENCE = ROOT / "evidence/quality-gated-check-in"
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


def open_study(args, meta):
    try:
        return load_study(args.galleries, meta)
    except ValueError as error:
        raise SystemExit(str(error))


def enroll_from_camera(args, embed, detect):
    """5 good frames; each frame's crop under every rule is embedded."""
    limits = None if args.size_only else load_check_in_settings(args.config).limits
    source, window = Source(args), Window("Enroll", True)
    samples, capturing, last = [], False, 0.0
    try:
        while len(samples) < ENROLL_FRAMES:
            frame_bgr, frame_rgb, now = source.read()
            if frame_bgr is None:
                raise SystemExit("The video ended before enrollment finished.")
            faces = detect(frame_bgr)
            text = f"Enroll {args.name}: SPACE to capture"
            if capturing and len(faces) == 1 and now - last >= 0.3:
                quality = measure(crop(frame_rgb, faces[0]), faces[0].landmarks)
                reasons = ("small",) if quality.size < MEASURE_SIZE else ()
                reasons = check(quality, limits) if limits and not reasons else reasons
                if reasons:
                    text = prompt(reasons)
                else:
                    samples.append(study_embeddings(frame_rgb, faces[0], embed))
                    last = now
            elif capturing:
                text = "One face, please" if faces else "No face found"
            key = window.draw(frame_bgr, faces, [text, "Turn your head slightly between frames"],
                              len(samples), ENROLL_FRAMES)
            if key == ord(" "):
                capturing = True
            if window.quit(key):
                raise SystemExit("Enrollment cancelled; nothing saved.")
    finally:
        source.close()
        window.close()
    return samples


def enroll_from_images(args, embed, detect):
    """The first 5 usable photos, through the same detection and crops as the webcam."""
    try:
        paths = image_paths(args.images)
    except ValueError as error:
        raise SystemExit(str(error))
    samples = []
    for sample in photo_samples(paths, detect, embed):
        if len(samples) == ENROLL_FRAMES:
            print(f"  not used (5 already taken): {sample.path.name}")
        elif sample.reason:
            print(f"  skipped {sample.path.name}: {sample.reason}")
        else:
            samples.append(sample.embeddings)
            print(f"  used {sample.path.name}: face {sample.quality.size} px")
    if len(samples) < ENROLL_FRAMES:
        raise SystemExit(f"Only {len(samples)} usable photos; enrollment needs {ENROLL_FRAMES}. Nothing saved.")
    return samples


def run_enroll(args):
    embed, meta = load_embedder(args.checkpoint)
    galleries = open_study(args, meta)
    detect = Detector(detector_file(args.download_detector))
    samples = enroll_from_images(args, embed, detect) if args.images else enroll_from_camera(args, embed, detect)
    try:
        closest = enroll_study(galleries, args.name, {rule: [sample[rule] for sample in samples] for rule in galleries},
                               allow_similar=args.allow_similar)
    except ValueError as error:
        hint = "" if args.allow_similar else " (for the crop study, --allow-similar keeps look-alikes)"
        raise SystemExit(f"Not registered under any crop rule: {error}{hint}")
    for rule, (label, cosine) in closest.items():
        if label is not None:
            flag = "  >= tau: these crops cannot tell them apart" if cosine >= galleries[rule].tau else ""
            print(f"  {rule}: closest template {label}, cosine {cosine:.3f}{flag}")
    for rule, gallery in galleries.items():
        gallery.save(study_path(args.galleries, rule))
    people = ", ".join(galleries[SCHEMA_RULE].labels)
    print(f"Registered {args.name} under {', '.join(galleries)}; the galleries now hold {people}")
    return 0


def describe(decision, prompt_text):
    if decision.accepted:
        return f"{decision.label} ({decision.score:.2f})"
    return prompt_text if decision.score is None else f"Unknown ({decision.score:.2f})"


def run_check_in(args):
    """One mode, or both on the same frames (--mode both) so the comparison is paired."""
    embed_frame, meta = load_embedder(args.checkpoint)
    galleries = open_study(args, meta)
    gallery = galleries[args.crop]
    if not gallery.labels:
        raise SystemExit("The galleries are empty; register someone with the enroll step first.")
    settings = load_check_in_settings(args.config)
    try:
        margin, min_score = load_margin(args.decision), load_min_score(args.decision)
    except (OSError, ValueError) as error:
        raise SystemExit(f"{args.decision}: {error}")
    paired = args.mode == "both"
    log = args.log or args.galleries / ("attempts-paired.csv" if paired else "attempts.csv")
    trace = args.trace or (args.galleries / "trace.csv" if paired else None)
    cache = {}

    def embed(face_rgb):
        # Both modes and the trace see the same crop of a frame; embed it once.
        key = (face_rgb.__array_interface__["data"][0], face_rgb.shape)
        if key not in cache:
            cache[key] = embed_frame(face_rgb)
        return cache[key]

    try:
        modes = ("single", "fused") if paired else (args.mode,)
        check_ins = {mode: CheckIn(mode, gallery, embed, margin=margin, min_score=min_score, settings=settings) for mode in modes}
    except ValueError as error:
        raise SystemExit(str(error))
    main = check_ins["fused" if paired else args.mode]
    if trace:
        try:
            append_rows(trace, TRACE_COLUMNS, [])  # refuse an older trace file before the camera opens
        except ValueError as error:
            raise SystemExit(str(error))
    detect, source = Detector(detector_file(args.download_detector)), Source(args)
    window = Window(f"Check-in ({args.mode})", not args.no_window)
    session = time.strftime("%Y%m%dT%H%M%S")
    shown, shown_until, count, attempt, started, traced = [], 0.0, 0, None, 0.0, []
    try:
        while True:
            frame_bgr, frame_rgb, now = source.read()
            if frame_bgr is None:
                break
            cache.clear()
            idle = not any(c.active for c in check_ins.values())
            if args.no_window and idle:
                count, attempt, started, shown = count + 1, f"{session}-{count + 1}", now, []
                for c in check_ins.values():
                    c.start(now)
            faces = detect(frame_bgr)
            if trace and any(c.active for c in check_ins.values()):
                if len(faces) == 1:
                    quality = measure(crop(frame_rgb, faces[0]), faces[0].landmarks)
                    reasons = check(quality, settings.limits) if settings.limits else ()
                    embeddings = study_embeddings(frame_rgb, faces[0], embed)
                    traced.append(trace_row(attempt, args.person, now - started, 1, quality, reasons, gallery,
                                            embeddings[args.crop],
                                            study={rule: (galleries[rule], embeddings[rule]) for rule in galleries}))
                else:
                    traced.append(trace_row(attempt, args.person, now - started, len(faces)))
            displays = {mode: c.step(frame_rgb, faces, now) for mode, c in check_ins.items()}
            for mode, display in displays.items():
                if display.decision:
                    append_log(log, args.person, display.decision, attempt if paired else None)
                    text = describe(display.decision, display.prompt)
                    shown.append((f"{mode}: {text}" if paired else text,
                                  GREEN if display.decision.accepted else RED))
                    shown_until = now + 2.0
                    print(f"attempt {count} {mode}: {text}, {1000 * display.decision.elapsed_s:.0f} ms, "
                          f"{display.decision.frames_used} frame(s), end={display.decision.end}", flush=True)
            if traced and not any(c.active for c in check_ins.values()):
                append_rows(trace, TRACE_COLUMNS, traced)
                traced = []
            current = displays[main.mode]
            if main.active or not shown or now >= shown_until:
                lines = [current.prompt] + [text for text, _ in shown]
                colour, kept, needed = AMBER if main.active else WHITE, current.kept, current.needed
            else:
                lines, colour, kept, needed = [text for text, _ in shown], shown[-1][1], 0, 0
            key = window.draw(frame_bgr, faces, [f"{args.mode} | {args.person} | attempts {count}", *lines],
                              kept, needed, colour)
            if key == ord(" ") and not any(c.active for c in check_ins.values()):
                count, attempt, started, shown = count + 1, f"{session}-{count + 1}", now, []
                for c in check_ins.values():
                    c.start(now)
            if window.quit(key):
                break
    finally:
        if traced:
            append_rows(trace, TRACE_COLUMNS, traced)
        source.close()
        window.close()
    print(f"{count} attempts logged to {log}" + (f"; frames traced to {trace}" if trace else ""))
    return 0


def run_probe(args):
    """Held-out photos of an enrolled person or an impostor, scored under every crop rule into the trace."""
    embed, meta = load_embedder(args.checkpoint)
    galleries = open_study(args, meta)
    enrolled = args.person in galleries[SCHEMA_RULE].labels
    if enrolled == args.impostor:
        raise SystemExit(f"{args.person!r} is enrolled; drop --impostor." if enrolled else
                         f"{args.person!r} is not enrolled; add --impostor if these are photos of someone never enrolled.")
    limits = load_check_in_settings(args.config).limits
    try:
        paths = image_paths(args.images)
        append_rows(args.trace, TRACE_COLUMNS, [])
    except ValueError as error:
        raise SystemExit(str(error))
    session, rows = time.strftime("%Y%m%dT%H%M%S"), []
    for index, sample in enumerate(photo_samples(paths, Detector(detector_file(args.download_detector)), embed), 1):
        if sample.reason:
            print(f"  skipped {sample.path.name}: {sample.reason}")
            continue
        reasons = check(sample.quality, limits) if limits else ()
        rows.append(trace_row(f"{session}-photo-{index}", args.person, 0.0, 1, sample.quality, reasons,
                              galleries[args.crop], sample.embeddings[args.crop],
                              study={rule: (galleries[rule], sample.embeddings[rule]) for rule in galleries},
                              source="image"))
        match = galleries[args.crop].match(sample.embeddings[args.crop])
        print(f"  {sample.path.name}: {args.crop} -> {match.label or 'unknown'} ({match.score:.2f})")
    append_rows(args.trace, TRACE_COLUMNS, rows)
    print(f"{len(rows)} of {len(paths)} photos scored into {args.trace}")
    return 0


def load_saved_study(directory):
    """The study galleries as saved, for their templates; no checkpoint is loaded."""
    import numpy as np

    with np.load(study_path(directory, SCHEMA_RULE)) as saved:
        meta = {"model_id": str(saved["model_id"]), "model_version": str(saved["model_version"]),
                "threshold": float(saved["tau"])}
    return load_study(directory, meta)


def saved_tau(path):
    import numpy as np

    with np.load(path) as saved:
        return float(saved["tau"])


def run_diagnose(args):
    tau = args.tau
    if tau is None and study_path(args.galleries, SCHEMA_RULE).exists():
        tau = saved_tau(study_path(args.galleries, SCHEMA_RULE))
    result = diagnose(read_rows(args.trace), read_rows(args.log) if args.log.exists() else [], tau=tau)
    if all(study_path(args.galleries, rule).exists() for rule in STUDY_RULES):
        result["crop_margins"]["template_cosine"] = template_cosines(load_saved_study(args.galleries))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
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
    """Limits from the dataset's quality range; the webcam recordings only check them."""
    dataset = json.loads(args.dataset.read_text())
    if "limits_start" not in dataset:
        raise SystemExit(f"{args.dataset} has no limits_start; use the discard-validation output")
    limits = dataset_limits(dataset["limits_start"], min_size=MEASURE_SIZE)
    result = {
        "rule": "sharpness and contrast at least the validation images' p5, brightness within their p5..p95, "
                "face at least the model input size, fixed head pose limits",
        "limits": limits,
        "dataset_file": str(args.dataset.relative_to(ROOT) if args.dataset.is_relative_to(ROOT) else args.dataset),
        "dataset_file_sha256": sha256(args.dataset.read_bytes()).hexdigest(),
        "webcam_check": check_limits(limits, read_rows(args.good), read_rows(args.bad)),
        "good_file_sha256": sha256(args.good.read_bytes()).hexdigest(),
        "bad_file_sha256": sha256(args.bad.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.write_config:
        source = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
        table = limits_toml(limits, [
            f"From {source}: the quality range of the validation images the model was checked on",
            "(p5 of sharpness and contrast, p5..p95 of brightness); min_size is the model input size;",
            "max_yaw and max_roll are fixed in check_in_live.py. Webcam recordings only check these limits.",
        ])
        args.config.write_text(with_limits(args.config.read_text(), table))
        load_check_in_settings(args.config)
        print(f"Limits written to {args.config}")
    return 0


def run_replay(args):
    settings = load_check_in_settings(args.config)
    if settings.limits is None:
        raise SystemExit(f"{args.config} has no [check_in.limits] to replay")
    result = {"config": str(args.config.relative_to(ROOT) if args.config.resolve().is_relative_to(ROOT) else args.config),
              "limits": asdict(settings.limits), "traces": {}}
    for path in args.trace:
        name = str(path.relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path)
        result["traces"][name] = replay_gate(read_rows(path), settings.limits, k=settings.k,
                                             min_frames=settings.min_frames, window_s=settings.window_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
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

    def crop_rule(value):
        try:
            crop_margin(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error))
        if value not in STUDY_RULES:
            raise argparse.ArgumentTypeError(f"choose one of the study's rules: {', '.join(STUDY_RULES)}")
        return value

    def model_options(sub):
        sub.add_argument("--checkpoint", type=Path, required=True, help="deploy.pt from FAC-05")
        sub.add_argument("--galleries", type=Path, default=STUDY,
                         help="folder of the crop study's gallery-<rule>.npz files, logs and trace")
        sub.add_argument("--config", type=Path, default=CONFIG)

    def camera_options(sub, checkpoint=True):
        source = sub.add_mutually_exclusive_group()
        source.add_argument("--camera", type=int, default=0)
        source.add_argument("--video", type=Path, help="a recorded clip instead of the camera")
        sub.add_argument("--download-detector", action="store_true", help="fetch the pinned YuNet file if missing")
        if checkpoint:
            model_options(sub)

    enroll = steps.add_parser("enroll", help="register a person under every crop rule")
    camera_options(enroll)
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--images", type=Path, nargs="+",
                        help="photos or folders instead of the camera; the first 5 usable photos are used")
    enroll.add_argument("--allow-similar", action="store_true",
                        help="crop study: keep a person whose template matches someone else's under a rule")
    enroll.add_argument("--size-only", action="store_true",
                        help="webcam: check only the face size, not the calibrated quality limits")

    run = steps.add_parser("run", help="check in and log each attempt")
    camera_options(run)
    run.add_argument("--mode", choices=("single", "fused", "both"), required=True,
                     help="both: single and fused decide on the same frames of each attempt")
    run.add_argument("--decision", type=Path, default=DECISION_CONFIG,
                     help="the [decision] margin and min_score both modes name a face under")
    run.add_argument("--person", required=True, help='who is in front of the camera, or "impostor"')
    run.add_argument("--crop", type=crop_rule, default=SCHEMA_RULE,
                     help=f"the crop rule whose gallery decides (default {SCHEMA_RULE}, the schema's)")
    run.add_argument("--log", type=Path,
                     help="default: attempts.csv, or attempts-paired.csv for both, in the galleries folder")
    run.add_argument("--trace", type=Path,
                     help="per-frame quality and scores; default trace.csv in the galleries folder for both")
    run.add_argument("--no-window", action="store_true", help="no display; start attempts automatically")

    probe = steps.add_parser("probe", help="score held-out photos of an enrolled person under every crop rule")
    model_options(probe)
    probe.add_argument("--download-detector", action="store_true", help="fetch the pinned YuNet file if missing")
    probe.add_argument("--images", type=Path, nargs="+", required=True, help="photos or folders")
    probe.add_argument("--person", required=True, help="who is in the photos")
    probe.add_argument("--impostor", action="store_true",
                       help="the person was never enrolled; their photos measure false accepts")
    probe.add_argument("--crop", type=crop_rule, default=SCHEMA_RULE, help="the crop rule shown on screen")
    probe.add_argument("--trace", type=Path, help="default: trace.csv in the galleries folder")

    calibrate = steps.add_parser("calibrate", help="record quality numbers of good or bad frames")
    camera_options(calibrate, checkpoint=False)
    calibrate.add_argument("--label", choices=("good", "bad"), required=True)
    calibrate.add_argument("--frames", type=int, default=60)
    calibrate.add_argument("--output", type=Path)

    limits = steps.add_parser("limits", help="limits from the dataset's quality range, checked on the recordings")
    limits.add_argument("--dataset", type=Path, default=EVIDENCE / "dataset-quality-validation.json",
                        help="discard-validation output of evaluate_check_in_quality.py, with limits_start")
    limits.add_argument("--good", type=Path, default=DEMO / "calibration-good.csv")
    limits.add_argument("--bad", type=Path, default=DEMO / "calibration-bad.csv")
    limits.add_argument("--output", type=Path, default=EVIDENCE / "calibration.json")
    limits.add_argument("--config", type=Path, default=CONFIG)
    limits.add_argument("--write-config", action="store_true", help="replace [check_in.limits] in the config")

    replay = steps.add_parser("replay", help="recorded attempts through the recorded gate and the current limits")
    replay.add_argument("--trace", type=Path, nargs="+", default=[DEMO / "trace.csv", STUDY / "trace.csv"])
    replay.add_argument("--config", type=Path, default=CONFIG)
    replay.add_argument("--output", type=Path, default=EVIDENCE / "gate-replay.json")

    summary = steps.add_parser("summarize", help="before/after summary of the attempt log")
    summary.add_argument("--log", type=Path, default=DEMO / "attempts.csv")
    summary.add_argument("--impostor", default="impostor")
    summary.add_argument("--output", type=Path, default=EVIDENCE / "live-ab.json")

    doctor = steps.add_parser("diagnose", help="gate, paired decisions and crop rules, from the trace")
    doctor.add_argument("--galleries", type=Path, default=STUDY, help="where the trace, log and galleries are")
    doctor.add_argument("--trace", type=Path, help="default: trace.csv in the galleries folder")
    doctor.add_argument("--log", type=Path, help="default: attempts-paired.csv in the galleries folder")
    doctor.add_argument("--tau", type=float, help=f"default: the threshold saved in gallery-{SCHEMA_RULE}.npz")
    doctor.add_argument("--output", type=Path,
                        default=EVIDENCE / "crop-study-diagnosis.json")

    args = parser.parse_args(argv)
    if getattr(args, "galleries", None) is not None:
        if hasattr(args, "trace") and args.trace is None and args.command in ("probe", "diagnose"):
            args.trace = args.galleries / "trace.csv"
        if args.command == "diagnose" and args.log is None:
            args.log = args.galleries / "attempts-paired.csv"
    handlers = {"enroll": run_enroll, "run": run_check_in, "probe": run_probe, "calibrate": run_calibrate,
                "limits": run_limits, "replay": run_replay, "summarize": run_summarize, "diagnose": run_diagnose}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
