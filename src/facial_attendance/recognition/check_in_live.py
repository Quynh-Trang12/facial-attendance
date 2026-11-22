"""Live check-in logic for the webcam runner, kept free of camera and network code.

The runner (scripts/run_check_in_demo.py) feeds frames, detected faces and an
embedding function into CheckIn, which applies either the schema's single-frame rule
("single") or the quality gate with multi-frame fusion ("fused"). Everything here is
testable with generated frames and a fake embedder.

Biometric data stays local: the gallery holds templates only, never frames, and is
written under the git-ignored artifacts/ directory by the runner.
"""

import csv
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path

import numpy as np

from .probe_fusion import ProbeCollector, Ready, Retry, best_match, fuse
from .quality import check, measure, prompt

ENROLL_FRAMES = 5
ATTEMPT_TIMEOUT_S = 10.0
LOG_COLUMNS = ("mode", "person", "label", "accepted", "score", "frames_used", "retries", "elapsed_ms", "end")


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2, clipped to the frame
    landmarks: np.ndarray | None  # YuNet order: right eye, left eye, nose, mouth right, mouth left


def faces_from_yunet(rows, width, height) -> list[Face]:
    """Turn YuNet rows (x, y, w, h, 10 landmark values, score) into clipped faces.

    The box is clipped the same way as the GUI's YuNet wrapper, so a crop here is the
    crop the team GUI sends to recognition.
    """
    faces = []
    for row in [] if rows is None else rows:
        x, y, w, h = map(float, row[:4])
        box = (max(0, floor(x)), max(0, floor(y)), min(width, ceil(x + w)), min(height, ceil(y + h)))
        if box[2] > box[0] and box[3] > box[1]:
            faces.append(Face(box, np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)))
    return faces


def crop(frame, face: Face):
    x1, y1, x2, y2 = face.box
    return frame[y1:y2, x1:x2]


@dataclass
class Gallery:
    """One averaged template per person, valid only for the model that built it."""

    model_id: str
    model_version: str
    tau: float
    labels: list[str] = field(default_factory=list)
    templates: np.ndarray | None = None

    def matrix(self, dim: int) -> np.ndarray:
        return self.templates if self.templates is not None else np.zeros((0, dim))

    def match(self, probe):
        return best_match(probe, self.labels, self.matrix(len(probe)), self.tau)

    def enroll(self, label: str, embeddings) -> np.ndarray:
        """Average the frames; refuse a mixed capture, a duplicate face or a taken name."""
        if not label.strip() or any(label.casefold() == name.casefold() for name in self.labels):
            raise ValueError(f"The name {label!r} is empty or already registered")
        frames = np.asarray(embeddings, dtype=np.float64)
        template = fuse(frames)
        unit = frames / np.linalg.norm(frames, axis=1, keepdims=True)
        if (unit @ template).min() < self.tau:
            raise ValueError("The frames do not agree with each other; capture again")
        duplicate = self.match(template)
        if duplicate.accepted:
            raise ValueError(f"This face already matches {duplicate.label!r} (score {duplicate.score:.3f})")
        self.labels.append(label)
        self.templates = np.vstack([self.matrix(len(template)), template])
        return template

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        templates = self.templates if self.templates is not None else np.zeros((0, 0))
        np.savez(temporary, model_id=self.model_id, model_version=self.model_version, tau=self.tau,
                 labels=np.array(self.labels, dtype=str), templates=templates)
        temporary.replace(path)

    @classmethod
    def load_or_new(cls, path: Path, meta: dict) -> "Gallery":
        """Open the saved gallery for this checkpoint, or start an empty one."""
        if not Path(path).exists():
            return cls(meta["model_id"], meta["model_version"], float(meta["threshold"]))
        with np.load(path) as saved:
            gallery = cls(str(saved["model_id"]), str(saved["model_version"]), float(saved["tau"]),
                          saved["labels"].tolist(), saved["templates"] if len(saved["labels"]) else None)
        if (gallery.model_id, gallery.model_version) != (meta["model_id"], meta["model_version"]):
            raise ValueError("The gallery was built with another checkpoint; register everyone again")
        return gallery


@dataclass(frozen=True)
class Decision:
    mode: str
    label: str | None
    accepted: bool
    score: float | None
    frames_used: int
    retries: int
    elapsed_s: float
    end: str  # "match", "unknown" or "timeout"


@dataclass(frozen=True)
class Display:
    prompt: str
    kept: int = 0
    needed: int = 0
    decision: Decision | None = None


class CheckIn:
    """One attempt at a time; step() is called for every camera frame.

    single: the first frame with exactly one face is embedded and matched, as the
    schema's rule does today. fused: every frame passes the quality gate first; good
    frames go to a ProbeCollector, and its fused probe is matched. A Retry (too few
    good frames or inconsistent frames) shows its reason and collection continues
    until the attempt times out.
    """

    def __init__(self, mode, gallery: Gallery, embed, *, settings=None, timeout_s=ATTEMPT_TIMEOUT_S):
        if mode not in ("single", "fused"):
            raise ValueError("mode is 'single' or 'fused'")
        if mode == "fused" and (settings is None or settings.limits is None):
            raise ValueError("The fused mode needs calibrated [check_in.limits]")
        self.mode, self.gallery, self.embed, self.settings, self.timeout_s = mode, gallery, embed, settings, timeout_s
        self.collector = (ProbeCollector(tau=gallery.tau, k=settings.k, min_frames=settings.min_frames,
                                         window_s=settings.window_s) if mode == "fused" else None)
        self.started = None

    @property
    def active(self) -> bool:
        return self.started is not None

    def start(self, now):
        self.started, self.retries = now, 0
        if self.collector:
            self.collector.reset()

    def _decide(self, probe, frames_used, now):
        match = self.gallery.match(probe)
        decision = Decision(self.mode, match.label, match.accepted, match.score, frames_used, self.retries,
                            now - self.started, "match" if match.accepted else "unknown")
        self.started = None
        return Display("Welcome" if match.accepted else "Not recognised", decision=decision)

    def step(self, frame_rgb, faces, now) -> Display:
        if not self.active:
            return Display("Press SPACE to check in")
        if now - self.started >= self.timeout_s:
            decision = Decision(self.mode, None, False, None, 0, self.retries, now - self.started, "timeout")
            self.started = None
            return Display("Timed out, please try again", decision=decision)
        if len(faces) != 1:
            if self.collector:
                self.collector.reset()
            return Display("No face found" if not faces else "One person at a time")
        face_crop = crop(frame_rgb, faces[0])
        if self.mode == "single":
            return self._decide(self.embed(face_crop), 1, now)

        reasons = check(measure(face_crop, faces[0].landmarks), self.settings.limits)
        embedding = None if reasons else self.embed(face_crop)
        state = self.collector.add(embedding, reasons, now)
        if isinstance(state, Ready):
            return self._decide(state.probe, state.frames_used, now)
        if isinstance(state, Retry):
            self.retries += 1
            return Display(prompt((state.reason,)) or "Hold still and try again", 0, self.settings.k)
        return Display(prompt(state.reasons) or "Hold still", state.kept, state.needed)


def append_log(path: Path, person: str, decision: Decision):
    """One row per attempt: numbers and names only, never images."""
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([decision.mode, person, decision.label or "", int(decision.accepted),
                         "" if decision.score is None else f"{decision.score:.6f}", decision.frames_used,
                         decision.retries, round(1000 * decision.elapsed_s), decision.end])


def summarize(rows, impostor="impostor") -> dict:
    """Per mode: genuine accept rate, misidentifications, impostor accept rate, decision time."""
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        attempts = [row for row in rows if row["mode"] == mode]
        genuine = [row for row in attempts if row["person"] != impostor]
        impostors = [row for row in attempts if row["person"] == impostor]
        times = np.array([int(row["elapsed_ms"]) for row in attempts])
        correct = sum(row["accepted"] == "1" and row["label"] == row["person"] for row in genuine)
        result[mode] = {
            "attempts": len(attempts),
            "people": sorted({row["person"] for row in genuine}),
            "genuine_attempts": len(genuine),
            "genuine_accept_rate": correct / len(genuine) if genuine else None,
            "misidentified": sum(row["accepted"] == "1" and row["label"] != row["person"] for row in genuine),
            "timeouts": sum(row["end"] == "timeout" for row in attempts),
            "impostor_attempts": len(impostors),
            "impostor_accept_rate": (sum(row["accepted"] == "1" for row in impostors) / len(impostors)
                                     if impostors else None),
            "decision_ms_median": float(np.median(times)) if len(times) else None,
            "decision_ms_p90": float(np.percentile(times, 90)) if len(times) else None,
        }
    return result


def calibrate_limits(good, bad, *, min_size, keep=0.95) -> dict:
    """Set the limits so that about a share `keep` of good webcam frames passes all of them.

    Six limits that each kept 95 % would together keep only about 74 % of good frames
    (0.95 ** 6), so each limit keeps keep ** (1 / 6) instead. good/bad: rows with
    sharpness, brightness, contrast, size, yaw and roll (yaw/roll may be empty).
    min_size is fixed by the model input, not calibrated. Reports the share of good
    frames that pass all limits and how many bad frames each limit rejects.
    """
    def values(rows, name):
        return np.array([float(row[name]) for row in rows if row.get(name) not in ("", None)])

    tail = 100 * (1 - keep ** (1 / 6))
    brightness = values(good, "brightness")
    limits = {
        "min_sharpness": float(np.percentile(values(good, "sharpness"), tail)),
        "min_brightness": float(np.percentile(brightness, tail / 2)),
        "max_brightness": float(np.percentile(brightness, 100 - tail / 2)),
        "min_contrast": float(np.percentile(values(good, "contrast"), tail)),
        "min_size": int(min_size),
        "max_yaw": float(np.percentile(np.abs(values(good, "yaw")), 100 - tail)),
        "max_roll": float(np.percentile(np.abs(values(good, "roll")), 100 - tail)),
    }

    def failures(row):
        reasons = []
        if float(row["sharpness"]) < limits["min_sharpness"]:
            reasons.append("blur")
        if not limits["min_brightness"] <= float(row["brightness"]) <= limits["max_brightness"]:
            reasons.append("light")
        if float(row["contrast"]) < limits["min_contrast"]:
            reasons.append("low_contrast")
        if int(float(row["size"])) < limits["min_size"]:
            reasons.append("small")
        if row.get("yaw") not in ("", None) and abs(float(row["yaw"])) > limits["max_yaw"]:
            reasons.append("turned")
        if row.get("roll") not in ("", None) and abs(float(row["roll"])) > limits["max_roll"]:
            reasons.append("tilted")
        return reasons

    bad_reasons = [failures(row) for row in bad]
    return {
        "limits": limits,
        "keep": keep,
        "keep_per_limit": keep ** (1 / 6),
        "good_frames": len(good),
        "good_pass_rate": float(np.mean([not failures(row) for row in good])) if good else None,
        "bad_frames": len(bad),
        "bad_reject_rate": float(np.mean([bool(r) for r in bad_reasons])) if bad else None,
        "bad_rejected_by": {name: int(sum(name in r for r in bad_reasons))
                            for name in ("blur", "light", "low_contrast", "small", "turned", "tilted")},
    }


def limits_toml(limits: dict, source: str) -> str:
    """The [check_in.limits] table, with where the numbers came from."""
    lines = ["", "[check_in.limits]", f"# From {source}: together the limits keep about 95 % of good webcam frames;",
             "# min_size is the model input size, not calibrated."]
    lines += [f"{name} = {value}" if isinstance(value, int) else f"{name} = {value:.6g}" for name, value in limits.items()]
    return "\n".join(lines) + "\n"
