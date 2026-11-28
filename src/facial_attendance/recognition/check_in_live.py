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

from .decision import decide
from .probe_fusion import ProbeCollector, Ready, Retry, best_match, fuse
from .quality import REASONS, FaceQuality, QualityLimits, check, measure, prompt

ENROLL_FRAMES = 5
ATTEMPT_TIMEOUT_S = 10.0
# score is the best person's cosine and runner_up the best other person's (empty with one
# person enrolled). With best_label and the margin the attempt was decided under, the same
# attempts can be re-decided under another margin afterwards.
LOG_COLUMNS = ("mode", "person", "label", "accepted", "score", "frames_used", "retries", "elapsed_ms", "end",
               "runner_up", "best_label", "margin")


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2, clipped to the frame
    landmarks: np.ndarray | None  # YuNet order: right eye, left eye, nose, mouth right, mouth left
    raw: tuple[float, float, float, float] | None = None  # YuNet's x, y, w, h before clipping


def faces_from_yunet(rows, width, height) -> list[Face]:
    """Turn YuNet rows (x, y, w, h, 10 landmark values, score) into clipped faces.

    The box is clipped the same way as the GUI's YuNet wrapper, so a "box" crop here
    is the crop the team GUI sends to recognition. The unclipped box is kept for the
    schema's square crop, whose centre must not move when the face touches the edge.
    """
    faces = []
    for row in [] if rows is None else rows:
        x, y, w, h = map(float, row[:4])
        box = (max(0, floor(x)), max(0, floor(y)), min(width, ceil(x + w)), min(height, ceil(y + h)))
        if box[2] > box[0] and box[3] > box[1]:
            faces.append(Face(box, np.asarray(row[4:14], dtype=np.float64).reshape(5, 2), (x, y, w, h)))
    return faces


# A crop rule is "box" (the bare detector box, as the GUI stub crops today) or
# "square-<m>": the embedding schema's square of side m x max(w, h) around the box
# centre, clipped to the frame. The schema's default margin is 1.2.
SCHEMA_RULE = "square-1.2"
STUDY_RULES = ("box", "square-1.0", "square-1.2", "square-1.4")


def crop_margin(rule: str) -> float | None:
    """None for the bare box, else the square's margin m; anything else is refused."""
    if rule == "box":
        return None
    prefix, _, value = str(rule).partition("-")
    try:
        margin = float(value)
    except ValueError:
        margin = 0.0
    if prefix != "square" or not np.isfinite(margin) or margin <= 0:
        raise ValueError(f"Unknown crop rule {rule!r}; use 'box' or 'square-<margin>', such as {SCHEMA_RULE!r}")
    return margin


def crop(frame, face: Face, rule: str = "box"):
    """The face crop under a crop rule; the gate always measures the bare box."""
    margin = crop_margin(rule)
    if margin is None:
        x1, y1, x2, y2 = face.box
        return frame[y1:y2, x1:x2]
    if face.raw is not None:
        x, y, w, h = face.raw
    else:
        x, y, w, h = face.box[0], face.box[1], face.box[2] - face.box[0], face.box[3] - face.box[1]
    side = round(margin * max(w, h))
    left, top = round(x + w / 2 - side / 2), round(y + h / 2 - side / 2)
    height, width = frame.shape[:2]
    return frame[max(0, top):min(height, top + side), max(0, left):min(width, left + side)]


def study_embeddings(frame_rgb, face: Face, embed, rules=STUDY_RULES) -> dict:
    """One embedding per crop rule, from the same frame and face."""
    return {rule: embed(crop(frame_rgb, face, rule)) for rule in rules}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Phone photos are shrunk to about webcam size, so YuNet sees faces of a similar scale.
MAX_IMAGE_SIDE = 1280


def image_paths(inputs) -> list[Path]:
    """Image files given directly or found under folders, in a stable order."""
    paths = []
    for item in map(Path, inputs):
        if item.is_dir():
            paths += sorted((p for p in item.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                            key=lambda p: str(p.relative_to(item)).casefold())
        elif item.is_file():
            paths.append(item)
    if not paths:
        raise ValueError(f"Found no image in {', '.join(map(str, inputs))}")
    return paths


def read_image(path: Path, max_side=MAX_IMAGE_SIDE):
    """A BGR image like a camera frame; OpenCV applies the photo's EXIF rotation."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path} cannot be read as an image (HEIC is not supported; export JPEG)")
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


@dataclass(frozen=True)
class PhotoSample:
    path: Path
    face: Face | None = None
    quality: object = None
    embeddings: dict | None = None
    reason: str | None = None  # why the photo is not usable


def photo_samples(paths, detect, embed, rules=STUDY_RULES) -> list[PhotoSample]:
    """Each photo through the webcam path: YuNet, exactly one face of the model size, every crop rule."""
    import cv2

    from .quality import MEASURE_SIZE

    samples = []
    for path in paths:
        try:
            frame_bgr = read_image(path)
        except ValueError as error:
            samples.append(PhotoSample(path, reason=str(error)))
            continue
        faces = detect(frame_bgr)
        if len(faces) != 1:
            samples.append(PhotoSample(path, reason=f"{len(faces)} faces" if faces else "no face"))
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        quality = measure(crop(frame_rgb, faces[0]), faces[0].landmarks)
        if quality.size < MEASURE_SIZE:
            samples.append(PhotoSample(path, faces[0], quality, reason=f"face smaller than {MEASURE_SIZE} px"))
            continue
        samples.append(PhotoSample(path, faces[0], quality, study_embeddings(frame_rgb, faces[0], embed, rules)))
    return samples


@dataclass
class Gallery:
    """One averaged template per person, valid only for the model and crop rule that built it."""

    model_id: str
    model_version: str
    tau: float
    crop_rule: str
    labels: list[str] = field(default_factory=list)
    templates: np.ndarray | None = None

    def matrix(self, dim: int) -> np.ndarray:
        return self.templates if self.templates is not None else np.zeros((0, dim))

    def match(self, probe):
        return best_match(probe, self.labels, self.matrix(len(probe)), self.tau)

    def enroll(self, label: str, embeddings, allow_similar=False) -> np.ndarray:
        """Average the frames; refuse a mixed capture, a duplicate face or a taken name.

        allow_similar skips only the duplicate guard, for the crop study, where a crop
        that makes two people look alike is a result to measure, not a capture to redo.
        """
        if not label.strip() or any(label.casefold() == name.casefold() for name in self.labels):
            raise ValueError(f"The name {label!r} is empty or already registered")
        frames = np.asarray(embeddings, dtype=np.float64)
        template = fuse(frames)
        unit = frames / np.linalg.norm(frames, axis=1, keepdims=True)
        if (unit @ template).min() < self.tau:
            raise ValueError("The frames do not agree with each other; capture again")
        duplicate = self.match(template)
        if duplicate.accepted and not allow_similar:
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
                 crop_rule=self.crop_rule, labels=np.array(self.labels, dtype=str), templates=templates)
        temporary.replace(path)

    @classmethod
    def load_or_new(cls, path: Path, meta: dict, crop_rule: str) -> "Gallery":
        """Open the saved gallery for this checkpoint and crop rule, or start an empty one.

        Templates from one crop are not comparable with probes from another, so a
        gallery built with a different rule, or before the rule was recorded, is refused.
        """
        crop_margin(crop_rule)
        if not Path(path).exists():
            return cls(meta["model_id"], meta["model_version"], float(meta["threshold"]), crop_rule)
        with np.load(path) as saved:
            if "crop_rule" not in saved.files:
                raise ValueError(f"{path} has no crop rule; it predates the crop study, so register everyone again")
            gallery = cls(str(saved["model_id"]), str(saved["model_version"]), float(saved["tau"]),
                          str(saved["crop_rule"]), saved["labels"].tolist(),
                          saved["templates"] if len(saved["labels"]) else None)
        if (gallery.model_id, gallery.model_version) != (meta["model_id"], meta["model_version"]):
            raise ValueError("The gallery was built with another checkpoint; register everyone again")
        if gallery.crop_rule != crop_rule:
            raise ValueError(f"{path} was built with the crop rule {gallery.crop_rule!r}, not {crop_rule!r}")
        return gallery


def study_path(directory: Path, rule: str) -> Path:
    return Path(directory) / f"gallery-{rule}.npz"


def load_study(directory: Path, meta: dict, rules=STUDY_RULES) -> dict:
    """The crop study's galleries, one per rule; they must hold the same people."""
    galleries = {rule: Gallery.load_or_new(study_path(directory, rule), meta, rule) for rule in rules}
    if len({tuple(gallery.labels) for gallery in galleries.values()}) > 1:
        raise ValueError(f"The galleries in {directory} do not hold the same people; register everyone again")
    return galleries


def enroll_study(galleries: dict, label: str, embeddings: dict, allow_similar=False) -> dict:
    """Register one person under every crop rule from the same frames, or not at all.

    Returns, per rule, the closest person already enrolled and the cosine between the
    two templates, or (None, None) for the first person.
    """
    staged, closest = {}, {}
    for rule, gallery in galleries.items():
        copy = Gallery(gallery.model_id, gallery.model_version, gallery.tau, gallery.crop_rule,
                       list(gallery.labels), gallery.templates)
        try:
            template = copy.enroll(label, embeddings[rule], allow_similar)
        except ValueError as error:
            raise ValueError(f"{rule}: {error}") from None
        if gallery.labels:
            scores = gallery.templates @ template
            best = int(np.argmax(scores))
            closest[rule] = (gallery.labels[best], float(scores[best]))
        else:
            closest[rule] = (None, None)
        staged[rule] = copy
    for rule, copy in staged.items():
        galleries[rule].labels, galleries[rule].templates = copy.labels, copy.templates
    return closest


def template_cosines(galleries: dict) -> dict:
    """Per crop rule, the cosine between every two people's templates; lower separates better."""
    result = {}
    for rule, gallery in galleries.items():
        pairs = {}
        for i, first in enumerate(gallery.labels):
            for j in range(i + 1, len(gallery.labels)):
                pairs[f"{first} / {gallery.labels[j]}"] = float(gallery.templates[i] @ gallery.templates[j])
        result[rule] = pairs
    return result


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
    runner_up: float | None = None  # best other person's score; None with fewer than two people
    best_label: str | None = None  # the best match, named or not
    margin: float | None = None  # the margin this decision was made under


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

    Both modes name a face by the same rule, decide() in decision.py: the best score
    must reach max(tau, min_score) and beat every other enrolled person by margin.
    """

    def __init__(self, mode, gallery: Gallery, embed, *, margin, min_score=None, settings=None,
                 timeout_s=ATTEMPT_TIMEOUT_S):
        if mode not in ("single", "fused"):
            raise ValueError("mode is 'single' or 'fused'")
        if mode == "fused" and (settings is None or settings.limits is None):
            raise ValueError("The fused mode needs calibrated [check_in.limits]")
        self.mode, self.gallery, self.embed, self.settings, self.timeout_s = mode, gallery, embed, settings, timeout_s
        self.margin, self.min_score = margin, min_score
        decide([], [], gallery.tau, margin, min_score)  # refuse a bad margin or min_score now
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
        labels = self.gallery.labels
        rule = decide(labels, _scores(self.gallery, probe) if labels else [], self.gallery.tau, self.margin,
                      self.min_score)
        decision = Decision(self.mode, rule.label, rule.accepted, rule.best_score, frames_used, self.retries,
                            now - self.started, "match" if rule.accepted else "unknown",
                            rule.runner_up_score if len(set(labels)) >= 2 else None, rule.best_label, self.margin)
        self.started = None
        return Display("Welcome" if rule.accepted else "Not recognised", decision=decision)

    def step(self, frame_rgb, faces, now) -> Display:
        if not self.active:
            return Display("Press SPACE to check in")
        if now - self.started >= self.timeout_s:
            decision = Decision(self.mode, None, False, None, 0, self.retries, now - self.started, "timeout",
                                margin=self.margin)
            self.started = None
            return Display("Timed out, please try again", decision=decision)
        if len(faces) != 1:
            if self.collector:
                self.collector.reset()
            return Display("No face found" if not faces else "One person at a time")
        face_crop = crop(frame_rgb, faces[0], self.gallery.crop_rule)
        if self.mode == "single":
            return self._decide(self.embed(face_crop), 1, now)

        # The limits were calibrated on the bare box, so the gate measures that crop.
        reasons = check(measure(crop(frame_rgb, faces[0]), faces[0].landmarks), self.settings.limits)
        embedding = None if reasons else self.embed(face_crop)
        state = self.collector.add(embedding, reasons, now)
        if isinstance(state, Ready):
            return self._decide(state.probe, state.frames_used, now)
        if isinstance(state, Retry):
            self.retries += 1
            return Display(prompt((state.reason,)) or "Hold still and try again", 0, self.settings.k)
        return Display(prompt(state.reasons) or "Hold still", state.kept, state.needed)


def append_log(path: Path, person: str, decision: Decision, attempt: str | None = None):
    """One row per decision: numbers and names only, never images.

    A paired run passes the attempt id, so the single and fused decisions made on the
    same frames can be matched; such a log has an extra "attempt" column.
    """
    path = Path(path)
    columns = LOG_COLUMNS if attempt is None else (*LOG_COLUMNS, "attempt")
    if path.exists():
        with path.open(newline="") as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != columns:
            raise ValueError(f"{path} has other columns; use a new log file for this kind of run")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(columns)
        row = [decision.mode, person, decision.label or "", int(decision.accepted),
               "" if decision.score is None else f"{decision.score:.6f}", decision.frames_used,
               decision.retries, round(1000 * decision.elapsed_s), decision.end,
               "" if decision.runner_up is None else f"{decision.runner_up:.6f}", decision.best_label or "",
               "" if decision.margin is None else f"{decision.margin:g}"]
        writer.writerow(row if attempt is None else [*row, attempt])


STUDY_COLUMNS = tuple(f"{kind}:{rule}" for rule in STUDY_RULES for kind in ("own", "other", "other_label"))
TRACE_COLUMNS = ("attempt", "person", "source", "t_ms", "faces", "size", "sharpness", "brightness", "contrast",
                 "yaw", "roll", "reasons", "best_label", "best_score", "own_score", *STUDY_COLUMNS)


def _scores(gallery, embedding):
    probe = np.asarray(embedding, dtype=np.float64)
    return gallery.templates @ (probe / np.linalg.norm(probe))


def trace_row(attempt, person, t_s, faces, quality=None, reasons=(), gallery=None, embedding=None, *,
              study=None, source="camera"):
    """One frame of an attempt: its quality, why the gate failed it, and how it would score.

    Every face frame is embedded while tracing, including frames the gate rejects, so
    the scores of kept and rejected frames can be compared. gallery/embedding give the
    deciding crop's best match; study maps each crop rule to (gallery, embedding) and
    records the person's own score and the best score of anyone else under that rule.
    source is "camera" or "image". Numbers and names only, no images.
    """
    row = {"attempt": attempt, "person": person, "source": source, "t_ms": round(1000 * t_s), "faces": faces}
    if quality is not None:
        row |= {"size": quality.size, "sharpness": f"{quality.sharpness:.3f}",
                "brightness": f"{quality.brightness:.3f}", "contrast": f"{quality.contrast:.3f}",
                "yaw": "" if quality.yaw is None else f"{quality.yaw:.4f}",
                "roll": "" if quality.roll is None else f"{quality.roll:.3f}", "reasons": "|".join(reasons)}
    if embedding is not None and gallery is not None and gallery.labels:
        scores = _scores(gallery, embedding)
        best = int(np.argmax(scores))
        row |= {"best_label": gallery.labels[best], "best_score": f"{scores[best]:.6f}"}
        if person in gallery.labels:
            row["own_score"] = f"{scores[gallery.labels.index(person)]:.6f}"
    for rule, (study_gallery, study_embedding) in (study or {}).items():
        if not study_gallery.labels:
            continue
        scores = _scores(study_gallery, study_embedding)
        others = [index for index, label in enumerate(study_gallery.labels) if label != person]
        if person in study_gallery.labels:
            row[f"own:{rule}"] = f"{scores[study_gallery.labels.index(person)]:.6f}"
        if others:
            other = max(others, key=lambda index: scores[index])
            row |= {f"other:{rule}": f"{scores[other]:.6f}", f"other_label:{rule}": study_gallery.labels[other]}
    return [row.get(column, "") for column in TRACE_COLUMNS]


def append_rows(path: Path, columns, rows):
    """Append to a CSV file; a file with other columns is refused, never mixed."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != tuple(columns):
            raise ValueError(f"{path} has other columns; use a new file for this kind of run")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(columns)
        writer.writerows(rows)


def _spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _stats(values, spread=False):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    result = {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values))}
    if spread:
        result |= {"std": float(values.std()), "min": float(values.min()), "p10": float(np.percentile(values, 10))}
    return result


def _identity(frames, own_key, other_key, label_key, tau):
    own = np.array([float(row[own_key]) for row in frames])
    other = np.array([float(row[other_key]) for row in frames])
    wins = own > other
    confused = {}
    for row, won in zip(frames, wins):
        if not won:
            confused[row[label_key]] = confused.get(row[label_key], 0) + 1
    entry = {
        "frames": len(frames),
        "correct_identity_rate": float(wins.mean()),
        "own_minus_other": _stats(own - other, spread=True),
        "own_score": _stats(own, spread=True),
        "best_other_score": _stats(other),
        "confused_with": dict(sorted(confused.items())),
    }
    if tau is not None:
        entry["accept_rate"] = float((wins & (own >= tau)).mean())
    return entry


def crop_margins(trace, tau=None, rules=STUDY_RULES) -> dict:
    """Per crop rule: how well each enrolled person is told apart, and how impostors score.

    For an enrolled person (a frame with an own score), correct_identity_rate is the
    share of face frames whose own score is above the best other score; accept_rate
    also needs the own score to reach tau; own_minus_other is the margin by which the
    right person wins (negative when someone else wins). per_source splits camera
    frames from image probes. A person with no own score was never enrolled, so their
    frames are impostor probes: false_accept_rate is the share whose best score reaches
    tau. The balanced figures average the enrolled people, so a person with many
    frames does not dominate; the ranking sorts the rules by balanced correct rate,
    then balanced margin.
    """
    face_frames = [row for row in trace if row["faces"] == "1"]
    result = {}
    for rule in rules:
        keys = (f"own:{rule}", f"other:{rule}", f"other_label:{rule}")
        people, impostors = {}, {}
        for person in sorted({row["person"] for row in face_frames}):
            mine = [row for row in face_frames if row["person"] == person and row.get(keys[1])]
            frames = [row for row in mine if row.get(keys[0])]
            if frames:
                sources = sorted({row.get("source") or "camera" for row in frames})
                people[person] = _identity(frames, *keys, tau) | {"sources": sources, "per_source": {
                    source: _identity([row for row in frames if (row.get("source") or "camera") == source], *keys, tau)
                    for source in sources}}
            elif mine:
                best = np.array([float(row[keys[1]]) for row in mine])
                entry = {"frames": len(mine), "sources": sorted({row.get("source") or "camera" for row in mine}),
                         "best_score": _stats(best)}
                if tau is not None:
                    accepted = {}
                    for row, score in zip(mine, best):
                        if score >= tau:
                            accepted[row[keys[2]]] = accepted.get(row[keys[2]], 0) + 1
                    entry |= {"false_accept_rate": float((best >= tau).mean()),
                              "accepted_as": dict(sorted(accepted.items()))}
                impostors[person] = entry
        result[rule] = {
            "per_person": people,
            "impostors": impostors,
            "balanced_correct_identity_rate": (float(np.mean([p["correct_identity_rate"] for p in people.values()]))
                                               if people else None),
            "balanced_own_minus_other_mean": (float(np.mean([p["own_minus_other"]["mean"] for p in people.values()]))
                                              if people else None),
        }
    ranked = [rule for rule in rules if result[rule]["per_person"]]
    ranked.sort(key=lambda rule: (result[rule]["balanced_correct_identity_rate"],
                                  result[rule]["balanced_own_minus_other_mean"]), reverse=True)
    return {"tau": tau, "rules": result, "ranking": ranked}


def diagnose(trace, log, tau=None) -> dict:
    """Why the fused decision differs from the single one, and which crop separates people best.

    trace: rows of TRACE_COLUMNS; log: decision rows with an attempt column. The gate
    and paired figures use camera frames only; the crop study uses camera and image probes.
    """
    face_frames = [row for row in trace
                   if row["faces"] == "1" and row["sharpness"] and (row.get("source") or "camera") == "camera"]
    scored = [row for row in face_frames if row["own_score"]]
    passed = [float(row["own_score"]) for row in scored if not row["reasons"]]
    failed = [float(row["own_score"]) for row in scored if row["reasons"]]
    by_reason = {}
    for row in scored:
        for reason in filter(None, row["reasons"].split("|")):
            by_reason.setdefault(reason, []).append(float(row["own_score"]))
    own = np.array([float(row["own_score"]) for row in scored])
    measures = {}
    for name in ("sharpness", "brightness", "contrast", "size"):
        values = np.array([float(row[name]) for row in scored])
        measures[name] = _spearman(values, own)
    for name in ("yaw", "roll"):
        pairs = [(abs(float(row[name])), float(row["own_score"])) for row in scored if row[name]]
        measures[f"abs_{name}"] = _spearman(*map(np.array, zip(*pairs))) if pairs else None

    people = {}
    for person in sorted({row["person"] for row in face_frames}):
        frames = [row for row in face_frames if row["person"] == person]
        reasons = {}
        for row in frames:
            for reason in filter(None, row["reasons"].split("|")):
                reasons[reason] = reasons.get(reason, 0) + 1
        people[person] = {"face_frames": len(frames),
                          "gate_pass_rate": sum(not row["reasons"] for row in frames) / len(frames),
                          "failed_by": reasons}

    decisions = {}
    for row in log:
        decisions.setdefault(row["attempt"], {})[row["mode"]] = row

    def correct(row):
        return row["accepted"] == "1" and row["label"] == row["person"]

    paired = [d for d in decisions.values() if {"single", "fused"} <= set(d)]
    genuine = [d for d in paired if d["single"]["person"] != "impostor"]
    outcome = {"both_correct": 0, "only_single": 0, "only_fused": 0, "neither": 0}
    for d in genuine:
        key = {(True, True): "both_correct", (True, False): "only_single",
               (False, True): "only_fused", (False, False): "neither"}[(correct(d["single"]), correct(d["fused"]))]
        outcome[key] += 1
    score_pairs = [(float(d["single"]["score"]), float(d["fused"]["score"])) for d in genuine
                   if d["single"]["score"] and d["fused"]["score"] and correct(d["single"]) and correct(d["fused"])]
    return {
        "face_frames": len(face_frames),
        "own_score_gate_passed": _stats(passed),
        "own_score_gate_failed": _stats(failed),
        "own_score_by_reason": {reason: _stats(values) for reason, values in sorted(by_reason.items())},
        "spearman_measure_vs_own_score": measures,
        "per_person": people,
        "paired_attempts": len(paired),
        "paired_genuine_outcomes": outcome,
        "paired_scores_when_both_correct": {
            "n": len(score_pairs),
            "single_mean": float(np.mean([s for s, _ in score_pairs])) if score_pairs else None,
            "fused_mean": float(np.mean([f for _, f in score_pairs])) if score_pairs else None,
        },
        "impostor_paired": [{"single_accepted": d["single"]["accepted"] == "1",
                             "fused_accepted": d["fused"]["accepted"] == "1"}
                            for d in paired if d["single"]["person"] == "impostor"],
        "crop_margins": crop_margins(trace, tau),
    }


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


# Head pose limits are fixed, not measured: the dataset images carry no landmarks, and
# one webcam session is too narrow to set them. Yaw is the nose offset in inter-eye
# distances (pose() in quality.py); 0.35 is roughly a 30-degree turn for an average
# face. Live frames up to 0.28 and up to 13 degrees of roll still matched the right
# person, so these limits stop only clearly turned or tilted heads.
MAX_YAW = 0.35
MAX_ROLL = 15.0
GATE_MEASURES = ("sharpness", "brightness", "contrast")


def dataset_limits(percentiles, *, min_size, max_yaw=MAX_YAW, max_roll=MAX_ROLL) -> dict:
    """Image limits from the range of face images the model was validated on.

    percentiles is limit_percentiles() of the validation images: each measure's p5 and
    p95. A frame inside that range looks like the faces the model already handles, so
    only frames outside it are refused: sharpness and contrast below p5, brightness
    outside p5..p95. min_size is the model input size; the pose limits are fixed.
    """
    for name in GATE_MEASURES:
        if not {"p5", "p95"} <= set(percentiles.get(name, {})):
            raise ValueError(f"The dataset percentiles need p5 and p95 of {name}")
    return {
        "min_sharpness": float(percentiles["sharpness"]["p5"]),
        "min_brightness": float(percentiles["brightness"]["p5"]),
        "max_brightness": float(percentiles["brightness"]["p95"]),
        "min_contrast": float(percentiles["contrast"]["p5"]),
        "min_size": int(min_size),
        "max_yaw": float(max_yaw),
        "max_roll": float(max_roll),
    }


def _row_quality(row) -> FaceQuality:
    def optional(name):
        return None if row.get(name) in ("", None) else float(row[name])

    return FaceQuality(sharpness=float(row["sharpness"]), brightness=float(row["brightness"]),
                       contrast=float(row["contrast"]), size=int(float(row["size"])),
                       yaw=optional("yaw"), roll=optional("roll"))


def gate_reasons(row, limits: QualityLimits) -> tuple[str, ...]:
    """The gate's verdict on a recorded frame, from its measured numbers."""
    return check(_row_quality(row), limits)


def check_limits(limits: dict, good, bad) -> dict:
    """How the limits treat the webcam recordings; the recordings no longer set them."""
    gate = QualityLimits.from_mapping(limits)

    def tally(rows):
        reasons = [gate_reasons(row, gate) for row in rows]
        return {
            "frames": len(rows),
            "pass_rate": float(np.mean([not r for r in reasons])) if rows else None,
            "rejected_by": {name: int(sum(name in r for r in reasons)) for name in REASONS},
        }

    return {"good": tally(good), "bad": tally(bad)}


def limits_toml(limits: dict, note) -> str:
    """The [check_in.limits] table, with comment lines saying where the numbers came from."""
    notes = [note] if isinstance(note, str) else list(note)
    lines = ["", "[check_in.limits]", *(f"# {line}" for line in notes)]
    lines += [f"{name} = {value}" if isinstance(value, int) else f"{name} = {value:.6g}" for name, value in limits.items()]
    return "\n".join(lines) + "\n"


def with_limits(config_text: str, table: str) -> str:
    """The config text with its [check_in.limits] table, which must come last, replaced."""
    marker = "\n[check_in.limits]"
    head = config_text.split(marker, 1)[0] if marker in config_text else config_text
    return head.rstrip("\n") + "\n" + table


def replay_gate(trace, limits: QualityLimits, *, k, min_frames, window_s, timeout_s=ATTEMPT_TIMEOUT_S) -> dict:
    """Replay recorded attempts through the gate and the frame collector, old and new verdicts.

    Each camera attempt of a trace is fed frame by frame to a ProbeCollector, once with
    the reasons recorded live and once with the reasons the given limits produce. A
    frame without exactly one face resets the collector, as in CheckIn.step. The trace
    holds scores, not embeddings, so every kept frame gets the same vector: this counts
    how often the gate lets an attempt reach a decision and how fast, but neither the
    consistency check nor the fused score is replayed; that needs a new live run.
    """
    attempts = {}
    for row in trace:
        if row.get("source", "camera") == "camera":
            attempts.setdefault(row["attempt"], []).append(row)
    face_rows = [row for rows in attempts.values() for row in rows if row["faces"] == "1" and row["sharpness"]]
    new_reasons = {id(row): gate_reasons(row, limits) for row in face_rows}

    def run(verdict):
        ends, times, frames_used = {}, [], []
        for rows in attempts.values():
            collector = ProbeCollector(tau=0.0, k=k, min_frames=min_frames, window_s=window_s)
            end = "timeout"
            for row in sorted(rows, key=lambda r: int(r["t_ms"])):
                t_s = int(row["t_ms"]) / 1000
                if t_s >= timeout_s:
                    break
                if row["faces"] != "1" or not row["sharpness"]:
                    collector.reset()
                    continue
                reasons = verdict(row)
                state = collector.add(None if reasons else np.ones(2), reasons, t_s)
                if isinstance(state, Ready):
                    end = "decided"
                    times.append(t_s)
                    frames_used.append(state.frames_used)
                    break
            ends[end] = ends.get(end, 0) + 1
        return {
            "attempts": len(attempts),
            "decided": ends.get("decided", 0),
            "no_decision": ends.get("timeout", 0),
            "median_decision_s": float(np.median(times)) if times else None,
            "mean_frames_used": float(np.mean(frames_used)) if frames_used else None,
        }

    def frames(verdict):
        kept = [row for row in face_rows if not verdict(row)]
        rejected = [row for row in face_rows if verdict(row)]

        def own(rows):
            return _stats([float(row["own_score"]) for row in rows if row.get("own_score")])

        return {
            "pass_rate": len(kept) / len(face_rows) if face_rows else None,
            "rejected_by": {name: int(sum(name in verdict(row) for row in face_rows)) for name in REASONS},
            "own_score_kept": own(kept),
            "own_score_rejected": own(rejected),
        }

    recorded = lambda row: tuple(reason for reason in row["reasons"].split("|") if reason)  # noqa: E731
    current = lambda row: new_reasons[id(row)]  # noqa: E731
    return {
        "face_frames": len(face_rows),
        "note": "kept frames are replayed with one shared vector, so neither the consistency check "
                "nor the fused score is replayed; this counts decisions, not their correctness",
        "recorded_gate": {"frames": frames(recorded), "attempts": run(recorded)},
        "new_gate": {"frames": frames(current), "attempts": run(current)},
    }
