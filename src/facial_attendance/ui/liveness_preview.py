"""Connect the selected AENet release to the read-only local workspace."""

from dataclasses import dataclass
from datetime import datetime
from math import fsum, isfinite
from pathlib import Path
from time import monotonic

from ..liveness.dataset_files import file_sha256
from .emotion_presentation import geometry
from .live_preview import evaluate_adapters

CHECKPOINT_SHA256 = "6715d53461e6d123d34ca714c7b24b8b66a726949889bcf4f72955fceb1aa430"
MODEL_ID = "liveness-resnet18-aenet-full"
MODEL_VERSION = "f16fbfaf36046f75"
EPOCH = 18
THRESHOLD = 0.46483826637268066


def valid_spoof_score(prediction, threshold):
    if prediction is None or prediction["status"] not in {"live", "spoof"}:
        return None
    score = prediction["spoof_score"]
    if (
        score is not None
        and isfinite(score)
        and 0 <= score <= 1
        and prediction["threshold"] == threshold
    ):
        return score
    return None


def decide_liveness(scores, threshold):
    if not scores:
        return "uncertain", None
    mean = fsum(scores) / len(scores)
    return ("spoof" if mean >= threshold else "live"), mean


@dataclass(frozen=True)
class LivenessSample:
    frame_status: str
    spoof_score: float | None = None


@dataclass(frozen=True)
class LastLivenessCheck:
    decision: str
    mean_spoof_score: float | None
    valid_frames: int
    local_time: str


class ReleasedLiveness:
    def __init__(self, checkpoint, detector_model):
        from ..liveness.inference.liveness_adapter import LivenessAdapter
        from ..liveness.models.face_detector import YuNetFaceDetector

        checkpoint = Path(checkpoint)
        if file_sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError(
                "Select the AENet-Full epoch 18 checkpoint; the file checksum does not match."
            )
        self.adapter = LivenessAdapter(checkpoint, device="cpu")
        meta = self.adapter.predictor.meta
        if (
            meta["model_id"],
            meta["model_version"],
            meta["epoch"],
            meta["threshold"],
        ) != (MODEL_ID, MODEL_VERSION, EPOCH, THRESHOLD):
            raise ValueError(
                "The AENet-Full checkpoint metadata does not match the selected release."
            )
        self.detector = YuNetFaceDetector(detector_model)
        self.provenance = {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "epoch": EPOCH,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "threshold": THRESHOLD,
            "precision": "cpu_float32",
        }

    def detect(self, frame):
        # Trang's detector returns clipped source-pixel xywh. The GUI extracts
        # the exact same original-frame crop from xyxy.
        return [
            (x, y, x + width, y + height)
            for x, y, width, height in self.detector.detect(frame)
        ]

    def evaluate(self, job):
        import cv2
        from PIL import Image

        request, adapters, *_ = job
        start = monotonic()
        try:
            image = Image.fromarray(cv2.cvtColor(request.face_crop, cv2.COLOR_BGR2RGB))
            prediction = self.adapter.predict_face(image)
        except (RuntimeError, ValueError, cv2.error):
            prediction = None
        finally:
            adapters.diagnostics.duration("liveness", monotonic() - start)
        return evaluate_adapters(job, unverified_preview=True), prediction


class LivenessPreview:
    """Score one webcam scan at a time using the mean of its valid frames."""

    SCAN_SECONDS = 3.0

    def __init__(self, threshold):
        self.threshold = threshold
        self.clear()

    def clear(self):
        self._started_at = None
        self._scores = []
        self._generation = self._person_id = None
        self._last_capture = None
        self.token = self.sample = self.last_check = None
        self.last_check_window = None
        self.box = None

    def advance(self, captured_at, *, pending_at=None):
        """Close a scan once its sampled frames have completed or expired."""
        if self._started_at is None:
            self._started_at = captured_at
            return
        deadline = self._started_at + self.SCAN_SECONDS
        if captured_at < deadline or (pending_at is not None and pending_at < deadline):
            return
        count = len(self._scores)
        decision, mean = decide_liveness(self._scores, self.threshold)
        self.last_check = LastLivenessCheck(
            decision, mean, count, datetime.now().strftime("%H:%M:%S")
        )
        self.last_check_window = (self._started_at, captured_at)
        self._scores.clear()
        self._started_at = captured_at

    def accept(self, token, box, captured_at, state, prediction):
        if (
            self._started_at is None
            or not self._started_at
            <= captured_at
            < self._started_at + self.SCAN_SECONDS
            or (self._last_capture is not None and captured_at <= self._last_capture)
        ):
            return
        self._last_capture = captured_at
        self.token = token
        self.sample = LivenessSample("error")
        if prediction is None:
            return
        frame_status = prediction["status"]
        self.sample = LivenessSample(frame_status, prediction["spoof_score"])
        score = valid_spoof_score(prediction, self.threshold)
        if score is None:
            return

        recognition = state.recognition
        person_id = (
            recognition.person_id
            if state.ui_state == "preview"
            and recognition is not None
            and recognition.source == "model"
            and recognition.accepted is True
            else None
        )
        identity_changed = (
            self._person_id is not None
            and person_id is not None
            and person_id != self._person_id
        )
        separated = self.box is not None and geometry(self.box, box)[0] == 0
        if self._generation is not None and (
            token[0] != self._generation or identity_changed or separated
        ):
            # End the previous subject's scan without mixing its scores into this one.
            self._scores.clear()
            self._started_at = captured_at
            self.last_check = self.last_check_window = None
            self._person_id = None
        self._generation = token[0]
        if person_id is not None:
            self._person_id = person_id
        self.box = box
        self._scores.append(score)

    def current(self, live, now, max_age):
        if self.sample is None or live.token != self.token or len(live.boxes) != 1:
            return None
        if not 0 <= now - live.sample_time <= max_age:
            return None
        return self.sample

    def display_text(self):
        if self.last_check is None:
            return "Liveness · Collecting frames…"
        return f"Liveness · Last scan at {self.last_check.local_time}\n{self.last_check.decision}"
