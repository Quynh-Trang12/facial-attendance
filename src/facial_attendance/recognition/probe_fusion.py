"""Multi-frame check-in: fuse several good frames into one probe before matching.

The attendance rule in docs/embedding-schema.md matches one live embedding against
the gallery. One webcam frame is noisy, so a registered person can fall below the
threshold. Here the frames that pass the quality gate are collected for a short
window, averaged with the enrollment formula normalize(mean(normalize(e_i))), and
the fused probe is matched with the same rule and the same threshold.

A consistency check mirrors enrollment's: every kept frame must score at least the
threshold against the fused probe, so a face swap during the window is refused
instead of averaged.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import tomllib

import numpy as np

from .quality import REASONS, QualityLimits


def fuse(embeddings) -> np.ndarray:
    """Return normalize(mean(normalize(e_i))) of an (n, D) array, n >= 1."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be a finite, non-empty (n, D) array")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("An embedding has zero length")
    mean = (matrix / lengths).mean(axis=0)
    length = np.linalg.norm(mean)
    # Opposite frames cancel out; such a set has no direction to match.
    if length < 1e-12:
        raise ValueError("The embeddings cancel out")
    return mean / length


@dataclass(frozen=True)
class Collecting:
    """Still collecting: kept good frames so far, frames needed, and this frame's problems."""

    kept: int
    needed: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Ready:
    """A fused probe and each kept frame's cosine against it."""

    probe: np.ndarray
    frames_used: int
    consistency: tuple[float, ...]


@dataclass(frozen=True)
class Retry:
    """No decision: 'timeout', 'inconsistent' or the most frequent quality problem."""

    reason: str


def _check_counts(k, min_frames, window_s):
    for name, value in (("k", k), ("min_frames", min_frames)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a whole number")
    if not 1 <= min_frames <= k:
        raise ValueError("Need 1 <= min_frames <= k")
    if isinstance(window_s, bool) or not isinstance(window_s, Real) or not isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be a positive number of seconds")


class ProbeCollector:
    """Collect up to k good frames within window_s seconds of the first frame.

    Call add() once per detected face frame. A frame with quality problems may pass
    None as its embedding, so the caller can skip the network for it. Call reset()
    when the face is lost or a new attempt starts. A Ready or Retry result also
    starts the next attempt from zero.
    """

    def __init__(self, *, tau, k=3, min_frames=2, window_s=2.0):
        _check_counts(k, min_frames, window_s)
        if isinstance(tau, bool) or not isinstance(tau, Real) or not isfinite(tau):
            raise ValueError("tau must be a finite number")
        self.tau, self.k, self.min_frames, self.window_s = float(tau), int(k), int(min_frames), float(window_s)
        self.reset()

    def reset(self):
        self._kept = []
        self._problems = Counter()
        self._start = None
        self._last = None

    def add(self, embedding, reasons, now):
        reasons = tuple(reasons)
        if self._last is not None and now < self._last:
            raise ValueError("Frame times must not go backwards")
        if self._start is None:
            self._start = now
        self._last = now
        if reasons:
            self._problems.update(reasons)
        elif embedding is None:
            raise ValueError("A frame that passes the quality gate needs its embedding")
        else:
            self._kept.append(np.asarray(embedding, dtype=np.float64))

        if len(self._kept) >= self.k:
            return self._finish()
        if now - self._start >= self.window_s:
            if len(self._kept) >= self.min_frames:
                return self._finish()
            return self._give_up()
        return Collecting(len(self._kept), self.k, reasons)

    def _finish(self):
        kept = np.stack(self._kept)
        self.reset()
        try:
            probe = fuse(kept)
        except ValueError:
            # Frames that cancel out cannot come from one steady face.
            return Retry("inconsistent")
        consistency = tuple(float(score) for score in kept @ probe / np.linalg.norm(kept, axis=1))
        if min(consistency) < self.tau:
            return Retry("inconsistent")
        return Ready(probe, len(consistency), consistency)

    def _give_up(self):
        problems = self._problems
        self.reset()
        if not problems:
            return Retry("timeout")
        # Most frequent problem; ties go to the more actionable one.
        reason = min(problems, key=lambda name: (-problems[name], REASONS.index(name)))
        return Retry(reason)


@dataclass(frozen=True)
class Match:
    """The schema's decision: label is None (unknown) unless the best score reaches tau."""

    label: str | None
    score: float | None
    accepted: bool


def best_match(probe, labels, templates, tau) -> Match:
    """Cosine of a unit probe against unit templates; accept the best if it is >= tau."""
    templates = np.asarray(templates, dtype=np.float64)
    labels = list(labels)
    if templates.ndim != 2 or len(templates) != len(labels):
        raise ValueError("Need one (D,) template per label")
    if not labels:
        return Match(None, None, False)
    # The model's raw embedding is not unit length; normalise so the score is a cosine.
    probe = np.asarray(probe, dtype=np.float64)
    scores = templates @ (probe / np.linalg.norm(probe))
    best = int(np.argmax(scores))
    score = float(scores[best])
    accepted = score >= tau
    return Match(labels[best] if accepted else None, score, accepted)


@dataclass(frozen=True)
class CheckInSettings:
    enabled: bool
    k: int
    min_frames: int
    window_s: float
    limits: QualityLimits | None


def load_check_in_settings(config_file) -> CheckInSettings:
    """Read the [check_in] table; limits are required once the feature is enabled."""
    with open(config_file, "rb") as handle:
        section = dict(tomllib.load(handle).get("check_in", {}))
    limits = section.pop("limits", None)
    expected = {"enabled", "k", "min_frames", "window_s"}
    if set(section) != expected:
        raise ValueError(f"Expected check_in keys: {', '.join(sorted(expected))} and optional limits")
    if not isinstance(section["enabled"], bool):
        raise ValueError("enabled must be true or false")
    _check_counts(section["k"], section["min_frames"], section["window_s"])
    limits = QualityLimits.from_mapping(limits) if limits is not None else None
    if section["enabled"] and limits is None:
        raise ValueError("The check-in needs calibrated [check_in.limits] before it can be enabled")
    return CheckInSettings(section["enabled"], section["k"], section["min_frames"],
                           float(section["window_s"]), limits)
