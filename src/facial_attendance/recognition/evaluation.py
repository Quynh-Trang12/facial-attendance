"""Consent-gated recognition research session, never production enrollment.

The only gallery is constructed empty here. No external gallery path, trained
liveness assertion, attendance writer or production controller is accepted.
"""

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..ui.diagnostic_snapshot import utc_timestamp
from ..ui.gallery import FaceGallery
from ..ui.recognition_adapter import TrainedRecognitionAdapter
from ..ui.webcam_prototype import (
    CONTRACT_VERSION,
    Context,
    FaceRequest,
    OperationError,
    Recognition,
    box_in_frame,
    frame_size,
)
from .contracts import FaceInferenceContract
from .gallery_matching import GalleryMatcher, empty_gallery
from .inference import cosine, normalized_vector

CONDITIONS = {
    "glasses": ("not_stated", "without", "with"),
    "lighting": ("normal_indoor", "backlight", "dim", "other"),
    "pose": ("front", "left", "right", "up", "down"),
    "distance": ("normal", "near", "far"),
    "presentation": ("live_person", "photograph"),
}


@dataclass(frozen=True)
class EvaluationJob:
    epoch: int
    serial: int
    kind: str
    source: tuple
    request: FaceRequest
    captured_at: float
    truth: str
    conditions: tuple


@dataclass(frozen=True)
class EvaluationResult:
    job: EvaluationJob
    fingerprint: str
    vector: object = None
    completion: object = None


@dataclass(frozen=True)
class SandboxDecision:
    test_id: str | None
    test_name: str | None
    score: float | None
    accepted: bool
    threshold: float
    can_save: bool = field(default=False, init=False)


def frame_fingerprint(frame):
    frame_size(frame)
    if not isinstance(frame, np.ndarray):
        raise ValueError("A real BGR snapshot is required")
    digest = sha256(str(frame.shape).encode())
    digest.update(frame.tobytes())
    return digest.hexdigest()


class RecognitionEvaluation:
    MIN_REFERENCES = 3
    TARGET_REFERENCES = 5
    MAX_REFERENCES = 10
    MAX_SUBJECTS = 10
    MAX_TRIALS = 500
    MAX_SAMPLE_AGE = 0.8

    def __init__(self, *, source_kind="webcam"):
        if source_kind not in ("webcam", "generated"):
            raise ValueError("Invalid sample source")
        self.source_kind = source_kind
        self.policy = FaceInferenceContract()
        self.matcher = GalleryMatcher(empty_gallery())
        self.epoch = 0
        self.reset()

    @property
    def can_save(self):
        return False

    @property
    def test_ids(self):
        return tuple(p.person_id for p in self.matcher._gallery.people)

    @property
    def next_test_id(self):
        return f"test-{len(self.test_ids) + 1:03d}"

    def reset(self):
        self.epoch += 1
        self.run_id = uuid4().hex
        self.consent = False
        with self.matcher._lock:
            self.matcher._gallery = empty_gallery()
        self.pending = None
        self.serial = 0
        self.references = []
        self.reference_conditions = []
        self.reference_hashes = set()
        self.evaluation_hashes = set()
        self.records = []
        self.subjects = []
        self.reload_checks = 0

    def set_consent(self, value):
        self.consent = value is True
        if not self.consent:
            self.pending = None

    def cancel(self):
        self.pending = None

    def discard_draft(self):
        self.pending = None
        self.references.clear()
        self.reference_conditions.clear()

    def job(
        self,
        kind,
        source,
        frame,
        boxes,
        captured_at,
        *,
        now,
        truth="unspecified",
        conditions=None,
        registration_context=None,
    ):
        if not self.consent:
            raise PermissionError(
                "Informed consent is required before collecting samples."
            )
        if self.pending is not None:
            raise ValueError("One sandbox sample is already processing.")
        if kind not in ("preview", "reference", "evaluation"):
            raise ValueError("Unknown sandbox operation")
        if not 0 <= now - captured_at <= self.MAX_SAMPLE_AGE:
            raise ValueError("Sample expired; wait for a fresh face.")
        width, height = frame_size(frame)
        if len(boxes) != 1 or not box_in_frame(boxes[0], width, height):
            raise ValueError("Exactly one valid detected face is required.")
        if kind == "reference" and (
            len(self.references) >= self.MAX_REFERENCES
            or len(self.test_ids) >= self.MAX_SUBJECTS
        ):
            raise ValueError("Reference/session limit reached; commit or reset.")
        if kind == "reference" and len(self.reference_hashes) >= 1000:
            raise ValueError("Reference history limit reached; reset the sandbox.")
        if kind == "evaluation" and len(self.records) >= self.MAX_TRIALS:
            raise ValueError("Trial limit reached; export and reset.")
        if truth not in ("unspecified", "unknown", *self.test_ids):
            raise ValueError("Choose a sandbox ground-truth test ID.")
        conditions = conditions or {k: v[0] for k, v in CONDITIONS.items()}
        if set(conditions) != set(CONDITIONS) or any(
            conditions[k] not in v for k, v in CONDITIONS.items()
        ):
            raise ValueError("Use the supplied non-identifying condition choices.")
        self.serial += 1
        context = registration_context
        if context is None:
            # Sandbox callers use the schema only, with no attendance service.
            context = Context(
                CONTRACT_VERSION, self.run_id, "attendance", uuid4().hex, uuid4().hex
            )
        elif kind != "reference" or context.action != "registration":
            raise ValueError("A registration context requires a reference capture.")
        request = FaceRequest(context, tuple(boxes[0]), frame)
        self.pending = EvaluationJob(
            self.epoch,
            self.serial,
            kind,
            source,
            request,
            captured_at,
            truth,
            tuple(conditions.items()),
        )
        return self.pending

    def accept(self, result, *, current_source, now):
        job = result.job
        if job is not self.pending or job.epoch != self.epoch:
            raise ValueError("Stale sandbox result discarded.")
        self.pending = None
        if (
            not self.consent
            or current_source != job.source
            or not 0 <= now - job.captured_at <= self.MAX_SAMPLE_AGE
        ):
            raise ValueError("Scene/capture changed or sample expired; capture again.")
        key = result.fingerprint
        if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
            raise ValueError("Invalid sample fingerprint")
        if job.kind == "reference":
            if key in self.reference_hashes or key in self.evaluation_hashes:
                raise ValueError(
                    "This exact frame was already used; capture a distinct reference."
                )
            self.policy.validate_normalized_embedding(result.vector)
            self.references.append(tuple(result.vector))
            self.reference_hashes.add(key)
            self.reference_conditions.append(dict(job.conditions))
            return None
        if key in self.reference_hashes:
            raise ValueError(
                "Reference frame excluded: wait for a distinct evaluation capture."
            )
        completion = result.completion
        if completion is None or completion.context != job.request.context:
            raise ValueError("Mismatched result context")
        rec = completion.result
        if isinstance(rec, OperationError):
            raise ValueError(f"Recognition failed ({rec.code}); no trial recorded.")
        if (
            not isinstance(rec, Recognition)
            or rec.source != "model"
            or rec.model_id != self.policy.model_id
            or rec.model_version != self.policy.model_version
            or type(rec.accepted) is not bool
        ):
            raise ValueError(
                "Recognition model/result does not match the sandbox contract."
            )
        if not rec.accepted and (rec.person_id is not None or rec.name is not None):
            raise ValueError("An unknown recognition result must carry no identity.")
        if rec.score is None:
            if rec.accepted or self.test_ids:
                raise ValueError("Invalid empty-gallery decision")
        else:
            meets_threshold = self.policy.accepts_cosine(rec.score)
            if rec.accepted and not meets_threshold:
                raise ValueError("Recognition accepted a score below the threshold.")
        if rec.accepted and rec.person_id not in self.test_ids:
            raise ValueError("Result is not from the isolated gallery.")
        decision = SandboxDecision(
            rec.person_id,
            rec.name,
            rec.score,
            rec.accepted,
            self.policy.cosine_threshold,
        )
        if job.kind == "evaluation":
            if key in self.evaluation_hashes:
                raise ValueError("This evaluation frame was already recorded.")
            self.evaluation_hashes.add(key)
            outcome = (
                "unscored"
                if job.truth == "unspecified"
                else "correct_reject"
                if job.truth == "unknown" and not rec.accepted
                else "false_accept"
                if job.truth == "unknown"
                else "false_reject"
                if not rec.accepted
                else "correct_accept"
                if rec.person_id == job.truth
                else "false_accept"
            )
            self.records.append(
                {
                    "trial": len(self.records) + 1,
                    "observed_utc": utc_timestamp(),
                    "ground_truth": job.truth,
                    "predicted_test_id": rec.person_id,
                    "cosine": rec.score,
                    "threshold": self.policy.cosine_threshold,
                    "decision": "Known" if rec.accepted else "Unknown",
                    "outcome": outcome,
                    "wrong_identity": bool(
                        rec.accepted
                        and job.truth in self.test_ids
                        and rec.person_id != job.truth
                    ),
                    **dict(job.conditions),
                    "photo_accepted": bool(
                        rec.accepted
                        and dict(job.conditions)["presentation"] == "photograph"
                    ),
                }
            )
        return decision

    def commit_reference(self, *, display_name=None, save_gallery=None):
        if not self.consent:
            raise PermissionError("Informed consent is required.")
        if self.pending is not None:
            raise ValueError("Wait for the active sample.")
        if not self.MIN_REFERENCES <= len(self.references) <= self.MAX_REFERENCES:
            raise ValueError("Collect 3–10 distinct references; target five.")
        if display_name is not None and (
            not isinstance(display_name, str)
            or not 1 <= len(display_name) <= 24
            or not display_name.isprintable()
        ):
            raise ValueError("Use a single-line display name of up to 24 characters.")
        template = normalized_vector(np.mean(self.references, axis=0))
        if any(
            not self.policy.accepts_cosine(cosine(v, template)) for v in self.references
        ):
            raise ValueError(
                "References fail the existing consistency rule; discard and recapture."
            )
        with self.matcher._lock:
            match = self.matcher.match_threshold_only(template)
            if match.accepted:
                if save_gallery is not None:
                    raise ValueError(
                        f"Face already matches the local profile “{match.name}”."
                    )
                raise ValueError(
                    "Reference matches an existing test subject; no duplicate enrollment."
                )
            test_id = self.next_test_id
            current = self.matcher._gallery
            gallery = (
                FaceGallery.from_dict(current.to_dict())
                if save_gallery is not None
                else current
            )
            gallery.register(
                test_id,
                display_name or f"Test subject {test_id[-3:]}",
                template,
                n_images=len(self.references),
            )
            if save_gallery is not None:
                save_gallery(gallery)
                self.matcher._gallery = gallery
        self.subjects.append(
            {
                "test_id": test_id,
                "reference_count": len(self.references),
                "reference_conditions": list(self.reference_conditions),
            }
        )
        self.references = []
        self.reference_conditions = []
        self.epoch += 1
        self.consent = False
        return test_id

    def reload_in_memory(self):
        if self.pending is not None:
            raise ValueError("Wait for the active sample.")
        with self.matcher._lock:
            clone = FaceGallery.from_dict(self.matcher._gallery.to_dict())
            GalleryMatcher(
                clone
            )  # Same model/version/dimension checks as normal gallery matching.
            self.matcher._gallery = clone
        self.epoch += 1
        self.reload_checks += 1

    def report(self):
        def counts(rows):
            tally = Counter(r["outcome"] for r in rows)
            return {
                k: tally[k]
                for k in (
                    "correct_accept",
                    "false_reject",
                    "correct_reject",
                    "false_accept",
                    "unscored",
                )
            }

        return {
            "kind": "Recognition Evaluation / SANDBOX",
            "source": self.source_kind,
            "physical_samples_recorded": self.source_kind == "webcam"
            and bool(self.records),
            "model_id": self.policy.model_id,
            "model_version": self.policy.model_version,
            "embedding_dimensions": self.policy.embedding_dim,
            "threshold": self.policy.cosine_threshold,
            "recognition_only": True,
            "liveness_evaluated": False,
            "can_save": False,
            "subjects": list(self.subjects),
            "counts": counts(self.records),
            "live_person_counts": counts(
                [r for r in self.records if r["presentation"] == "live_person"]
            ),
            "photograph_counts": counts(
                [r for r in self.records if r["presentation"] == "photograph"]
            ),
            "photo_accepts": sum(r["photo_accepted"] for r in self.records),
            "reload_checks": self.reload_checks,
            "trials": list(self.records),
            "limitations": "Counts only; small/correlated samples do not establish reliable accuracy or spoof resistance.",
        }


class SandboxInference:
    def __init__(self, inference, session):
        policy = session.policy
        meta = inference.provenance
        if (
            inference.contract != policy
            or not isinstance(meta, dict)
            or meta.get("model_id") != policy.model_id
            or meta.get("model_version") != policy.model_version
            or meta.get("threshold") != policy.cosine_threshold
        ):
            raise ValueError(
                "Sandbox requires the unchanged selected model and policy."
            )
        self.inference = inference
        self.adapter = TrainedRecognitionAdapter(inference, session.matcher)

    def __call__(self, job):
        key = frame_fingerprint(job.request.frame)
        if job.kind == "reference":
            return EvaluationResult(
                job,
                key,
                vector=self.inference.embed(job.request.frame, job.request.bbox),
            )
        return EvaluationResult(job, key, completion=self.adapter(job.request))


class SandboxReports:
    """Only score metadata under a fixed ignored sandbox root; never a gallery."""

    def __init__(self, repository):
        self.base = Path(repository) / "artifacts" / "recognition-evaluation"

    def _directory(self, run_id):
        if not re.fullmatch("[0-9a-f]{32}", run_id):
            raise ValueError("Invalid sandbox run")
        root = self.base.parent
        if root.is_symlink() or self.base.is_symlink():
            raise ValueError("Refusing a linked sandbox output root")
        target = self.base / run_id
        if target.is_symlink() or not target.resolve().is_relative_to(
            self.base.resolve()
        ):
            raise ValueError("Unsafe sandbox destination")
        return target

    def export(self, session):
        directory = self._directory(session.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "scores.json"
        if path.is_symlink():
            raise ValueError("Refusing linked report")
        fd, temporary = tempfile.mkstemp(prefix=".scores-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(session.report(), indent=2, allow_nan=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary, path
            )  # Replace our entry; do not overwrite a hard-linked external file.
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def delete(self, run_id):
        directory = self._directory(run_id)
        path = directory / "scores.json"
        if path.is_symlink():
            raise ValueError("Refusing linked report")
        path.unlink(missing_ok=True)
        if directory.exists():
            directory.rmdir()  # Never recursively delete unexpected files.

    def delete_all(self):
        self._directory("0" * 32)  # Validate the fixed root even if no files exist.
        if not self.base.exists():
            return
        runs = []
        for directory in self.base.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Unexpected sandbox file")
            self._directory(directory.name)
            entries = list(directory.iterdir())
            if any(p.name != "scores.json" or p.is_symlink() for p in entries):
                raise ValueError("Unexpected sandbox content")
            if entries:
                payload = json.loads(entries[0].read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("kind") != "Recognition Evaluation / SANDBOX"
                ):
                    raise ValueError("Not a sandbox score log")
            runs.append(directory.name)
        for run_id in runs:
            self.delete(run_id)
