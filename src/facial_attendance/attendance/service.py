"""Serialize dedicated attendance capture, verification, reset and save commands."""

from collections.abc import Callable
from dataclasses import dataclass
import re
import sqlite3
from threading import Lock
from time import monotonic
from uuid import uuid4

from ..recognition.contracts import FaceInferenceContract
from ..ui.webcam_prototype import (
    CONTRACT_VERSION, Completion, Context, FaceRequest, Liveness, OperationError, Recognition,
    box_in_frame, frame_size,
)
from .store import AttendanceRecord


@dataclass(frozen=True)
class VerificationPolicy:
    """Pins from a reviewed per-capture adapter and the selected recognition release."""

    liveness_model: str
    liveness_preprocessing: str
    liveness_decision: str
    recognition_checkpoint: str
    recognition_preprocessing: str
    max_age_seconds: float

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip() for value in
                (self.liveness_model, self.liveness_preprocessing, self.liveness_decision,
                 self.recognition_preprocessing))
                or not isinstance(self.recognition_checkpoint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.recognition_checkpoint)
                or type(self.max_age_seconds) not in (float, int)
                or not 0 < self.max_age_seconds < float("inf")):
            raise ValueError("Attendance requires complete approved model versions and a finite freshness limit.")


@dataclass(frozen=True)
class AttendanceState:
    code: str
    verified_name: str | None = None
    record: AttendanceRecord | None = None

    @property
    def can_record(self):
        return self.code == "ready"


class AttendanceService:
    def __init__(self, store, *, policy: VerificationPolicy | None = None,
                 registered_name: Callable[[str], str | None] | None = None, clock=monotonic):
        if (policy is None) != (registered_name is None):
            raise ValueError("Attendance requires both an approved verification policy and a gallery lookup.")
        self._store = store
        self._policy = policy
        self._registered_name = registered_name
        self._clock = clock
        self._recognition_contract = FaceInferenceContract()
        self._lock = Lock()
        self._session_id = None
        self._clear("session_inactive")

    @property
    def session_id(self):
        with self._lock:
            return self._session_id

    @property
    def state(self):
        with self._lock:
            self._expire()
            return self._state

    def start_session(self):
        with self._lock:
            if self._session_id is None:
                self._session_id = uuid4().hex
                self._clear(self._waiting_status())
            return self._session_id

    def end_session(self):
        with self._lock:
            self._session_id = None
            self._clear("session_inactive")

    def reset(self):
        """Call on explicit retry, operation change or camera/gallery invalidation."""
        with self._lock:
            self._clear(self._waiting_status() if self._session_id else "session_inactive")

    def invalidate_capture(self, code):
        """Revoke current evidence while preserving a spoof abort until explicit retry."""
        if code not in {"verification_pending", "invalid_verification", "capture_error", "stale",
                        "inference_error", "model_unavailable", "no_face", "multiple_faces",
                        "unknown", "spoof", "uncertain"}:
            raise ValueError("Invalid attendance rejection state.")
        with self._lock:
            if self._session_id and self._state.code != "spoof":
                self._clear(code)

    def recognition_status(self, result):
        with self._lock:
            return self._recognition_status(result) if self._policy is not None else "model_unavailable"

    def begin_capture(self, frame, boxes, captured_at, *, context=None):
        """Start a dedicated attempt and return its original-source model request."""
        with self._lock:
            if not self._session_id or self._policy is None or self._state.code == "spoof":
                return None
            self._clear("verification_pending")
            try:
                width, height = frame_size(frame)
            except ValueError:
                self._clear("capture_error")
                return None
            if not isinstance(boxes, (tuple, list)):
                self._clear("capture_error")
                return None
            if len(boxes) != 1:
                self._clear("multiple_faces" if boxes else "no_face")
                return None
            if not box_in_frame(boxes[0], width, height):
                self._clear("capture_error")
                return None
            if not self._fresh(captured_at):
                self._clear("stale")
                return None
            if context is None:
                context = Context(CONTRACT_VERSION, uuid4().hex, "attendance", uuid4().hex, uuid4().hex)
            elif (not isinstance(context, Context) or context.contract_version != CONTRACT_VERSION
                    or context.action != "attendance"
                    or not all(isinstance(value, str) and value.strip()
                               for value in (context.attempt_id, context.capture_id, context.face_id))):
                self._clear("invalid_verification")
                return None
            self._captured_at = captured_at
            self._context = context
            return FaceRequest(self._context, tuple(int(value) for value in boxes[0]), frame)

    def complete(self, recognition: Completion, liveness: Completion):
        """Apply one pair of results. An obsolete completion leaves current work intact."""
        with self._lock:
            self._expire()
            if self._context is None or self._state.code != "verification_pending":
                return False
            if not isinstance(recognition, Completion) or not isinstance(liveness, Completion):
                self._clear("invalid_verification")
                return False
            if recognition.context != self._context or liveness.context != self._context:
                return False

            live_status = self._liveness_status(liveness.result)
            if live_status != "live":
                self._clear(live_status)
                return False

            identity = recognition.result
            match_status = self._recognition_status(identity)
            if match_status != "known":
                self._clear(match_status)
                return False
            self._identity = identity
            self._liveness = liveness.result
            self._state = AttendanceState("ready", identity.name)
            return True

    def record(self):
        """Recheck the live attempt and serialize the write with every reset and session change."""
        with self._lock:
            self._expire()
            if self._state.code == "recorded":
                return AttendanceState("already_recorded")
            if not self._state.can_record:
                return self._state
            identity = self._identity
            if (not self._session_id or self._context is None or self._policy is None or identity is None
                    or self._liveness_status(self._liveness) != "live"):
                self._clear("invalid_verification")
                return self._state
            match_status = self._recognition_status(identity)
            if match_status != "known":
                self._clear(match_status)
                return self._state
            self._expire()
            if not self._state.can_record:
                return self._state

            # Hold the lock through the short, fail-fast SQLite transaction.
            # A reset can revoke unused authorization or follow a finished save.
            self._state = AttendanceState("saving")
            try:
                record = self._store._insert(self._session_id, identity.person_id, identity.name)
            except (OSError, sqlite3.Error):
                self._clear("save_failed")
            else:
                code = "recorded" if record is not None else "already_recorded"
                self._clear(code)
                self._state = AttendanceState(code, record=record)
            return self._state

    def _waiting_status(self):
        return "model_unavailable" if self._policy is None else "verification_pending"

    def _clear(self, code):
        self._context = None
        self._identity = None
        self._liveness = None
        self._captured_at = None
        self._state = AttendanceState(code)

    def _fresh(self, captured_at):
        if type(captured_at) not in (float, int):
            return False
        try:
            return 0 <= self._clock() - captured_at <= self._policy.max_age_seconds
        except OverflowError:
            return False

    def _expire(self):
        if self._context is not None and not self._fresh(self._captured_at):
            self._clear("stale")

    def _liveness_status(self, result):
        if isinstance(result, OperationError):
            return self._error_status(result)
        if not isinstance(result, Liveness):
            return "invalid_verification"
        versions = (result.model_version, result.preprocessing_version, result.decision_policy_version)
        expected = (self._policy.liveness_model, self._policy.liveness_preprocessing, self._policy.liveness_decision)
        if (not isinstance(result.source, str) or result.source != "model"
                or any(not isinstance(value, str) for value in versions) or versions != expected
                or not isinstance(result.status, str) or result.status not in ("live", "spoof", "uncertain")
                or not isinstance(result.reason, str) or not result.reason.strip()):
            return "invalid_verification"
        if result.status == "uncertain" and result.p_spoof is None:
            return "uncertain"
        if type(result.p_spoof) not in (float, int) or not 0 <= result.p_spoof <= 1:
            return "invalid_verification"
        if result.status == "live" and result.reason != "accepted_live":
            return "invalid_verification"
        return result.status

    @staticmethod
    def _error_status(result):
        if result.module in ("capture", "camera"):
            return "capture_error"
        return "model_unavailable" if result.code == "model_unavailable" else "inference_error"

    def _recognition_status(self, result):
        if isinstance(result, OperationError):
            return self._error_status(result)
        if not isinstance(result, Recognition):
            return "invalid_verification"
        versions = (result.model_id, result.model_version, result.checkpoint_sha256, result.preprocessing_version)
        expected = (self._recognition_contract.model_id, self._recognition_contract.model_version,
                    self._policy.recognition_checkpoint, self._policy.recognition_preprocessing)
        if (not isinstance(result.source, str) or result.source != "model" or type(result.accepted) is not bool
                or any(not isinstance(value, str) for value in versions) or versions != expected):
            return "invalid_verification"
        if not result.accepted and result.person_id is None and result.name is None and result.score is None:
            return "unknown"
        try:
            accepted = self._recognition_contract.accepts_cosine(result.score)
        except (ValueError, OverflowError):
            return "invalid_verification"
        if accepted != result.accepted:
            return "invalid_verification"
        if not accepted:
            return "unknown" if result.person_id is None and result.name is None else "invalid_verification"
        if any(not isinstance(value, str) or not value.strip() or value != value.strip()
               for value in (result.person_id, result.name)):
            return "invalid_verification"
        try:
            registered_name = self._registered_name(result.person_id)
        except (LookupError, OSError, sqlite3.Error, ValueError):
            return "inference_error"
        return "known" if registered_name == result.name else "unknown"
