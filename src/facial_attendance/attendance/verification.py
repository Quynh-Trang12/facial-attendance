"""Candidate attendance verification on the existing current-frame model pipeline."""

from dataclasses import dataclass
from math import isfinite
from time import monotonic
from uuid import uuid4

from ..liveness.temporal_liveness import TemporalLivenessGate
from ..ui.emotion_presentation import geometry
from ..ui.trial_profiles import REGISTRATION_PREPROCESSING, REGISTRATION_RELEASE
from ..ui.webcam_prototype import (
    CONTRACT_VERSION, Completion, Context, FaceRequest, Liveness, OperationError, Recognition,
    box_in_frame, frame_size,
)
from .service import AttendanceService, VerificationPolicy


DECISION_POLICY = "aenet-full-5-live-1s-gap-0.5s-attendance-candidate-v1"


@dataclass(frozen=True)
class _Capture:
    token: tuple[int, int]
    request: FaceRequest
    captured_at: float


class AttendanceVerification:
    """Consume fresh, original-context results on the Tk thread. Never use preview summaries."""

    def __init__(self, store, release, recognition, registered_name, *,
                 max_age_seconds, clock=monotonic):
        if any(release.get(key) != value for key, value in REGISTRATION_RELEASE.items()):
            raise ValueError("The candidate requires the pinned AENet-Full release.")
        self._policy = VerificationPolicy(release["model_version"], REGISTRATION_PREPROCESSING, DECISION_POLICY,
                                          recognition["checkpoint_sha256"], recognition["preprocessing_version"],
                                          max_age_seconds)
        self.service = AttendanceService(store, policy=self._policy, registered_name=registered_name, clock=clock)
        self._clock = clock
        self._gate = TemporalLivenessGate()
        self._session_id = None
        self._last_token = None
        self._clear_attempt()

    def _clear_attempt(self):
        self._gate.reset()
        self._expected = None
        self._attempt_id = uuid4().hex
        self._started_at = self._clock()
        self._person_id = None
        self._region = None

    def sync_session(self):
        session_id = self.service.session_id
        if session_id != self._session_id:
            self._session_id = session_id
            self._clear_attempt()

    def reset(self):
        """An explicit camera retry starts a fresh attempt, including after spoof."""
        self._clear_attempt()
        self.service.reset()

    def invalidate(self, code="verification_pending"):
        self._clear_attempt()
        self.service.invalidate_capture(code)

    def pending_capture(self):
        """A newer capture supersedes permission, while keeping the live streak."""
        self._expected = None
        if self.service.state.can_record:
            self.service.invalidate_capture("verification_pending")

    def expect(self, token, request, captured_at):
        """Register just the current detector result whose existing inference job was submitted."""
        if not self._session_id or self.service.state.code == "spoof":
            return
        if self._last_token is not None and token <= self._last_token:
            return
        self._last_token = token
        self.pending_capture()
        if not isinstance(request, FaceRequest) or not self._valid_context(request.context):
            self.invalidate("invalid_verification")
            return
        try:
            size = frame_size(request.frame)
        except ValueError:
            self.invalidate("capture_error")
            return
        if not box_in_frame(request.bbox, *size):
            self.invalidate("capture_error")
            return
        if not self._fresh(captured_at):
            self.invalidate("stale")
            return

        if self._region is not None:
            generation, box, dimensions, previous_at = self._region
            gap = captured_at - previous_at
            if (token[0] != generation or size != dimensions or not 0 < gap <= self._gate.config.maximum_gap_seconds
                    or geometry(box, request.bbox)[0] < .20):
                self.invalidate()
                self._started_at = captured_at

        self._expected = _Capture(token, request, captured_at)

    def accept(self, token, request, results, prediction, *, error=None):
        """Join the liveness prediction and recognition from this one worker capture."""
        self.sync_session()
        expected = self._expected
        if expected is None or token != expected.token:
            return False
        self._expected = None
        if request is not expected.request or not self._fresh(expected.captured_at):
            self.invalidate("stale")
            return False
        if error is not None or prediction is None:
            self.invalidate("inference_error")
            return False
        if not self._valid_prediction(prediction):
            self.invalidate("invalid_verification")
            return False

        status = prediction["status"]
        if status != "live":
            self.invalidate(status)
            return False

        identity = self._recognition_result(results, request.context)
        match = self.service.recognition_status(identity)
        if match != "known":
            self.invalidate(match)
            return False
        if self._person_id is not None and identity.person_id != self._person_id:
            self.invalidate()
            self._started_at = expected.captured_at
        self._person_id = identity.person_id
        self._region = (token[0], request.bbox, frame_size(request.frame), expected.captured_at)
        gated = self._gate.observe(prediction, stream_id=self._attempt_id, timestamp=expected.captured_at)
        if not gated["allow_attendance"]:
            self.service.invalidate_capture("verification_pending")
            return False

        # The original capture and face IDs come from the exact request above.
        # The dedicated attendance attempt spans the gate's consecutive samples.
        source = request.context
        context = Context(CONTRACT_VERSION, self._attempt_id, "attendance", source.capture_id, source.face_id)
        sample = self.service.begin_capture(request.frame, [request.bbox], expected.captured_at, context=context)
        if sample is None:
            return False
        evidence = Liveness("live", "accepted_live", gated["spoof_score"], self._policy.liveness_model,
                            self._policy.liveness_preprocessing, self._policy.liveness_decision, "model")
        return self.service.complete(Completion(context, identity), Completion(context, evidence))

    def _fresh(self, captured_at):
        return (type(captured_at) in (int, float) and self._started_at <= captured_at
                and 0 <= self._clock() - captured_at <= self._policy.max_age_seconds)

    @staticmethod
    def _valid_context(context):
        return (isinstance(context, Context) and context.contract_version == CONTRACT_VERSION
                and context.action == "attendance"
                and all(isinstance(value, str) and value.strip()
                        for value in (context.attempt_id, context.capture_id, context.face_id)))

    @staticmethod
    def _valid_prediction(result):
        if (not isinstance(result, dict) or result.get("source", "model") != "model"
                or any(result.get(key) != REGISTRATION_RELEASE[key] for key in
                       ("model_id", "model_version", "threshold", "precision"))):
            return False
        status, score, allow = result.get("status"), result.get("spoof_score"), result.get("allow_attendance")
        threshold = REGISTRATION_RELEASE["threshold"]
        if score is None:
            return status == "uncertain" and allow is False
        if type(score) not in (float, int) or not 0 <= score <= 1 or not isfinite(score):
            return False
        if score >= threshold:
            return status == "spoof" and allow is False
        return (status == "live" and allow is True) or (status == "uncertain" and allow is False)

    @staticmethod
    def _recognition_result(batch, context):
        if not isinstance(batch, (tuple, list)):
            return None
        matches = [item for item in batch if isinstance(item, Completion)
                   and (isinstance(item.result, Recognition) or
                        (isinstance(item.result, OperationError) and item.result.module == "recognition"))]
        if len(matches) != 1 or matches[0].context != context:
            return None
        return matches[0].result
