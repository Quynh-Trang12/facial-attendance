"""Require repeated live-frame decisions before allowing attendance.

By default, one tracked face must produce at least five consecutive live results
over at least one second, with no gap longer than 0.5 seconds. A spoof/uncertain
frame or excessive gap resets the count. Changing stream_id starts a new person.
"""

import math
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TemporalLivenessConfig:
    """Admission evidence required for one tracked person, independent of model training."""

    minimum_live_frames: int = 5
    minimum_live_seconds: float = 1.0
    maximum_gap_seconds: float = 0.5

    def validate(self):
        """Reject policies that cannot require a sustained sequence of live observations."""
        if (
            self.minimum_live_frames < 2
            or not math.isfinite(self.minimum_live_seconds)
            or self.minimum_live_seconds <= 0
            or not math.isfinite(self.maximum_gap_seconds)
            or self.maximum_gap_seconds <= 0
        ):
            raise ValueError("Invalid temporal liveness policy")


class TemporalLivenessGate:
    """Keep one current track's live evidence; do not share an instance across independent streams."""

    def __init__(self, config=TemporalLivenessConfig()):
        config.validate()
        self.config = config
        self.reset()

    def reset(self):
        """Discard the tracked identity, timestamps and accumulated live-frame evidence."""
        self.stream_id = self.last_time = self.started = None
        self.count = 0

    def observe(self, prediction, *, stream_id, timestamp=None):
        """Update the current person's live-frame count and attendance permission.

        prediction is the per-frame adapter result. timestamp is a monotonic time in
        seconds; the caller changes stream_id when the tracked person changes.
        """
        now = time.monotonic() if timestamp is None else timestamp
        if not stream_id or not math.isfinite(now):
            raise ValueError("A tracked stream ID and finite timestamp are required")

        if stream_id != self.stream_id:
            self.reset()
            self.stream_id = stream_id
        if self.last_time is not None and now < self.last_time:
            raise ValueError("Frame timestamps must be monotonic within a stream")

        gap = (
            self.last_time is not None
            and now - self.last_time > self.config.maximum_gap_seconds
        )
        self.last_time = now

        status = prediction.get("status")
        if status not in ("live", "spoof", "uncertain"):
            raise ValueError("Unknown per-frame liveness decision")
        score, threshold = prediction.get("spoof_score"), prediction.get("threshold")

        # A live label alone is insufficient: permission and the saved score
        # threshold must agree before this observation contributes evidence.
        valid_live = (
            status == "live"
            and prediction.get("allow_attendance") is True
            and score is not None
            and threshold is not None
            and math.isfinite(score)
            and math.isfinite(threshold)
            and 0 <= score < threshold <= 1
        )

        if gap or not valid_live:
            self.started, self.count = None, 0
        if valid_live:
            if self.started is None:
                self.started = now
            self.count += 1
        duration = 0.0 if self.started is None else now - self.started
        # Require elapsed time as well as frame count; a fast burst must not
        # satisfy a policy intended to observe the person for a full second.
        allowed = (
            valid_live
            and self.count >= self.config.minimum_live_frames
            and duration >= self.config.minimum_live_seconds
        )
        result = {
            **prediction,
            "frame_status": status,
            "status": "live"
            if allowed
            else "spoof"
            if status == "spoof"
            else "uncertain",
            "allow_attendance": bool(allowed),
            "consecutive_live_frames": self.count,
            "live_duration_seconds": duration,
            "temporal_policy": asdict(self.config),
        }
        if not allowed and status != "spoof":
            result["reason"] = prediction.get("reason") or "awaiting_stable_live"
        return result
