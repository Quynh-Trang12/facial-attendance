"""Diagnostic-only immutable completion reference and allowlisted JSON export.

No snapshot is an authorization input. Export is built afresh from known scalar
fields, never by dumping an envelope, gallery, exception or arbitrary metadata.
"""

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ..emotion.model_contracts import DIAGNOSTIC_LABELS


def utc_timestamp():
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class CompletedDiagnostic:
    state: object  # Frozen SYS-02 Envelope; contains no image or embedding.
    timestamp: str


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def choice(value, choices):
    return value if isinstance(value, str) and value in choices else None


def boolean(value):
    return value if type(value) is bool else None


def digest(value):
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{12}|[0-9a-f]{64}", value)
        else None
    )


def prediction(value, module):
    value = value if isinstance(value, dict) else {}
    result = {"source": choice(value.get("source"), ("stub", "model"))}
    if module == "recognition":
        result.update(
            accepted=boolean(value.get("accepted")), score=number(value.get("score"))
        )
    elif module == "emotion":
        result.update(
            class_id=number(value.get("class_id")),
            label=choice(value.get("label"), DIAGNOSTIC_LABELS),
            confidence=number(value.get("confidence")),
        )
    else:
        result.update(
            status=choice(value.get("status"), ("live", "spoof", "uncertain")),
            p_spoof=number(value.get("p_spoof")),
        )
    return result


def privacy_safe_snapshot(data, metadata, *, paused=False):
    """Positive allowlist: no arbitrary text, names, IDs, paths or arrays pass."""
    if not isinstance(data, dict):
        raise ValueError("Displayed snapshot must be a JSON object")
    is_raw = "ui_state" in data
    raw = data if is_raw else {}
    recognition = data.get("recognition", {}) if not is_raw else {}
    emotion = data.get("emotion", {}) if not is_raw else {}
    liveness = data.get("liveness", {}) if not is_raw else {}
    runtime = data.get("runtime", {}) if not is_raw else {}
    authorization = data.get("authorization", {}) if not is_raw else raw
    sections = (recognition, emotion, liveness, runtime, authorization)
    if not all(isinstance(v, dict) for v in sections):
        raise ValueError("Invalid displayed diagnostic sections")
    try:
        timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
        timestamp = (
            timestamp.isoformat(timespec="milliseconds")
            if timestamp.tzinfo is not None
            else None
        )
    except (TypeError, ValueError):
        timestamp = None
    source = choice(
        metadata.get("source"), ("runtime", "current_sys02", "completed_inference")
    )
    report = {
        "schema": "privacy-safe-diagnostics-v1",
        "snapshot": {
            "timestamp": timestamp,
            "source": source,
            "paused": bool(paused),
            "historical": bool(paused or source == "completed_inference"),
            "diagnostic_only": True,
            "note": "Historical diagnostics never authorize an operation.",
        },
        "runtime": {
            "camera_index": number(runtime.get("camera_index")),
            "face_count": number(runtime.get("face_count")),
            "face_detector": {
                "yunet (CPU)": "yunet_cpu",
                "haar (CPU)": "haar_cpu",
                "YuNet (AENet-Full release, CPU)": "yunet_cpu",
                "controlled regions": "controlled_test",
                "Controlled regions (test only)": "controlled_test",
            }.get(runtime.get("face_detector"))
            if isinstance(runtime.get("face_detector"), str)
            else None,
            "processing_device": choice(
                runtime.get("processing_device"), ("CPU", "CUDA", "MPS")
            ),
            "camera_status": choice(
                runtime.get("camera_status"),
                ("unavailable", "delivering frames", "opening / awaiting frame"),
            ),
            "detector_busy": boolean(runtime.get("detector_busy")),
            "inference_busy": boolean(runtime.get("inference_busy")),
        },
        "recognition": prediction(
            raw.get("recognition") if is_raw else recognition.get("raw"), "recognition"
        ),
        "emotion": prediction(
            raw.get("emotion") if is_raw else emotion.get("raw_current_prediction"),
            "emotion",
        ),
        "liveness": prediction(
            raw.get("liveness") if is_raw else liveness.get("raw"), "liveness"
        ),
        "authorization": {
            "can_save": boolean(authorization.get("can_save")),
            "genuine_registration_available": False,
            "attendance_available": False,
            "liveness_is_simulated": authorization.get("liveness_is_simulated")
            is not False
            and metadata.get("liveness_is_simulated") is not False
            and raw.get("ui_state") != "preview",
            "identity_verified": False,
        },
    }
    for module, section in (("recognition", recognition), ("emotion", emotion)):
        checkpoint = (raw.get(module) if is_raw else section.get("checkpoint")) or {}
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        report[module]["checkpoint_sha256"] = digest(
            checkpoint.get("checkpoint_sha256")
        )
        report[module]["model_version"] = digest(checkpoint.get("model_version"))
    report["recognition"]["configured_threshold"] = number(
        recognition.get("configured_threshold")
    )
    report["recognition"]["embedding_dimensions"] = number(
        recognition.get("embedding_dimensions")
    )
    report["emotion"].update(
        displayed_label=choice(
            emotion.get("displayed_label"), (*DIAGNOSTIC_LABELS, "Unavailable")
        ),
        display_age_ms=number(emotion.get("display_age_ms")),
        prediction_interval_s=number(emotion.get("prediction_interval_s")),
        display_limit_ms=number(emotion.get("display_limit_ms")),
    )
    report["emotion"]["display_reason"] = choice(
        emotion.get("display_reason"),
        (
            "latest_valid_sample",
            "detector_gap_grace",
            "startup",
            "retry",
            "multiple_faces",
            "no_face",
            "sustained_face_loss",
            "face_evidence_gap",
            "ambiguous_box_jump",
            "ambiguous_replacement",
            "freshness_expired",
            "camera_failure",
            "camera_frame_expired",
            "detector_failure",
            "invalid_prediction",
            "spoof",
            "model_error_model_unavailable",
            "model_error_inference_error",
            "model_error_invalid_input",
            "model_error_version_mismatch",
        ),
    )
    if liveness.get("active_source") == "AENet-Full preview":
        preview = liveness.get("released_preview") or {}
        scan = liveness.get("last_completed_scan") or {}
        checkpoint = liveness.get("trained_model_metadata") or {}
        if not all(
            isinstance(section, dict) for section in (preview, scan, checkpoint)
        ):
            raise ValueError("Invalid displayed AENet diagnostics")
        report["liveness"]["released_preview"] = {
            "checkpoint_sha256": digest(checkpoint.get("checkpoint_sha256")),
            "model_version": choice(
                checkpoint.get("model_version"), ("f16fbfaf36046f75",)
            ),
            "epoch": number(checkpoint.get("epoch")),
            "threshold": number(liveness.get("trained_threshold")),
            "frame_status": choice(
                preview.get("frame_status"), ("live", "spoof", "uncertain", "error")
            ),
            "spoof_score": number(preview.get("spoof_score")),
        }
        report["liveness"]["last_completed_scan"] = {
            "decision": choice(scan.get("decision"), ("live", "spoof", "uncertain")),
            "mean_spoof_score": number(scan.get("mean_spoof_score")),
            "valid_frames": number(scan.get("valid_frames")),
        }
    trial = liveness.get("experimental_comparison")
    if isinstance(trial, dict):
        scores = trial.get("scores")
        report["liveness"]["experimental_comparison"] = {
            "method": choice(
                trial.get("method"), ("experimental-softmax-spoof-index-1",)
            ),
            "status": choice(trial.get("status"), ("ready", "unavailable", "error")),
            "can_authorize": False,
            "scores": [
                {
                    "variant": choice(score.get("variant"), ("Full", "Baseline")),
                    "model_version": choice(
                        score.get("model_version"),
                        ("ee6f1341785d22ad", "97208d8b15f8ac34"),
                    ),
                    "score": number(score.get("score")),
                    "threshold": number(score.get("threshold")),
                    "threshold_met": boolean(score.get("threshold_met")),
                }
                for score in (scores[:2] if isinstance(scores, list) else [])
                if isinstance(score, dict)
            ],
        }
    presentation = data.get("presentation", {})
    state = (
        raw.get("ui_state")
        if is_raw
        else (
            presentation.get("authoritative_state")
            if isinstance(presentation, dict)
            else None
        )
    )
    report["authoritative_state_at_snapshot"] = choice(
        state,
        (
            "idle",
            "checking",
            "live",
            "preview",
            "spoof",
            "uncertain",
            "error",
            "no_face",
            "multiple_faces",
        ),
    )
    # Measured timing keys and fields are fixed; unknown counter names/text are omitted.
    performance = data.get("measured_performance") or {}
    timing_names = (
        "camera_read",
        "camera_copy",
        "capture_interval",
        "detector",
        "adapters",
        "recognition",
        "emotion",
        "liveness",
        "liveness_trial",
        "render",
        "gui_tick",
        "gui_interval",
        "event_loop_delay",
        "preview_age",
        "preview_interval",
        "debug_json",
    )
    if isinstance(performance, dict):
        timings = performance.get("timings_ms", {})
        if not isinstance(timings, dict):
            timings = {}
        report["measured_performance"] = {
            "window_s": number(performance.get("window_s")),
            "capture_fps": number(performance.get("capture_fps")),
            "preview_fps": number(performance.get("preview_fps")),
            "timings_ms": {
                name: {key: number(value.get(key)) for key in ("mean", "max", "n")}
                for name in timing_names
                if isinstance(value := timings.get(name), dict)
            },
        }
    return report


def replace_text(text, value):
    """Preserve both scroll axes and an unchanged/uniquely relocated selection."""
    y, x = text.yview(), text.xview()
    ranges = text.tag_ranges("sel")
    selected = text.get(*ranges[:2]) if len(ranges) >= 2 else None
    indices = tuple(str(v) for v in ranges[:2])
    text.configure(state="normal")
    text.delete("1.0", "end")
    text.insert("1.0", value)
    if selected:
        if text.get(*indices) == selected:
            text.tag_add("sel", *indices)
        elif value.count(selected) == 1:
            start = value.index(selected)
            text.tag_add("sel", f"1.0+{start}c", f"1.0+{start + len(selected)}c")
    text.configure(state="disabled")
    if y:
        text.yview_moveto(y[0])
    if x:
        text.xview_moveto(x[0])
