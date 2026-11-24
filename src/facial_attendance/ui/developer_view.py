"""Read-only presentation helpers and a single non-modal diagnostics window.

Neither mode owns inference, gallery state or authorization. Export uses a
separate scalar allowlist; it never dumps the raw on-screen identity data.
"""
from dataclasses import asdict
import json
from pathlib import Path

from ..emotion.model_contracts import LEGACY_LABELS
from .diagnostic_snapshot import privacy_safe_snapshot, replace_text, utc_timestamp


def standard_fields(state, view, emotion_view):
    """Concise preview text; no confidence, timing or internal identifiers."""
    identity = view.identity if view is not None else "Unavailable"
    emotion = emotion_view.label if emotion_view is not None else (
        view.emotion if view is not None else "Unavailable")
    liveness = view.liveness if view is not None else "Unavailable"
    if identity == "Unknown":
        identity = "Unknown person"
    status = {
        "no_face": "No face detected",
        "multiple_faces": "Multiple faces detected",
        "spoof": "Simulated spoof — blocked",
        "uncertain": "Uncertain liveness — retry",
    }.get(state.ui_state)
    if state.ui_state == "error":
        status = "Camera unavailable" if any(e.module == "capture" for e in state.errors) else "Model unavailable"
    if status is None:
        status = "Unverified preview" if identity != "Unavailable" or emotion != "Unavailable" else "Checking..."
    if emotion_view is not None and emotion_view.sampled and emotion != "Unavailable":
        emotion = f"{emotion} (latest sample)"
    return {"status": status, "identity": f"Identity: {identity}",
            "emotion": f"Emotion: {emotion}", "liveness": f"Liveness (SIMULATED): {liveness}"}


def runtime_report(app, now):
    """Inspect already-loaded state; never call a model, matcher or controller."""
    state = app.controller.state
    face = getattr(app.adapters, "face_inference", None)
    emotion_adapter = getattr(app.adapters, "emotion_adapter", None)
    emotion_model = getattr(emotion_adapter, "inference", None)
    matcher = getattr(app.adapters, "matcher", None)
    gallery = getattr(matcher, "_gallery", None)
    display = getattr(app, "_last_display_view", None)
    emotion_display = getattr(app, "emotion_presentation", None)
    audit = emotion_display.audit() if emotion_display is not None else None
    current_emotion = getattr(app, "_last_emotion_view", None)
    strict_sample = getattr(getattr(app, "presentation", None), "_sample", None)
    captured_at = audit["captured_at"] if audit else None
    raw = asdict(state)
    controlled = getattr(app, "_controlled_registration", False)
    workspace = getattr(app, "_local_workspace", False)
    recognition = state.recognition
    identity_display = asdict(display) if display else None
    if identity_display is not None:
        identity_display.pop("emotion", None)  # Emotion has its own independent view below.
    liveness = {"active_source": "verified model unavailable" if workspace else "stub / SIMULATED",
                "raw": raw["liveness"],
                "genuine_liveness_verified": False, "trained_threshold": None, "trained_model_metadata": None}
    released = getattr(app, "released_liveness", None)
    if released is not None:
        sample = app.liveness_preview.current(app.live, now, app.MAX_AGE)
        scan = app.liveness_preview.last_check
        liveness.update(active_source="AENet-Full preview", trained_threshold=released.provenance["threshold"],
                        trained_model_metadata=released.provenance,
                        released_preview=asdict(sample) if sample is not None else None,
                        last_completed_scan=asdict(scan) if scan is not None else None)
    if getattr(app, "_liveness_trial_worker", None) is not None:
        from ..liveness.checkpoint_trial import SCORING_METHOD
        trial = app.current_liveness_trial()
        scores = [] if trial is None or trial.error is not None else [
            {**asdict(sample), "threshold_met": sample.threshold_met} for sample in trial.value
        ]
        liveness["experimental_comparison"] = {
            "method": SCORING_METHOD, "can_authorize": False,
            "status": "error" if trial is not None and trial.error is not None else "ready" if scores else "unavailable",
            "scores": scores,
        }
    return {
        "runtime": {**getattr(app, "_runtime_info", {}),
                    "camera_status": "unavailable" if any(e.module == "capture" for e in state.errors) else (
                        "delivering frames" if app._drawn_sequence is not None else "opening / awaiting frame"),
                    "face_count": len(app.live.boxes), "detection_status": app.live.status,
                    "detector_busy": app.detector.busy, "inference_busy": app.inference.busy},
        "recognition": {"checkpoint": getattr(face, "provenance", None),
                        "embedding_dimensions": getattr(getattr(face, "contract", None), "embedding_dim", None),
                        "configured_threshold": getattr(getattr(face, "contract", None), "cosine_threshold", None),
                        "raw": raw["recognition"],
                        "gallery_people_count": len(gallery.people) if gallery is not None else None,
                        "gallery_status": "recognition error" if any(e.module == "recognition" for e in state.errors) else (
                            "loaded compatible gallery" if gallery is not None else "not active"),
                        "decision": None if recognition is None else "known" if recognition.accepted else "unknown",
                        "identity_verified": False},
        "emotion": {"checkpoint": getattr(emotion_model, "provenance", None),
                    "label_mapping": dict(enumerate(getattr(emotion_model, "labels", LEGACY_LABELS))),
                    "prediction_interval_s": emotion_display.prediction_interval if emotion_display is not None else None,
                    "raw_current_prediction": raw["emotion"],
                    "displayed_label": current_emotion.label if current_emotion else (display.emotion if display else None),
                    "display_age_ms": round(max(0, now-captured_at)*1000) if captured_at is not None else None,
                    "display_limit_ms": audit["freshness_limit_ms"] if audit else 800,
                    "display_smoothing": "latest valid label; confidence omitted" if emotion_display else "bounded legacy majority",
                    "display_reason": current_emotion.reason if current_emotion else None,
                    "historical_display_audit": audit},
        "liveness": liveness,
        "presentation": {"authoritative_state": state.ui_state,
                         "identity_liveness_display": identity_display,
                         "identity_liveness_age_ms": round(max(0, now-strict_sample.captured_at)*1000) if strict_sample else None,
                         "display_expiry_ms": 800, "previous_box_retention_ms": round(app.live.stability.hold_seconds*1000),
                         "current_box_is_historical": app.live.visual_regions(now)[2],
                         "note": "Historical/display data never authorizes saving; continuity is not identity tracking."},
        "authorization": {"can_save": state.can_save, "identity_verified": False,
                          "genuine_registration_available": False, "attendance_available": False,
                          "controlled_generated_registration": controlled,
                          "liveness_is_simulated": not workspace,
                          "registration_action": app.controller.action,
                          "reason_blocked": ("AENet-Full is used in preview. Attendance saving is disabled." if released else
                                             "Verified liveness is unavailable. Local profiles cannot authorize attendance."
                                             if workspace else
                                             "Liveness is simulated. Genuine enrollment and attendance saving are disabled.")},
        "measured_performance": app.metrics.peek() if app.metrics.enabled else None,
        "raw_current_sys02": raw,
    }


class DeveloperDiagnostics:
    """One reusable, screen-bounded Toplevel with scrollable read-only tabs."""
    INTERVAL = .25

    def __init__(self, root, on_close, latest_completed=None, *, unverified_workspace=False):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title("Developer diagnostics | Facial Attendance")
        self.window.protocol("WM_DELETE_WINDOW", on_close)
        width = min(1000, max(320, root.winfo_screenwidth()-80))
        height = min(680, max(240, root.winfo_screenheight()-160))
        self.window.geometry(f"{width}x{height}+40+60")
        self.window.minsize(min(620, width), min(360, height))
        host = ttk.Frame(self.window, padding=12)
        host.pack(fill="both", expand=True)
        warning = ("Local verification preview • Attendance saving disabled"
                   if unverified_workspace else "Liveness SIMULATED • Identity UNVERIFIED • Genuine saving disabled")
        warning_color = "#a04400"
        if unverified_workspace:
            from .dashboard import ACCENT, BG, CARD, TEXT, TONES
            self.window.configure(background=BG)
            warning_color = TONES["warning"]
        ttk.Label(host, text=warning,
                  foreground=warning_color, wraplength=600).pack(anchor="w")
        ttk.Label(host, text="Raw current inference and display history are separate. No images or embeddings appear here.",
                  wraplength=600).pack(anchor="w", pady=(4, 8))
        controls = ttk.Frame(host)
        controls.pack(fill="x", pady=(0, 4))
        self.pause_button = ttk.Button(controls, text="Pause refresh", command=self.toggle_pause)
        self.pause_button.pack(side="left")
        self.copy_button = ttk.Button(controls, text="Copy displayed JSON", command=self.copy_json)
        self.copy_button.pack(side="left", padx=6)
        self.export_button = ttk.Button(controls, text="Export privacy-safe snapshot", command=self.export_json)
        self.export_button.pack(side="left")
        self.completed_button = ttk.Button(host, text="Capture latest completed inference (historical)",
                                           command=self.capture_completed)
        self.completed_button.pack(anchor="w", pady=4)
        self.status = tk.StringVar()
        ttk.Label(host, textvariable=self.status, wraplength=560).pack(anchor="w")
        self.notice = tk.StringVar()
        ttk.Label(host, textvariable=self.notice, wraplength=560).pack(anchor="w", pady=(0, 4))
        self.notebook = ttk.Notebook(host)
        self.notebook.pack(fill="both", expand=True)
        self.texts = []
        for title in ("Runtime and presentation", "Raw current SYS-02", "Completed inference (historical)"):
            page = ttk.Frame(self.notebook)
            self.notebook.add(page, text=title)
            text = ScrolledText(page, width=60, height=15, wrap="none", state="disabled", exportselection=False)
            if unverified_workspace:
                text.configure(background=CARD, foreground=TEXT, selectbackground=ACCENT, selectforeground=BG,
                               relief="flat", highlightthickness=0, padx=8, pady=8)
            horizontal = ttk.Scrollbar(page, orient="horizontal", command=text.xview)
            text.configure(xscrollcommand=horizontal.set)
            horizontal.pack(side="bottom", fill="x")
            text.pack(fill="both", expand=True)
            self.texts.append(text)
        self.runtime, self.payload, self.completed = self.texts
        self.visible = False
        self.paused = False
        self.raw_refresh_requested = False
        self._unverified_workspace = unverified_workspace
        self._latest_completed = latest_completed or (lambda: None)
        self._snapshot_meta = {}
        self._next_refresh = 0.
        self._last_runtime = None
        self.refresh_count = 0
        self.notebook.bind("<<NotebookTabChanged>>", lambda _: self._update_status())
        self._update_status()

    def show(self):
        self.visible = True
        self._next_refresh = 0.
        self.raw_refresh_requested = not self.paused
        self.window.deiconify()
        self.window.lift()
        self.window.focus_set()

    def hide(self):
        self.visible = False
        self.window.withdraw()

    def refresh(self, app, now):
        if not self.visible or self.paused or now < self._next_refresh:
            return
        self._next_refresh = now + self.INTERVAL
        report = runtime_report(app, now)
        # The separately refreshed raw tab stays accurate; avoid duplicating it.
        report.pop("raw_current_sys02")
        text = json.dumps(report, indent=2, allow_nan=False)
        if text == self._last_runtime:
            return
        replace_text(self.runtime, text)
        self._stamp(0, "runtime")
        self._last_runtime = text
        self.refresh_count += 1

    def update_raw(self, state):
        if not self.visible or self.paused:
            return False
        replace_text(self.payload, json.dumps(asdict(state), indent=2, allow_nan=False))
        self._stamp(1, "current_sys02")
        self.raw_refresh_requested = False
        return True

    def _stamp(self, index, source, timestamp=None):
        self._snapshot_meta[index] = {"source": source, "timestamp": timestamp or utc_timestamp()}
        if getattr(self, "_unverified_workspace", False):
            self._snapshot_meta[index]["liveness_is_simulated"] = False
        self._update_status()

    def _update_status(self):
        index = self.notebook.index(self.notebook.select())
        meta = self._snapshot_meta.get(index)
        source = ("HISTORICAL completed inference — diagnostic only" if index == 2 else
                  "FROZEN diagnostic snapshot — not current authorization" if self.paused else "current diagnostic snapshot")
        self.status.set(f"{'PAUSED' if self.paused else 'LIVE'} • {source} • Snapshot: "
                        + (meta['timestamp'] if meta else "not captured yet"))

    def toggle_pause(self):
        if not self.visible:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume refresh" if self.paused else "Pause refresh")
        if not self.paused:
            self._next_refresh = 0.
            self.raw_refresh_requested = True
            if self.notebook.index(self.notebook.select()) == 2:
                self.notebook.select(1)
        self._update_status()

    def capture_completed(self):
        if not self.visible:
            return False
        record = self._latest_completed()
        if record is None:
            self.notice.set("No completed inference is available yet.")
            return False
        replace_text(self.completed, json.dumps(asdict(record.state), indent=2, allow_nan=False))
        self.paused = True
        self.pause_button.configure(text="Resume refresh")
        self._stamp(2, "completed_inference", record.timestamp)
        self.notebook.select(2)
        self._update_status()
        self.notice.set("Historical diagnostic data only. Camera and inference continue; authorization is unchanged.")
        return True

    def copy_json(self):
        if not self.visible:
            return False
        index = self.notebook.index(self.notebook.select())
        text = self.texts[index].get("1.0", "end-1c")
        if not text:
            self.notice.set("No displayed JSON to copy.")
            return False
        import tkinter as tk
        try:
            self.window.clipboard_clear()
            self.window.clipboard_append(text)
        except tk.TclError:
            self.notice.set("Clipboard is unavailable.")
            return False
        self.notice.set("Displayed JSON copied.")
        return True

    def export_json(self):
        if not self.visible:
            return False
        index = self.notebook.index(self.notebook.select())
        try:
            data = json.loads(self.texts[index].get("1.0", "end-1c"))
            snapshot = privacy_safe_snapshot(data, self._snapshot_meta.get(index, {}), paused=self.paused)
        except (ValueError, TypeError):
            self.notice.set("No valid displayed snapshot to export.")
            return False
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(parent=self.window, title="Export privacy-safe diagnostic snapshot",
                    defaultextension=".json", initialfile="diagnostic-snapshot.json", filetypes=[("JSON", "*.json")])
        if not filename:
            return False
        try:
            Path(filename).write_text(json.dumps(snapshot, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        except OSError:
            self.notice.set("Export failed. Check the destination and retry.")
            return False
        self.notice.set("Privacy-safe JSON snapshot exported. Personal identity fields and local paths are omitted.")
        return True
