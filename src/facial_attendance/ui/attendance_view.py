"""Present manual attendance controls using only the attendance service state."""

from tkinter import ttk


MESSAGES = {
    "session_inactive": "Session inactive",
    "model_unavailable": "Model unavailable for attendance verification",
    "verification_pending": "Verification pending",
    "ready": "Ready to record",
    "recorded": "Attendance recorded",
    "already_recorded": "Already recorded in this session",
    "unknown": "Unknown identity",
    "spoof": "Spoof detected. Restart Camera to retry.",
    "uncertain": "Uncertain liveness",
    "no_face": "Show one face to the camera",
    "multiple_faces": "Multiple faces detected",
    "capture_error": "Camera or capture unavailable",
    "inference_error": "Verification failed",
    "invalid_verification": "Invalid verification result",
    "stale": "Verification expired",
}


class AttendanceView:
    def __init__(self, dashboard, service, *, retry_verification=None):
        self.service = service
        self._retry_verification = retry_verification
        self._preview_live = False
        self._count = 0
        self._recent_names = []
        self._last_receipt = None
        self._displayed = None
        section = ttk.Frame(dashboard.results, style="Card.TFrame")
        section.pack(fill="x", pady=(0, 12))
        ttk.Label(section, text="Attendance", style="Card.TLabel",
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(4, 6))
        self.session_status = ttk.Label(section, style="CardQuiet.TLabel")
        self.session_status.pack(anchor="w")
        self.session = ttk.Button(section)
        self.session.pack(anchor="w", pady=(4, 6))
        self.confirm = ttk.Button(section, text="Record attendance", command=self._record,
                                  style="Primary.TButton")
        self.confirm.pack(anchor="w", pady=(6, 0))
        self.retry_button = None
        if retry_verification is not None:
            self.retry_button = ttk.Button(section, text="Retry verification", command=self._retry)
        self.identity = ttk.Label(section, style="Card.TLabel")
        self.identity.pack(anchor="w", fill="x", pady=(3, 0))
        self.feedback = ttk.Label(section, style="Card.TLabel")
        self.feedback.pack(anchor="w", fill="x", pady=(3, 0))
        self.history = ttk.Label(section, style="CardQuiet.TLabel")
        for label in (self.identity, self.feedback, self.history):
            dashboard.wrap_label(label)
        self.refresh()

    def refresh(self, result=None, *, preview_live=None):
        if preview_live is not None:
            self._preview_live = preview_live
        state = self.service.state
        active = self.service.session_id is not None
        code = (result or state).code
        message = "Save failed. Verify again before retrying." if code == "save_failed" else MESSAGES.get(code, "")
        can_retry = self.retry_button is not None and active and state.code == "spoof"
        if can_retry:
            message = "Verification stopped after a spoof sample."
            if self._preview_live:
                message += " Live preview. You can retry verification."
        if result is not None and code in {"recorded", "already_recorded", "save_failed"}:
            self._last_receipt = message
        if active and code in {"verification_pending", "ready"} and self._last_receipt:
            message = f"{message}. {self._last_receipt}"
        if code == "session_inactive":
            message = ""
        display = (active, state.verified_name, state.can_record, can_retry, message,
                   self._count, tuple(self._recent_names))
        if display == self._displayed:
            return
        self._displayed = display
        self.session.configure(text="End session" if active else "Start session",
                               command=self._end if active else self._start)
        self.session_status.configure(text="Session active" if active else "Session inactive")
        self.identity.configure(text=f"Verified identity · {state.verified_name or '—'}")
        self.confirm.state(("!disabled",) if state.can_record else ("disabled",))
        if can_retry:
            self.retry_button.pack(after=self.confirm, anchor="w", pady=(6, 0))
        elif self.retry_button is not None:
            self.retry_button.pack_forget()
        self.feedback.configure(text=message)
        if self._recent_names:
            self.history.configure(text=f"Recorded this session ({self._count})\n" + "\n".join(self._recent_names))
            self.history.pack(anchor="w", fill="x", pady=(8, 0))
        else:
            self.history.pack_forget()

    def _start(self):
        self.service.start_session()
        self.refresh()

    def _end(self):
        self.service.end_session()
        self._count = 0
        self._recent_names.clear()
        self._last_receipt = None
        self.refresh()

    def _retry(self):
        if self.service.session_id is not None and self.service.state.code == "spoof":
            self._retry_verification()
            self._last_receipt = None
            self.refresh()

    def _record(self):
        result = self.service.record()
        if result.record is not None:
            self._count += 1
            self._recent_names.insert(0, result.record.person_name)
            del self._recent_names[3:]
        self.refresh(result)
