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
    "spoof": "Spoof detected",
    "uncertain": "Uncertain liveness",
    "no_face": "Show one face to the camera",
    "multiple_faces": "Multiple faces detected",
    "capture_error": "Camera or capture unavailable",
    "inference_error": "Verification failed",
    "invalid_verification": "Invalid verification result",
    "stale": "Verification expired",
}


class AttendanceView:
    def __init__(self, dashboard, service):
        self.service = service
        self._count = 0
        self._recent_names = []
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
        self.identity = ttk.Label(section, style="Card.TLabel")
        self.identity.pack(anchor="w", fill="x", pady=(3, 0))
        self.feedback = ttk.Label(section, style="Card.TLabel")
        self.feedback.pack(anchor="w", fill="x", pady=(3, 0))
        self.history = ttk.Label(section, style="CardQuiet.TLabel")
        for label in (self.identity, self.feedback, self.history):
            dashboard.wrap_label(label)
        self.refresh()

    def refresh(self, result=None):
        state = self.service.state
        active = self.service.session_id is not None
        self.session.configure(text="End session" if active else "Start session",
                               command=self._end if active else self._start)
        self.session_status.configure(text="Session active" if active else "Session inactive")
        self.identity.configure(text=f"Verified identity · {state.verified_name or '—'}")
        self.confirm.state(("!disabled",) if state.can_record else ("disabled",))
        code = (result or state).code
        message = "Save failed. Verify again before retrying." if code == "save_failed" else MESSAGES.get(code, "")
        self.feedback.configure(text=message if code != "session_inactive" else "")
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
        self.refresh()

    def _record(self):
        result = self.service.record()
        if result.record is not None:
            self._count += 1
            self._recent_names.insert(0, result.record.person_name)
            del self._recent_names[3:]
        self.refresh(result)
