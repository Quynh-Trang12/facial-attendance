"""Present attendance controls and verified write results."""

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
    def __init__(self, dashboard, service, *, retry_verification=None, retrying_after_spoof=None,
                 automatic_record=False):
        self.service = service
        self._retry_verification = retry_verification
        self._retrying_after_spoof = retrying_after_spoof
        self._automatic_record = automatic_record
        self._preview_live = False
        self._count = 0
        self._recent_names = []
        self._last_receipt = None
        self._displayed = None
        self._dashboard = dashboard
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
        if not automatic_record:
            self.confirm.pack(anchor="w", pady=(6, 0))
        self.retry_button = None
        if retry_verification is not None:
            self.retry_button = ttk.Button(section, text="Retry verification", command=self._retry)
        self.identity = ttk.Label(section, style="Card.TLabel")
        if not automatic_record:
            self.identity.pack(anchor="w", fill="x", pady=(3, 0))
        self._feedback_slot, self.feedback = dashboard.message_slot(section, style="Card.TLabel")
        self._feedback_slot.pack(anchor="w", fill="x", pady=(3, 0))
        self.history = ttk.Label(section, style="CardQuiet.TLabel")
        for label in (self.identity, self.history):
            dashboard.wrap_label(label)
        self.refresh()

    def refresh(self, result=None, *, preview_live=None):
        if preview_live is not None:
            self._preview_live = preview_live
        state = self.service.state
        active = self.service.session_id is not None
        code = (result or state).code
        message = "Save failed. Verify again before retrying." if code == "save_failed" else MESSAGES.get(code, "")
        if code == "save_failed" and self._automatic_record:
            message = "Save failed. Leave the frame, then try again."
        automatic_retry = self._retrying_after_spoof is not None and self._retrying_after_spoof(state.code)
        can_retry = self.retry_button is not None and active and state.code == "spoof" and not automatic_retry
        if automatic_retry:
            message = "Spoof detected. Retrying once with fresh samples."
        elif can_retry:
            message = "Verification stopped after a spoof sample."
            if self._preview_live:
                message += " Live preview. You can retry verification."
        if result is not None and code in {"recorded", "already_recorded", "save_failed"}:
            self._last_receipt = message
        if active and code in {"verification_pending", "ready"}:
            if self._automatic_record and not automatic_retry:
                message = self._last_receipt or "Waiting for a stable live match"
            elif self._last_receipt:
                message = f"{message}. {self._last_receipt}"
        if code == "session_inactive":
            message = ""
        name = None if self._automatic_record else state.verified_name
        can_confirm = state.can_record and not self._automatic_record
        display = (active, name, can_confirm, can_retry, message,
                   self._count, tuple(self._recent_names))
        if display == self._displayed:
            return
        was_active = self._displayed is not None and self._displayed[0]
        self._displayed = display
        self.session.configure(text="End session" if active else "Start session",
                               command=self._end if active else self._start)
        self.session_status.configure(text=("Session active · Auto record" if self._automatic_record else "Session active")
                                      if active else "Session inactive")
        self.identity.configure(text=f"Verified identity · {name or '—'}")
        self.confirm.state(("!disabled",) if can_confirm else ("disabled",))
        if active != was_active:
            self._dashboard.reserve_message_lines(self._feedback_slot, 3 if active else 0)
        if self.retry_button is not None and can_retry != bool(self.retry_button.winfo_manager()):
            if can_retry:
                after = self.session if self._automatic_record else self.confirm
                self.retry_button.pack(after=after, anchor="w", pady=(6, 0))
            else:
                self.retry_button.pack_forget()
        self.feedback.configure(text=message)
        if self._recent_names:
            self.history.configure(text=f"Recorded this session ({self._count})\n" + "\n".join(self._recent_names))
            if not self.history.winfo_manager():
                self.history.pack(anchor="w", fill="x", pady=(8, 0))
        elif self.history.winfo_manager():
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
        automatic_retry = self._retrying_after_spoof is not None and self._retrying_after_spoof("spoof")
        if self.service.session_id is not None and self.service.state.code == "spoof" and not automatic_retry:
            self._retry_verification()
            self._last_receipt = None
            self.refresh()

    def _record(self):
        if not self._automatic_record:
            self.show_record_result(self.service.record())

    def show_record_result(self, result):
        if result.record is not None:
            self._count += 1
            self._recent_names.insert(0, result.record.person_name)
            del self._recent_names[3:]
        self.refresh(result)
