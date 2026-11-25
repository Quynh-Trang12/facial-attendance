"""Presentation for the local webcam workspace. No inference or persistence."""

from dataclasses import dataclass

BG = "#191B1E"
CARD = "#232528"
SURFACE = "#2C2F33"
BORDER = "#3B3E44"
ACCENT = "#A1B9D4"
TEXT = "#F2F3F4"
QUIET = "#B3B8C0"
VIEWFINDER = "#101113"
INTERACTIVE = "#383C42"
TONES = {
    "neutral": QUIET,
    "info": TEXT,
    "success": "#80D5AE",
    "warning": "#F0C17D",
    "error": "#FFA3A6",
}
EMOTION_ICONS = {"Happy": "☺", "Positive": "☺", "Sad": "☹", "Negative": "☹", "Neutral": "–"}


@dataclass(frozen=True)
class Result:
    value: str
    detail: str
    tone: str = "neutral"


def camera_status(usable, failed):
    if usable:
        return Result("Camera active", "", "success")
    if failed:
        return Result("Camera unavailable", "", "error")
    return Result("Connecting camera", "", "warning")


def preview_phase(state, detecting, previous):
    if state == "idle" and detecting:
        return previous if previous in {"no_face", "multiple_faces"} else "checking"
    return state


def preview_results(state, view, emotion, *, scan, trained_liveness, trained_emotion, score_error=False,
                    phase=None):
    """Map display-safe results. A scan result alone carries no identity authority."""
    phase = phase or state.ui_state
    blocked = phase in {"idle", "no_face", "multiple_faces"}
    if phase == "error":
        return (Result("Unavailable", "Check the camera or model.", "error"),
                Result("Unavailable", "Waiting for a usable camera frame."),
                Result("Unavailable", "Waiting for a usable camera frame."))
    if blocked:
        identity = Result({
            "idle": "Waiting for camera", "no_face": "No face detected", "multiple_faces": "Multiple faces",
        }[phase], "Show one face to the camera.")
    elif view is not None and view.identity_name:
        identity = Result(view.identity_name, "Preview match", "info")
    elif view is not None and view.identity == "Unknown":
        identity = Result("Unknown", "Open Developer Mode to register.", "warning")
    elif any(error.module == "recognition" for error in state.errors):
        identity = Result("Recognition error", "Wait for a new sample.", "error")
    else:
        identity = Result("Checking match", "Verification pending", "warning")

    if not trained_liveness:
        simulated = state.liveness is not None and state.liveness.source == "stub"
        liveness = (Result("Simulated", "Simulated output — not verified", "warning") if simulated else
                    Result("Model unavailable", "Liveness unverified", "warning"))
    elif blocked:
        liveness = Result("uncertain" if phase == "no_face" else "Waiting for a face",
                          "No current liveness result")
    elif score_error and scan is None:
        liveness = Result("Scoring error", "Waiting for another sample", "error")
    elif scan is None:
        liveness = Result("Collecting frames", "Waiting for the first scan", "warning")
    else:
        tone = {"live": "success", "spoof": "error", "uncertain": "warning"}[scan.decision]
        liveness = Result(scan.decision, f"Last scan at {scan.local_time}", tone)

    if blocked:
        expression = Result("Waiting for a face", "Facial expression preview")
    elif any(error.module == "emotion" for error in state.errors):
        expression = Result("Emotion unavailable", "Waiting for another sample", "warning")
    elif emotion is not None and emotion.label != "Unavailable":
        icon = EMOTION_ICONS.get(emotion.label)
        expression = Result(f"{icon}  {emotion.label}" if icon else emotion.label,
                            "Facial expression preview", "info")
    else:
        expression = Result("Waiting for inference" if trained_emotion else "Model unavailable",
                            "Facial expression preview")
    return identity, liveness, expression


def corner_segments(box):
    """Return four display-space corners for an already transformed face box."""
    x1, y1, x2, y2 = box
    length = min(24, (x2 - x1) / 3, (y2 - y1) / 3)
    return (
        (x1, y1 + length, x1, y1, x1 + length, y1),
        (x2 - length, y1, x2, y1, x2, y1 + length),
        (x1, y2 - length, x1, y2, x1 + length, y2),
        (x2 - length, y2, x2, y2, x2, y2 - length),
    )


def apply_theme(root):
    from tkinter import ttk

    style = ttk.Style(root)
    name = "facial-attendance"
    if name not in style.theme_names():
        button = {"background": SURFACE, "foreground": TEXT, "bordercolor": BORDER,
                  "lightcolor": SURFACE, "darkcolor": SURFACE, "padding": (12, 7)}
        style.theme_create(name, parent="clam", settings={
            ".": {"configure": {"background": BG, "foreground": TEXT, "bordercolor": BORDER}},
            "TFrame": {"configure": {"background": BG}},
            "Card.TFrame": {"configure": {"background": CARD}},
            "TLabel": {"configure": {"background": BG, "foreground": TEXT}},
            "Card.TLabel": {"configure": {"background": CARD, "foreground": TEXT}},
            "Quiet.TLabel": {"configure": {"background": BG, "foreground": QUIET}},
            "CardQuiet.TLabel": {"configure": {"background": CARD, "foreground": QUIET}},
            "TSeparator": {"configure": {"background": BORDER}},
            "TButton": {"configure": button, "map": {
                "background": [("pressed", BORDER), ("active", INTERACTIVE)],
                "foreground": [("disabled", QUIET)], "bordercolor": [("focus", ACCENT)],
            }},
            "Primary.TButton": {"configure": {**button, "background": ACCENT, "foreground": BG, "bordercolor": ACCENT},
                                "map": {"foreground": [("disabled", QUIET)],
                                        "background": [("disabled", SURFACE), ("pressed", "#8CA8C6"),
                                                       ("active", "#B4C8E0")]}},
            "TEntry": {"configure": {"fieldbackground": SURFACE, "foreground": TEXT, "insertcolor": TEXT},
                       "map": {"bordercolor": [("focus", ACCENT)]}},
            "TCheckbutton": {"configure": {"background": BG, "foreground": TEXT, "indicatorcolor": SURFACE},
                             "map": {"indicatorcolor": [("selected", ACCENT)],
                                     "background": [("active", BG)]}},
            "TLabelframe": {"configure": {"background": BG, "bordercolor": BORDER}},
            "TLabelframe.Label": {"configure": {"background": BG, "foreground": TEXT}},
            "TNotebook": {"configure": {"background": BG, "bordercolor": BORDER}},
            "TNotebook.Tab": {"configure": {"background": SURFACE, "foreground": QUIET, "padding": (10, 6)},
                              "map": {"background": [("selected", CARD)], "foreground": [("selected", TEXT)]}},
        })
    style.theme_use(name)
    root.configure(background=BG)


class Dashboard:
    def __init__(self, parent, *, restart_camera, switch_mode, trained_liveness, attendance=None,
                 retry_attendance=None, retrying_after_spoof=None, automatic_record=False):
        import tkinter as tk
        from tkinter import font, ttk

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Facial Attendance", font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, sticky="w")
        self.camera = ttk.Label(header, width=23, anchor="center", font=("TkDefaultFont", 11))
        self.camera.grid(row=0, column=1, padx=(8, 12))
        self.restart_button = ttk.Button(header, text="Restart Camera", command=restart_camera)
        self.restart_button.grid(row=0, column=2, padx=(0, 8))
        self.mode_button = ttk.Button(header, text="Developer Mode", command=switch_mode)
        self.mode_button.grid(row=0, column=3)
        ttk.Separator(parent).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        self.camera_card = ttk.Frame(parent)
        self.camera_card.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        self.camera_card.columnconfigure(0, weight=1)
        self.camera_card.rowconfigure(1, weight=1)

        self.inspector = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12, 8, 12))
        self.inspector.grid(row=2, column=1, rowspan=3, sticky="nsew", padx=(10, 0))
        self.inspector.columnconfigure(0, weight=1)
        self.inspector.rowconfigure(1, weight=1)
        ttk.Label(self.inspector, text="Analysis results", style="Card.TLabel",
                  font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 9))
        self._scroll_area = tk.Canvas(self.inspector, width=1, height=1, background=CARD,
                                      highlightthickness=0, yscrollincrement=1)
        self._scroll_area.grid(row=1, column=0, sticky="nsew")
        self._scrollbar = ttk.Scrollbar(self.inspector, orient="vertical", command=self._scroll_area.yview)
        self.inspector.columnconfigure(1, minsize=self._scrollbar.winfo_reqwidth() + 3)
        self._scrollbar.grid(row=1, column=1, sticky="ns", padx=(3, 0))
        self._scrollbar.grid_remove()
        self._scroll_area.configure(yscrollcommand=self._scrollbar.set)
        self.results = ttk.Frame(self._scroll_area, style="Card.TFrame")
        self._inspector_window = self._scroll_area.create_window(0, 0, window=self.results, anchor="nw")
        self._wrapped_labels = []
        self._inspector_width = None
        self._scroll_bounds = None
        self._scrollbar_visible = False
        self._scroll_update_pending = False
        self._aqua = parent.tk.call("tk", "windowingsystem") == "aqua"
        self._message_line_height = font.nametofont("TkDefaultFont", root=parent).metrics("linespace")
        self.results.bind("<Configure>", self._update_scroll_region)
        self._scroll_area.bind("<Configure>", self._fit_inspector)
        parent.winfo_toplevel().bind("<MouseWheel>", self._scroll_inspector, add="+")
        caption_font = font.Font(root=parent, font=("TkDefaultFont", 11, "bold"))
        value_font = font.Font(root=parent, font=("TkDefaultFont", 15, "bold"))
        detail_font = font.Font(root=parent, font=("TkDefaultFont", 11))
        caption_height = caption_font.metrics("linespace")
        value_height = value_font.metrics("linespace") * 2
        detail_height = detail_font.metrics("linespace") * 2
        self.cards = []
        for title in ("Face Recognition", "Anti-Spoofing", "Emotion Detection"):
            section = ttk.Frame(self.results, style="Card.TFrame",
                                height=8 + caption_height + value_height + detail_height)
            section.pack(fill="x", pady=(7, 3))
            section.grid_propagate(False)
            section.columnconfigure(0, weight=1)
            section.rowconfigure(0, minsize=caption_height + 3)
            section.rowconfigure(1, minsize=value_height + 3)
            section.rowconfigure(2, minsize=detail_height)
            ttk.Label(section, text=title, style="CardQuiet.TLabel", font=caption_font).grid(row=0, column=0, sticky="nw")
            value = ttk.Label(section, style="Card.TLabel", font=value_font)
            value.grid(row=1, column=0, sticky="new")
            detail = ttk.Label(section, style="CardQuiet.TLabel", font=detail_font)
            detail.grid(row=2, column=0, sticky="new")
            self.wrap_label(value)
            self.wrap_label(detail)
            self.cards.append((value, detail))
            ttk.Separator(self.results).pack(fill="x", pady=(0, 3))
        self._card_fonts = (caption_font, value_font, detail_font)

        if attendance is None:
            ttk.Label(self.results, text="Attendance saving disabled", style="Card.TLabel",
                      font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(9, 2))
        ttk.Label(self.results, text="Trained liveness enabled" if trained_liveness else "Liveness model unavailable",
                  style="CardQuiet.TLabel").pack(anchor="w", pady=(0, 8))
        self.attendance_view = None
        if attendance is not None:
            from .attendance_view import AttendanceView
            self.attendance_view = AttendanceView(self, attendance, retry_verification=retry_attendance,
                                                  retrying_after_spoof=retrying_after_spoof,
                                                  automatic_record=automatic_record)
        self._camera_state = None
        self._results = None
        self._preview_phase = "idle"
        self.set_camera(camera_status(False, False))

    def wrap_label(self, label):
        self._wrapped_labels.append(label)
        label.configure(wraplength=max(160, self._scroll_area.winfo_width() - 4))

    def message_slot(self, parent, **label_options):
        from tkinter import ttk

        slot = ttk.Frame(parent, style="Card.TFrame")
        slot.columnconfigure(0, weight=1)
        label = ttk.Label(slot, justify="left", **label_options)
        label.grid(row=0, column=0, sticky="new")
        self.wrap_label(label)
        return slot, label

    def reserve_message_lines(self, slot, lines):
        slot.rowconfigure(0, minsize=self._message_line_height * lines + 4 if lines else 0)

    def _fit_inspector(self, event):
        if event.width != self._inspector_width:
            self._inspector_width = event.width
            self._scroll_area.itemconfigure(self._inspector_window, width=event.width)
            for label in self._wrapped_labels:
                label.configure(wraplength=max(1, event.width - 4))
        self._update_scroll_region()

    def _update_scroll_region(self, _event=None):
        if not self._scroll_update_pending:
            self._scroll_update_pending = True
            self._scroll_area.after_idle(self._sync_scroll_region)

    def _sync_scroll_region(self):
        self._scroll_update_pending = False
        bounds = self._scroll_area.bbox(self._inspector_window)
        if bounds is None:
            return
        _, top, _, bottom = bounds
        viewport_height = self._scroll_area.winfo_height()
        if bounds != self._scroll_bounds:
            # Keep the same pixel at the top when content below it changes size.
            scroll_top = self._scroll_area.canvasy(0)
            self._scroll_area.configure(scrollregion=bounds)
            self._scroll_bounds = bounds
            target = min(max(top, scroll_top), max(top, bottom - viewport_height))
            if bottom > top and self._scroll_area.canvasy(0) != target:
                self._scroll_area.yview_moveto((target - top) / (bottom - top))
        visible = bottom - top > viewport_height
        if visible != self._scrollbar_visible:
            self._scrollbar_visible = visible
            if visible:
                self._scrollbar.grid()
            else:
                self._scrollbar.grid_remove()

    def _scroll_inspector(self, event):
        widget = event.widget
        while widget is not None and widget is not self.inspector:
            widget = widget.master
        if widget is self.inspector and event.delta and self._scrollbar.winfo_ismapped():
            pixels = abs(event.delta) * (8 if self._aqua else 36 / 120)
            step = max(1, min(64, round(pixels)))
            self._scroll_area.yview_scroll(-step if event.delta > 0 else step, "units")
            return "break"

    def set_camera(self, status):
        if status != self._camera_state:
            self.camera.configure(text=f"●  {status.value}", foreground=TONES[status.tone])
            self._camera_state = status

    def set_results(self, results):
        previous = self._results or (None, None, None)
        for (value, detail), result, old in zip(self.cards, results, previous):
            if old is None or (result.value, result.tone) != (old.value, old.tone):
                value.configure(text=result.value, foreground=TONES[result.tone] if result.tone != "neutral" else TEXT)
            if old is None or result.detail != old.detail:
                detail.configure(text=result.detail)
        self._results = results

    def update_preview(self, state, view, emotion, *, detecting, scan, trained_liveness, trained_emotion, score_error):
        self._preview_phase = preview_phase(state.ui_state, detecting, self._preview_phase)
        results = preview_results(
            state, view, emotion, scan=scan, trained_liveness=trained_liveness, trained_emotion=trained_emotion,
            score_error=score_error, phase=self._preview_phase,
        )
        self.set_results(results)
        if self.attendance_view is not None:
            self.attendance_view.refresh(preview_live=results[1].value == "live")
