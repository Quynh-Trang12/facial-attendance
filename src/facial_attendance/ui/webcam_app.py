"""Automatic Tk webcam demo plus an explicit developer selection harness."""

import argparse
import json
from contextlib import suppress
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import monotonic

from .diagnostics import Diagnostics
from .face_detection import (
    YUNET_MODEL,
    FaceDetectorUnavailable,
    HaarFaceDetector,
    ScaledFaceDetector,
    YuNetFaceDetector,
)
from .live_preview import (
    CameraFeed,
    LiveController,
    SingleWorker,
    evaluate_adapters,
    prepare_preview_image,
)
from .webcam_prototype import (
    SCENARIOS,
    PrototypeController,
    StubAdapters,
    frame_size,
    run_stubs,
)

AUTOMATIC = "Automatic detection"
MANUAL = "Manual selection"
LOCAL_ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"
LOCAL_LIVENESS_CHECKPOINT = LOCAL_ARTIFACTS / "models/liveness/liveness-inference.pt"


def fixed_result_label(parent, height, **options):
    """Reserve pixel geometry independently of a ttk label's changing text."""
    from tkinter import ttk

    area = ttk.Frame(parent, width=420, height=height)
    area.pack(anchor="w", fill="x")
    area.pack_propagate(False)
    label = ttk.Label(area, wraplength=410, anchor="nw", justify="left", **options)
    label.pack(fill="both", expand=True)
    return label


class WebcamApp:
    """Explicit developer freeze/selection harness; retains the verified legacy API."""

    def __init__(
        self,
        root,
        open_camera,
        *,
        source_name: str,
        scenario=SCENARIOS[0],
        action="attendance",
    ):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.open_camera = open_camera
        self.controller = PrototypeController()
        self.detect_faces = HaarFaceDetector()
        self.camera = None
        self.latest = self.frozen = None
        self.photo = None
        self.boxes = []
        self.drag_start = None
        self.drag_outline = None
        self.preview_size = (0, 0)
        self.evaluated = False

        root.title("Facial Attendance | SYS-05 demo")
        root.protocol("WM_DELETE_WINDOW", self.close)
        panel = ttk.Frame(root, padding=14)
        panel.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        ttk.Label(
            panel, text="Facial Attendance", font=("TkDefaultFont", 18, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            panel,
            text=f"DEMO / SIMULATED OUTPUT • Developer manual harness • {source_name}",
        ).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.canvas = tk.Canvas(
            panel, width=640, height=360, background="#162332", highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, sticky="nw", pady=(10, 6))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._draw_drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)

        controls = ttk.Frame(panel)
        controls.grid(row=2, column=0, sticky="w")
        self.freeze_button = ttk.Button(
            controls, text="Freeze frame", command=self.freeze
        )
        self.freeze_button.pack(side="left")
        self.run_button = ttk.Button(controls, text="Run stubs", command=self.evaluate)
        self.run_button.pack(side="left", padx=6)
        self.clear_button = ttk.Button(
            controls, text="Clear regions", command=self._clear_regions
        )
        self.clear_button.pack(side="left")
        ttk.Button(controls, text="Retry / resume preview", command=self.retry).pack(
            side="left", padx=6
        )
        self.hint = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.hint, wraplength=630).grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )

        results = ttk.Frame(panel)
        results.grid(
            row=1, column=1, rowspan=3, sticky="nsew", padx=(16, 0), pady=(10, 0)
        )
        self.action = tk.StringVar(value=action)
        self.scenario = tk.StringVar(value=scenario)
        self.selection_mode = tk.StringVar(value=MANUAL)
        ttk.Label(
            results,
            text=f"Developer scenario: {scenario}\nOperation: {action} (non-persistent)",
            wraplength=340,
        ).pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(
            results, textvariable=self.summary, wraplength=340, justify="left"
        ).pack(anchor="w", pady=8)
        ttk.Label(results, text="SYS-02 result payload").pack(anchor="w")
        self.payload = ScrolledText(
            results, width=45, height=14, wrap="word", state="disabled"
        )
        self.payload.pack(fill="both", expand=True)
        self._render()
        self._timer = root.after(0, self._tick)

    def _change_run_settings(self, _event):
        if self.camera is not None:
            self._resume_preview()

    def _change_selection_mode(self, _event):
        if self.frozen is not None:
            if self.controller.state.ui_state in {"spoof", "uncertain"}:
                self._resume_preview()
            else:
                self._prepare_selection()

    def retry(self):
        self._resume_preview()
        if self.camera is None:
            self.start()

    def _resume_preview(self):
        self.controller.reset(self.action.get())
        self._discard_frame()
        self._render()

    def _discard_frame(self):
        self.frozen = self.latest = self.photo = None
        self.evaluated = False
        self.preview_size = (0, 0)
        self._preview_item = None
        self._erase_regions()
        self.canvas.delete("all")

    def start(self):
        import cv2

        self._release()
        try:
            self.camera = self.open_camera()
            if not self.camera.isOpened():
                self._camera_error("Camera unavailable. Check camera access and retry.")
        except (OSError, cv2.error):
            self._camera_error("Camera unavailable. Check camera access and retry.")

    def _release(self):
        import cv2

        camera, self.camera = self.camera, None
        if camera is not None:
            with suppress(OSError, cv2.error):
                camera.release()

    def _camera_error(self, message):
        self._release()
        self._discard_frame()
        self.controller.capture_error("camera_unavailable", message)
        self._render()

    def _tick(self):
        import cv2

        if self.camera is not None and self.frozen is None:
            try:
                ok, frame = self.camera.read()
                if not ok:
                    self._camera_error(
                        "Camera read failed. Reconnect the camera and retry."
                    )
                else:
                    frame_size(frame)
                    first_frame = self.latest is None
                    self.latest = frame
                    self._draw_frame(frame)
                    if first_frame:
                        self._render()
            except (OSError, cv2.error, ValueError):
                self._camera_error("Camera did not deliver a usable BGR frame. Retry.")
        self._timer = self.root.after(33, self._tick)

    def _draw_frame(self, frame):
        self._paint_preview(prepare_preview_image(frame))

    def _paint_preview(self, image):
        from PIL import Image, ImageTk

        if getattr(self, "_preview_mirrored", False):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        self.preview_size = image.size
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        if getattr(self, "_preview_item", None) is None:
            self._preview_item = self.canvas.create_image(
                0, 0, image=self.photo, anchor="nw", tags="preview"
            )
            self.canvas.tag_lower("preview")
        else:
            self.canvas.itemconfigure(self._preview_item, image=self.photo)

    def freeze(self):
        if self.latest is None or self.frozen is not None:
            return
        self.frozen = self.latest.copy()
        self.frozen.flags.writeable = False
        self._prepare_selection()

    def _prepare_selection(self):
        self._reset_selection()
        if self.selection_mode.get() == AUTOMATIC:
            self._detect_regions()
        self._render()

    def _detect_regions(self):
        import cv2

        try:
            boxes = self.detect_faces(self.frozen)
        except FaceDetectorUnavailable as error:
            self.controller.capture_error(
                "model_unavailable", str(error), module="detector"
            )
        except ValueError as error:
            self.controller.capture_error(
                "invalid_input", str(error), module="detector"
            )
        except cv2.error:
            self.controller.capture_error(
                "inference_error",
                "OpenCV could not detect face regions.",
                module="detector",
            )
        else:
            self.boxes.extend(boxes)
            width, height = frame_size(self.frozen)
            view_width, view_height = self.preview_size
            for x1, y1, x2, y2 in boxes:
                self.canvas.create_rectangle(
                    x1 * view_width / width,
                    y1 * view_height / height,
                    x2 * view_width / width,
                    y2 * view_height / height,
                    outline="#ffce65",
                    width=2,
                    tags="region",
                )
            self.hint.set(
                f"{len(boxes)} face region(s) detected. Run stubs or switch to Manual selection."
            )
            return
        self.hint.set(
            "Automatic detection unavailable for this frame. Switch to Manual selection."
        )

    def _start_drag(self, event):
        if self.frozen is None or self.evaluated or self.selection_mode.get() != MANUAL:
            return
        width, height = self.preview_size
        if 0 <= event.x < width and 0 <= event.y < height:
            self.drag_start = (event.x, event.y)

    def _drag_box(self, event):
        x, y = self.drag_start
        width, height = self.preview_size
        end_x = min(max(event.x, 0), width)
        end_y = min(max(event.y, 0), height)
        return min(x, end_x), min(y, end_y), max(x, end_x), max(y, end_y)

    def _draw_drag(self, event):
        if self.drag_start is not None:
            if self.drag_outline is not None:
                self.canvas.delete(self.drag_outline)
            self.drag_outline = self.canvas.create_rectangle(
                *self._drag_box(event), outline="#5df6cb", width=2, tags="region"
            )

    def _end_drag(self, event):
        if self.drag_start is None:
            return
        left, top, right, bottom = self._drag_box(event)
        self.drag_start = None
        if self.drag_outline is not None:
            self.canvas.delete(self.drag_outline)
            self.drag_outline = None
        if left == right or top == bottom:
            return
        width, height = frame_size(self.frozen)
        view_width, view_height = self.preview_size
        # Floor the start and ceil the end in the original half-open coordinates.
        box = (
            left * width // view_width,
            top * height // view_height,
            (right * width + view_width - 1) // view_width,
            (bottom * height + view_height - 1) // view_height,
        )
        self.boxes.append(box)
        self.canvas.create_rectangle(
            left, top, right, bottom, outline="#5df6cb", width=2, tags="region"
        )
        self.hint.set(
            f"{len(self.boxes)} region(s) selected. Run stubs or clear the selection."
        )

    def _erase_regions(self):
        self.boxes.clear()
        self.drag_start = self.drag_outline = None
        self.canvas.delete("region")

    def _reset_selection(self):
        self._erase_regions()
        self.controller.reset(self.action.get())
        self.evaluated = False
        if self.selection_mode.get() == MANUAL:
            self.hint.set("Draw a rectangle around one face, then select Run stubs.")
        else:
            self.hint.set(
                "No regions selected. Resume preview and freeze again to detect faces."
            )

    def _clear_regions(self):
        if self.frozen is None:
            return
        if self.controller.state.ui_state in {"spoof", "uncertain"}:
            self._resume_preview()
            return
        self._reset_selection()
        self._render()

    def evaluate(self):
        if (
            self.frozen is None
            or self.evaluated
            or self.controller.state.ui_state == "error"
        ):
            return
        self.evaluated = True
        request = self.controller.begin_capture(self.frozen, self.boxes)
        if request is not None:
            height, width, _ = request.face_crop.shape
            self.hint.set(
                f"Source crop {width} × {height} pixels. Resume preview for a fresh frame."
            )
            run_stubs(self.controller, request, StubAdapters(self.scenario.get()))
        else:
            self.hint.set(
                "Resume preview for a fresh frame or adjust the region selection."
            )
        self._render()

    def _render(self):
        state = self.controller.state
        messages = {
            "idle": "Frame frozen" if self.frozen is not None else "Preview",
            "checking": "Checking liveness",
            "live": "Live face (stub)",
            "spoof": "Spoof (stub). Attempt blocked. Retry with a fresh frame.",
            "uncertain": "Uncertain liveness (stub). Retry with a fresh frame.",
        }
        status = messages.get(
            state.ui_state, state.errors[0].message if state.errors else state.ui_state
        )
        emotion = state.emotion.label if state.emotion else "Unavailable"
        self.summary.set(
            f"{status}\nDemo identity: {self.controller.demo_identity}\n"
            f"Demo emotion: {emotion}\nSimulated values, not inferred from image content.\nSaving disabled in demo"
        )
        self.payload.configure(state="normal")
        self.payload.delete("1.0", "end")
        self.payload.insert("1.0", json.dumps(asdict(state), indent=2, allow_nan=False))
        self.payload.configure(state="disabled")
        self.freeze_button.configure(
            state="normal"
            if self.latest is not None and self.frozen is None
            else "disabled"
        )
        can_run = (
            self.frozen is not None and not self.evaluated and state.ui_state != "error"
        )
        self.run_button.configure(state="normal" if can_run else "disabled")
        self.clear_button.configure(
            state="normal" if self.frozen is not None else "disabled"
        )
        if self.frozen is None:
            self.hint.set(
                "Freeze a preview frame to select a face region."
                if self.latest is not None
                else status
            )

    def close(self):
        self.root.after_cancel(self._timer)
        try:
            self._release()
        finally:
            self.root.destroy()


class LiveWebcamApp:
    """Default automatic attendance preview. All methods execute on the Tk thread."""

    ANALYSIS_INTERVAL = (
        0.4  # Minimum spacing between analysis samples, not preview FPS.
    )
    MAX_AGE = 1.0
    STANDARD_CONTROLS_SPACE = 44
    STANDARD_CONTROLS_GAP = 10

    def __init__(
        self,
        root,
        open_camera,
        *,
        source_name,
        scenario=SCENARIOS[0],
        detector=None,
        adapters=None,
        diagnostics=False,
        runtime_info=None,
        liveness_trial=None,
        enable_trial_profiles=False,
        local_profile_store=None,
        released_liveness=None,
        attendance=None,
        attendance_verification=None,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self._mode_supported = True
        self._developer_mode = False
        self._source_name = source_name
        self._runtime_info = dict(runtime_info or {})
        self._completed_diagnostic = (
            None  # One frozen historical envelope, never authorization.
        )
        self._identity_overlay = None
        self._preview_mirrored = (
            True  # Applied only to the displayed image and overlay coordinates.
        )
        self._registration_overlay = None
        self._local_workspace = local_profile_store is not None
        self.released_liveness = released_liveness
        self._attendance_verification = attendance_verification
        self.liveness_preview = None
        if released_liveness is not None:
            if not self._local_workspace or liveness_trial is not None:
                raise ValueError(
                    "AENet-Full requires a local workspace and no experimental checkpoint comparison."
                )
            from .liveness_preview import LivenessPreview

            self.liveness_preview = LivenessPreview(
                released_liveness.provenance["threshold"]
            )
        loaded_gallery = local_profile_store.load() if self._local_workspace else None
        self.live = LiveController(unverified_preview=self._local_workspace)
        self.controller = self.live.controller
        self.adapters = adapters if adapters is not None else StubAdapters(scenario)
        self._trained_emotion_enabled = (
            getattr(self.adapters, "trained_emotion", False) is True
        )
        self._trained_recognition_enabled = (
            getattr(self.adapters, "trained_recognition", False) is True
        )
        self._controlled_registration = (
            getattr(self.adapters, "controlled_registration", False) is True
        )
        enable_profiles = self._local_workspace or enable_trial_profiles
        if enable_profiles and (
            not (self._local_workspace or liveness_trial is not None)
            or not self._trained_recognition_enabled
            or self._controlled_registration
        ):
            raise ValueError(
                "Local profiles require trained recognition in a local workspace or the three-model trial."
            )
        self._registration_worker = (
            SingleWorker(self.adapters.matcher.prepare, "test-enrollment")
            if self._controlled_registration
            else None
        )
        self._enrollment_token = None
        from .presentation import StablePresentation

        self.presentation = StablePresentation(
            unverified_preview=self._local_workspace,
            stable_trial=self._local_workspace or liveness_trial is not None,
        )
        self._presentation_request = None
        self._registration_requested = False
        self.metrics = Diagnostics(diagnostics)
        # Informational Emotion history has no identity/liveness authority.
        from .emotion_presentation import EmotionPresentation

        emotion_model = getattr(
            getattr(self.adapters, "emotion_adapter", None), "inference", None
        )
        self._emotion_architecture = getattr(
            emotion_model, "architecture", "EmotionCNN"
        )
        self.emotion_presentation = (
            EmotionPresentation(
                self.metrics,
                prediction_interval=getattr(emotion_model, "prediction_interval", 0.0),
            )
            if self._trained_emotion_enabled
            else None
        )
        self.adapters.diagnostics = self.metrics
        self.feed = CameraFeed(open_camera, self.metrics)
        self.detector = SingleWorker(
            detector
            if detector is not None
            else ScaledFaceDetector(YuNetFaceDetector()),
            "detector",
            self.metrics,
        )
        evaluator = (
            partial(evaluate_adapters, unverified_preview=True)
            if self._local_workspace
            else evaluate_adapters
        )
        if released_liveness is not None:
            evaluator = released_liveness.evaluate
        self.inference = SingleWorker(evaluator, "adapters", self.metrics)
        self._liveness_trial_worker = (
            SingleWorker(liveness_trial, "liveness_trial", self.metrics)
            if liveness_trial is not None
            else None
        )
        self._liveness_trial_result = None
        self._trial_profiles = None
        self._closed = False
        self._generation = 0
        self._drawn_sequence = None
        self._next_detection = 0
        self._detection_frame = None
        self._detector_failed = False
        self._opened_at = monotonic()
        self.photo = None
        self.preview_size = (0, 0)
        self._preview_viewport_size = (640, 360)
        self._preview_resize_pending = False
        self._last_render = None
        self._last_summary = None
        self._last_payload = None
        self._region_key = None
        self._preview_item = None
        self._sampled_sequence = 0
        self._last_tick_started = self._last_preview_at = self._next_tick_at = None
        self._next_debug_update = 0.0

        title = (
            "Facial Attendance · Local workspace"
            if self._local_workspace
            else "Facial Attendance"
        )
        automatic_record = (
            attendance_verification is not None
            and attendance_verification.automatic_record
        )
        if attendance_verification is not None:
            title += " · Attendance test"
            if automatic_record:
                title += " · Auto record"
        root.title(title)
        # A model's changing text must not resize the window or move the video.
        root.protocol("WM_DELETE_WINDOW", self.close)
        if self._local_workspace:
            from .dashboard import VIEWFINDER, Dashboard, apply_theme

            apply_theme(root)
            width = max(1, min(1240, root.winfo_screenwidth() - 48))
            preferred_height = max(620, round(width * 0.65 * 9 / 16) + 150)
            height = max(1, min(preferred_height, root.winfo_screenheight() - 96))
            root.geometry(f"{width}x{height}")
            root.minsize(min(820, width), min(560, height))
        else:
            root.geometry("1280x760")
            root.minsize(1120, 700)
        panel = ttk.Frame(root, padding=16)
        panel.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        self._dashboard = None
        self.source_label = tk.StringVar(
            value="Generated test preview"
            if self._controlled_registration
            else "Live preview"
        )
        if self._local_workspace:
            panel.columnconfigure(0, weight=65, uniform="workspace")
            panel.columnconfigure(1, weight=35, uniform="workspace")
            retry_attendance = (
                attendance_verification.reset
                if attendance_verification is not None
                else None
            )
            self._dashboard = Dashboard(
                panel,
                restart_camera=self.retry,
                switch_mode=self._switch_mode,
                trained_liveness=released_liveness is not None,
                attendance=attendance,
                retry_attendance=retry_attendance,
                retrying_after_spoof=(
                    attendance_verification.retrying_after_spoof
                    if attendance_verification is not None
                    else None
                ),
                automatic_record=automatic_record,
            )
            camera_parent = self._dashboard.camera_card
            self.source_label.set("Live Camera")
            ttk.Label(
                camera_parent,
                textvariable=self.source_label,
                font=("TkDefaultFont", 14, "bold"),
            ).grid(row=0, column=0, sticky="w", pady=(0, 9))
        else:
            panel.columnconfigure(0, weight=1)
            camera_parent = panel
            ttk.Label(
                panel, text="Facial Attendance", font=("TkDefaultFont", 20, "bold")
            ).grid(row=0, column=0, sticky="w")
            banner = (
                "Liveness SIMULATED • Identity UNVERIFIED"
                if self._trained_recognition_enabled
                else "DEMO / SIMULATED OUTPUT"
            )
            if self._controlled_registration:
                banner = "CONTROLLED TEST • Liveness SIMULATED • Identity UNVERIFIED"
            elif enable_trial_profiles:
                banner = "3-MODEL TRIAL • Test profiles only • Identity UNVERIFIED"
            elif self._liveness_trial_worker is not None:
                banner = "3-MODEL TRIAL • Identity UNVERIFIED • Saving disabled"
            banner_area = ttk.Frame(panel, width=420, height=55)
            banner_area.grid(row=0, column=1, sticky="w", padx=(18, 0))
            banner_area.pack_propagate(False)
            ttk.Label(
                banner_area,
                text=banner,
                foreground="#a04400",
                wraplength=410,
                anchor="nw",
                font=("TkDefaultFont", 13, "bold"),
            ).pack(fill="both", expand=True)
            ttk.Label(panel, textvariable=self.source_label).grid(
                row=1, column=0, sticky="w", pady=(4, 10)
            )

        viewport = self._preview_container = ttk.Frame(
            camera_parent,
            width=1 if self._dashboard else 640,
            height=1 if self._dashboard else 360,
        )
        viewport.grid(row=1 if self._dashboard else 2, column=0, sticky="nsew")
        viewport.bind("<Configure>", self._resize_preview)
        self.canvas = tk.Canvas(
            viewport,
            background=VIEWFINDER if self._dashboard else "#162332",
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.place(relx=0.5, y=0, anchor="n", width=640, height=360)
        controls = self._camera_controls = ttk.Frame(camera_parent)
        if self._dashboard:
            self._place_standard_camera_controls(360)
        else:
            controls.grid(row=3, column=0, sticky="nw", pady=8)
            ttk.Button(
                controls, text="Retry / restart camera", command=self.retry
            ).pack(side="left")
        self.hint = tk.StringVar(value="Opening camera…")
        ttk.Label(
            controls,
            textvariable=self.hint,
            wraplength=420,
            style="Quiet.TLabel" if self._dashboard else "TLabel",
        ).pack(side="left", padx=0 if self._dashboard else 12)
        if self._dashboard:
            results = self.results_panel = self._dashboard.results
            self.mode_button = self._dashboard.mode_button
        else:
            results = self.results_panel = ttk.Frame(panel, width=420, height=540)
            results.grid(row=2, column=1, rowspan=3, sticky="nw", padx=(18, 0))
            results.pack_propagate(False)
            heading = ttk.Frame(results)
            heading.pack(fill="x", pady=(0, 6))
            ttk.Label(
                heading, text="Live results", font=("TkDefaultFont", 16, "bold")
            ).pack(side="left")
            self.mode_button = ttk.Button(
                heading, text="Developer Mode", command=self._switch_mode
            )
            self.mode_button.pack(side="right")
        self.summary = tk.StringVar()
        self.display_status = tk.StringVar()
        self.display_identity = tk.StringVar()
        self.display_emotion = tk.StringVar()
        self.display_liveness = tk.StringVar()
        if self._dashboard:
            self.result_cells = tuple(value for value, _ in self._dashboard.cards)
        else:
            status_label = fixed_result_label(
                results,
                36,
                textvariable=self.display_status,
                font=("TkDefaultFont", 10, "bold"),
            )
            two_line_liveness = (
                self._liveness_trial_worker is not None or released_liveness is not None
            )
            rows = (
                (60, self.display_identity),
                (42, self.display_emotion),
                (70 if two_line_liveness else 42, self.display_liveness),
            )
            self.summary_label, emotion_label, liveness_label = (
                fixed_result_label(
                    results, height, textvariable=value, font=("TkDefaultFont", 12)
                )
                for height, value in rows
            )
            self.result_cells = (
                status_label,
                self.summary_label,
                emotion_label,
                liveness_label,
            )
            explanation = "Recent preview results only. Genuine registration and attendance saving are disabled."
            if self._liveness_trial_worker is not None:
                explanation = "Liveness scores are experimental. The registration and attendance gate uses simulated liveness."
            if self._controlled_registration:
                explanation += " Generated test input; memory gallery only."
            fixed_result_label(results, 65, text=explanation)
        if self._trained_emotion_enabled:
            self.model_info = tk.StringVar(
                value=f"Selected {self._emotion_architecture}; provenance appears after checkpoint validation."
            )
        profile_controls = results
        if self._local_workspace:
            profile_controls = self._profile_controls = ttk.Frame(
                results, style="Card.TFrame"
            )
            ttk.Separator(profile_controls).pack(fill="x", pady=(3, 7))
        if self._trained_recognition_enabled:
            self.registration_prompt = tk.StringVar()
            if self._dashboard:
                ttk.Label(
                    profile_controls,
                    text="Local profiles",
                    style="Card.TLabel",
                    font=("TkDefaultFont", 12, "bold"),
                ).pack(anchor="w", pady=(4, 2))
                prompt_slot, _ = self._dashboard.message_slot(
                    profile_controls,
                    textvariable=self.registration_prompt,
                    style="CardQuiet.TLabel",
                )
                self._dashboard.reserve_message_lines(prompt_slot, 3)
                prompt_slot.pack(fill="x")
            else:
                fixed_result_label(
                    profile_controls, 55, textvariable=self.registration_prompt
                )
            self.registration_button = ttk.Button(
                profile_controls,
                text="Registration options…",
                command=self._show_registration,
                style="Primary.TButton" if self._dashboard else "TButton",
            )
            self.registration_button.pack(anchor="w", pady=6)
        if enable_profiles:
            from .trial_profiles import TrialProfiles

            self._trial_profiles = TrialProfiles(
                panel,
                self.adapters,
                self._trial_gallery_changed,
                store=local_profile_store,
                loaded_gallery=loaded_gallery,
                liveness_provenance=(
                    released_liveness.provenance
                    if released_liveness is not None
                    else None
                ),
                liveness_preview=self.liveness_preview,
            )
            if self._local_workspace:
                from .registration_overlay import RegistrationOverlay

                self._registration_overlay = RegistrationOverlay(self.canvas)
                self.registration_button.configure(text="Register new person")
                ttk.Button(
                    profile_controls,
                    text="Clear local profiles…",
                    command=self._trial_profiles.clear,
                ).pack(anchor="w")
            else:
                self.registration_button.configure(text="Create test profile…")
        elif self._trained_recognition_enabled:
            self.registration_form = ttk.LabelFrame(
                panel, text="Registration", padding=8
            )
            self.registration_form.grid(row=4, column=0, sticky="new", pady=8)
            self.registration_id, self.registration_name = (
                tk.StringVar(),
                tk.StringVar(),
            )
            ttk.Label(self.registration_form, text="New person ID").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Entry(self.registration_form, textvariable=self.registration_id).grid(
                row=0, column=1
            )
            ttk.Label(self.registration_form, text="Display name").grid(
                row=1, column=0, sticky="w"
            )
            ttk.Entry(self.registration_form, textvariable=self.registration_name).grid(
                row=1, column=1
            )
            self.registration_message = tk.StringVar(
                value=(
                    "CONTROLLED TEST: five generated samples; no genuine enrollment or records."
                    if self._controlled_registration
                    else "Blocked: trained liveness is unavailable. No genuine enrollment."
                )
            )
            ttk.Label(
                self.registration_form,
                textvariable=self.registration_message,
                wraplength=610,
            ).grid(row=2, column=0, columnspan=2, sticky="w")
            self.registration_submit = ttk.Button(
                self.registration_form,
                text="Enroll generated test samples"
                if self._controlled_registration
                else "Genuine enrollment blocked",
                command=self._start_test_enrollment,
                state="normal" if self._controlled_registration else "disabled",
            )
            self.registration_submit.grid(row=3, column=0, columnspan=2, sticky="w")
            self.registration_form.grid_remove()
        profile_hint = (
            "Profiles are saved on this device."
            if self._local_workspace
            else "Test profiles stay in memory. No attendance records."
            if enable_trial_profiles
            else "Saving disabled • No attendance or enrollment records"
        )
        profile_label = ttk.Label(
            profile_controls,
            text=profile_hint,
            wraplength=330,
            style="CardQuiet.TLabel" if self._dashboard else "TLabel",
        )
        profile_label.pack(fill="x" if self._dashboard else "none", anchor="w", pady=10)
        if self._dashboard:
            self._dashboard.wrap_label(profile_label)
        self.diagnostics_button = ttk.Button(
            results, text="Open / close diagnostics", command=self._toggle_debug
        )
        # Standard Mode starts with developer-only controls unmapped.
        self._diagnostics_view = None  # Created once, on the first explicit open.
        self.debug = self.payload = None
        # Compatibility values for existing instrumentation; technical content
        # is now read-only inside the independent diagnostics window.
        self.emotion_display_info = tk.StringVar()
        self._debug_visible = False
        self._render()
        self._timer = root.after(0, self._tick)

    _paint_preview = WebcamApp._paint_preview

    def _place_standard_camera_controls(self, image_height):
        self._camera_controls.place(
            in_=self._preview_container,
            x=0,
            y=image_height + self.STANDARD_CONTROLS_GAP,
            relwidth=1,
        )

    def _resize_preview(self, event):
        size = (max(1, event.width), max(1, event.height))
        if size != self._preview_viewport_size:
            self._preview_viewport_size = size
            self._preview_resize_pending = True

    def _draw_frame(self, frame):
        width, height = self._preview_viewport_size
        standard_layout = getattr(self, "_dashboard", None) is not None
        if standard_layout:
            height = max(1, height - self.STANDARD_CONTROLS_SPACE)
        image = prepare_preview_image(frame, viewport=(width, height))
        if image.size != self.preview_size:
            self.canvas.place_configure(width=image.width, height=image.height)
            if standard_layout:
                self._place_standard_camera_controls(image.height)
        self._paint_preview(image)

    def _toggle_debug(self):
        # Retain the original embedding API for legacy headless harnesses.
        if not getattr(self, "_mode_supported", False):
            self._debug_visible = not self._debug_visible
            self.debug.pack(
                fill="both", expand=True
            ) if self._debug_visible else self.debug.pack_forget()
            self._last_render = None
            self._next_debug_update = 0.0
            self._render()
            return
        if self._debug_visible:
            self._close_debug()
        else:
            self._set_mode(True, open_diagnostics=False)
            if self._diagnostics_view is None:
                from .developer_view import DeveloperDiagnostics

                self._diagnostics_view = DeveloperDiagnostics(
                    self.root,
                    self._close_debug,
                    latest_completed=lambda: self._completed_diagnostic,
                    unverified_workspace=getattr(self, "_local_workspace", False),
                )
                self.debug = self._diagnostics_view.window
                self.payload = self._diagnostics_view.payload
            self._debug_visible = True
            self._diagnostics_view.show()
            self._last_payload = None
            self._last_render = None
            self._next_debug_update = 0.0
            self._render()

    def _close_debug(self):
        self._debug_visible = False
        if self._diagnostics_view is not None:
            self._diagnostics_view.hide()

    def _switch_mode(self):
        self._set_mode(not self._developer_mode)

    def _set_mode(self, developer, *, open_diagnostics=True):
        self._developer_mode = bool(developer)
        if getattr(self, "_dashboard", None) is not None:
            if developer:
                self._profile_controls.pack(fill="x")
                if self._trial_profiles.active:
                    self._trial_profiles.form.grid()
            else:
                self._profile_controls.pack_forget()
                self._trial_profiles.form.grid_remove()
        if developer:
            self.diagnostics_button.pack(anchor="w", pady=(4, 8))
        else:
            self.diagnostics_button.pack_forget()
        self.mode_button.configure(
            text="Standard Mode" if developer else "Developer Mode"
        )
        self.source_label.set(
            f"Developer Mode • {self._source_name}"
            if developer
            else (
                "Live Camera"
                if getattr(self, "_dashboard", None) is not None
                else "Generated test preview"
                if self._controlled_registration
                else "Live preview"
            )
        )
        box_hold_ms = round(self.live.stability.hold_seconds * 1000)
        preview_hint = (
            "Show one face to the camera."
            if getattr(self, "_local_workspace", False)
            else "Show one face to the camera. Preview results are unverified."
        )
        self.hint.set(
            f"Automatic sampled detection • Previous box ≤{box_hold_ms} ms • Display history ≤800 ms; never authorization."
            if developer
            else preview_hint
        )
        self._hint_rendered = True
        if not developer:
            self._close_debug()
        self._last_render = None
        self._render()
        if developer and open_diagnostics and not self._debug_visible:
            self._toggle_debug()

    def start(self):
        self.retry()

    def retry(self):
        if self._closed:
            return
        self.live.reset()
        self._registration_requested = False
        attendance_verification = getattr(self, "_attendance_verification", None)
        if attendance_verification is not None:
            attendance_verification.reset()
        profiles = getattr(self, "_trial_profiles", None)
        if profiles is not None:
            profiles.invalidate_source()
        presentation = getattr(self, "presentation", None)
        if presentation is not None:
            presentation.reset()
        liveness_preview = getattr(self, "liveness_preview", None)
        if liveness_preview is not None:
            liveness_preview.clear()
        emotion_display = getattr(self, "emotion_presentation", None)
        if emotion_display is not None:
            emotion_display.reset("retry")
        self._presentation_request = None
        self._enrollment_token = None
        if getattr(self, "_trained_recognition_enabled", False):
            self.controller.reset("attendance")
        self._generation = self.feed.restart()
        self._opened_at = monotonic()
        self._drawn_sequence = None
        self._sampled_sequence = 0
        self._next_detection = 0
        self._detector_failed = False
        self.photo = None
        self._preview_item = None
        self._region_key = None
        self.canvas.delete("all")
        if getattr(self, "_identity_overlay", None) is not None:
            self._identity_overlay.reset()
        if getattr(self, "_dashboard", None) is not None:
            from .dashboard import camera_status

            self._dashboard.set_camera(camera_status(False, False))
        self._render()

    def _tick(self):
        if self._closed:
            return
        now = monotonic()
        self._ui_time = now
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            previous = getattr(self, "_last_tick_started", None)
            if previous is not None:
                metrics.duration("gui_interval", now - previous)
                metrics.duration(
                    "event_loop_delay",
                    max(0.0, now - getattr(self, "_next_tick_at", now)),
                )
            self._last_tick_started = now
        packet = self.feed.latest()
        liveness_preview = getattr(self, "liveness_preview", None)
        usable = (
            packet is not None
            and packet.generation == self._generation
            and packet.error is None
            and now - packet.captured_at <= self.MAX_AGE
        )
        if usable:
            new_frame = packet.sequence != self._drawn_sequence
            if new_frame:
                presentation = getattr(self, "presentation", None)
                if presentation is not None:
                    presentation.observe_preview(packet.frame, packet.captured_at)
                emotion_display = getattr(self, "emotion_presentation", None)
                if emotion_display is not None:
                    emotion_display.observe_preview(packet.frame, packet.captured_at)
            if new_frame or getattr(self, "_preview_resize_pending", False):
                started = monotonic()
                self._draw_frame(packet.frame)
                self._preview_resize_pending = False
                if metrics is not None:
                    self.metrics.duration("render", monotonic() - started)
            if new_frame:
                if metrics is not None:
                    self.metrics.duration(
                        "preview_age", monotonic() - packet.captured_at
                    )
                    self.metrics.count("previewed")
                    previous = getattr(self, "_last_preview_at", None)
                    if previous is not None:
                        self.metrics.duration("preview_interval", started - previous)
                    self._last_preview_at = started
                    self.metrics.count(
                        "preview_dropped",
                        max(0, packet.sequence - (self._drawn_sequence or 0) - 1),
                    )
                self._drawn_sequence = packet.sequence
        elif packet is not None or now - self._opened_at > 5:
            self.live.fail(
                "camera_unavailable", "Camera unavailable — check access and retry."
            )
            self._clear_presentation(now, "camera_failure")
            self.canvas.delete("all")
            if getattr(self, "_identity_overlay", None) is not None:
                self._identity_overlay.reset()
            self.photo = None
            self._preview_item = None
            self._region_key = None

        # Rendering itself can take time. Recheck age at acceptance time,
        # rather than trusting the timestamp from the start of the Tk tick.
        accept_time = monotonic()
        if usable and accept_time - packet.captured_at > self.MAX_AGE:
            usable = False
            self.live.fail(
                "camera_unavailable",
                "Camera frame expired — waiting for a fresh frame.",
            )
            self._clear_presentation(accept_time, "camera_frame_expired")
            self.canvas.delete("all")
            if getattr(self, "_identity_overlay", None) is not None:
                self._identity_overlay.reset()
            self.photo = self._preview_item = self._region_key = None
        # Invalidate expired context BEFORE consuming any worker completions.
        expired = self.live.expire(accept_time, self.MAX_AGE)
        if expired:
            self._clear_presentation(accept_time, affect_emotion=False)
        metrics = getattr(self, "metrics", None)
        if metrics is not None and expired:
            metrics.count("analysis_expired")
        profiles = getattr(self, "_trial_profiles", None)
        if profiles is not None and (not usable or expired):
            profiles.invalidate_source()
        attendance_verification = getattr(self, "_attendance_verification", None)
        attendance_paused = False
        attendance_verified = False
        if attendance_verification is not None:
            attendance_verification.sync_session()
            attendance_paused = (
                bool(profiles and profiles.active) or self._registration_requested
            )
            if attendance_paused:
                attendance_verification.invalidate()
            elif not usable or expired:
                attendance_verification.invalidate(
                    "stale" if expired else "capture_error"
                )
        trial_worker = getattr(self, "_liveness_trial_worker", None)
        if trial_worker is not None:
            trial_result = trial_worker.take()
            if (
                trial_result is not None
                and usable
                and trial_result.token == self.live.token
            ):
                self._liveness_trial_result = trial_result
                presentation = getattr(self, "presentation", None)
                if presentation is not None:
                    presentation.accept_trial(
                        trial_result, self.live.sample_time, accept_time
                    )
        # Free a finished adapter slot before dispatching the next eligible
        # detection. The original token/context still governs acceptance.
        completed = self.inference.take()
        if completed is not None and usable:
            prediction = None
            if liveness_preview is not None and completed.error is None:
                batch, prediction = completed.value
                completed = replace(completed, value=batch)
            source = getattr(self, "_presentation_request", None)
            attendance_current = (
                attendance_verification is not None
                and not attendance_paused
                and source is not None
                and completed.token == self.live.token == source[0]
                and len(self.live.boxes) == 1
            )
            applied = self.live.completed(completed)
            if applied and self.controller.state.context is not None:
                from .diagnostic_snapshot import CompletedDiagnostic, utc_timestamp

                self._completed_diagnostic = CompletedDiagnostic(
                    self.controller.state, utc_timestamp()
                )
            presentation = getattr(self, "presentation", None)
            if attendance_current:
                if completed.error is None and self.live.token is None:
                    attendance_verification.invalidate("invalid_verification")
                else:
                    attendance_verified = attendance_verification.accept(
                        completed.token,
                        source[1],
                        completed.value,
                        prediction,
                        error=completed.error,
                    )
            if (
                liveness_preview is not None
                and source is not None
                and completed.token == self.live.token
                and source[0] == completed.token
                and len(self.live.boxes) == 1
                and self.controller.state.context == source[1].context
            ):
                liveness_preview.accept(
                    completed.token,
                    source[1].bbox,
                    source[2],
                    self.controller.state,
                    prediction,
                )
                if profiles is not None:
                    profiles.observe_liveness(completed.token, source[1], prediction)
                    if getattr(
                        self, "_registration_requested", False
                    ) and self._can_start_local_registration(accept_time):
                        self._registration_requested = False
                        profiles.show()
            if applied and source is not None and presentation is not None:
                token, request, captured_at, _ = source
                if token == completed.token:
                    presentation.accept(
                        token, request, self.controller.state, captured_at, accept_time
                    )
            emotion_display = getattr(self, "emotion_presentation", None)
            if emotion_display is not None and source is not None:
                token, request, captured_at, sampled_emotion = source
                if completed.token == token and sampled_emotion:
                    if completed.error is not None:
                        emotion_display.clear("adapter_worker_failure")
                    else:
                        accepted = emotion_display.accept(
                            token, request, completed.value, captured_at, accept_time
                        )
                        if accepted and not applied:
                            emotion_display.count(
                                "accepted_original_superseded_context"
                            )
            self._presentation_request = None
            if self.controller.state.ui_state in {"spoof", "uncertain", "error"}:
                self._enrollment_token = None
                if profiles is not None and self.controller.state.ui_state == "error":
                    profiles.invalidate_source()
                # Recognition errors/uncertain liveness invalidate authority,
                # but cannot erase a separate, fresh informational Emotion.
                self._clear_presentation(
                    accept_time,
                    "spoof",
                    affect_emotion=self.controller.state.ui_state == "spoof",
                )
            if metrics is not None and not applied:
                metrics.count("adapter_discarded")
        elif completed is not None and metrics is not None:
            metrics.count("adapter_discarded")
            self._presentation_request = None
        detected = self.detector.take()
        if detected is not None:
            if metrics is not None and (
                not usable or detected.token != self.live.token
            ):
                metrics.count("detection_discarded")
            if usable:
                current_detection = detected.token == self.live.token
                request = self.live.detected(detected, self._detection_frame)
                if profiles is not None and current_detection:
                    profiles.observe(detected.token, request, self.live.sample_time)
                if (
                    attendance_verification is not None
                    and not attendance_paused
                    and current_detection
                    and request is None
                ):
                    phase = self.controller.state.ui_state
                    attendance_verification.invalidate(
                        phase
                        if phase in ("no_face", "multiple_faces")
                        else "capture_error"
                    )
                emotion_display = getattr(self, "emotion_presentation", None)
                if emotion_display is not None and current_detection:
                    if self.controller.state.ui_state == "error":
                        emotion_display.clear("detector_failure")
                    else:
                        emotion_display.detected(
                            detected.token,
                            self._detection_frame,
                            self.live.boxes,
                            self.live.sample_time,
                        )
                presentation = getattr(self, "presentation", None)
                if presentation is not None:
                    if detected.token == self.live.token and request is not None:
                        presentation.detected(
                            detected.token,
                            self._detection_frame,
                            self.live.boxes,
                            self.live.sample_time,
                        )
                    elif self.controller.state.ui_state in {
                        "no_face",
                        "multiple_faces",
                        "error",
                    }:
                        self._clear_presentation(accept_time, affect_emotion=False)
                self._detector_failed = bool(
                    self.controller.state.errors
                    and self.controller.state.errors[0].module == "detector"
                )
                if request is not None:
                    if trial_worker is not None and not trial_worker.submit(
                        detected.token, request.face_crop
                    ):
                        self.metrics.count("liveness_trial_busy_dropped")
                    sampled_emotion = emotion_display is None or emotion_display.due(
                        self.live.sample_time
                    )
                    submitted = self.inference.submit(
                        detected.token, (request, self.adapters, sampled_emotion)
                    )
                    if submitted:
                        self._presentation_request = (
                            detected.token,
                            request,
                            self.live.sample_time,
                            sampled_emotion,
                        )
                        if (
                            attendance_verification is not None
                            and not attendance_paused
                        ):
                            attendance_verification.expect(
                                detected.token, request, self.live.sample_time
                            )
                        if emotion_display is not None and sampled_emotion:
                            emotion_display.register(
                                detected.token, request, self.live.sample_time
                            )
                    elif attendance_verification is not None and not attendance_paused:
                        attendance_verification.invalidate()
                    if metrics is not None and not submitted:
                        metrics.count("adapter_busy_dropped")
            self._detection_frame = None
        if profiles is not None:
            profiles.poll()
        if liveness_preview is not None:
            if usable and not self._detector_failed:
                # An in-flight frame belongs to the scan in which it was captured.
                pending_at = (
                    self.live.sample_time
                    if self.detector.busy or self.inference.busy
                    else None
                )
                liveness_preview.advance(packet.captured_at, pending_at=pending_at)
            else:
                liveness_preview.clear()
        # Sample the newest camera frame only when the single detector slot is
        # free. Starting a sample invalidates every previous capture/context.
        # With AENet, let the joined result finish and render once before taking
        # the next sample. Capture/preview keep running while the worker is busy.
        if (
            usable
            and not self._detector_failed
            and not self.detector.busy
            and (
                liveness_preview is None
                or (not self.inference.busy and completed is None)
            )
            and now >= self._next_detection
            and (packet.generation, packet.sequence) != self.live.token
        ):
            token = packet.generation, packet.sequence
            if self.detector.submit(token, packet.frame):
                if attendance_verification is not None and not attendance_paused:
                    attendance_verification.pending_capture()
                if metrics is not None:
                    previous = getattr(self, "_sampled_sequence", 0)
                    metrics.count(
                        "analysis_unsampled", max(0, packet.sequence - previous - 1)
                    )
                    metrics.count("analysis_started")
                    if self.live.token is not None:
                        metrics.count("analysis_superseded")
                self._sampled_sequence = packet.sequence
                self._detection_frame = packet.frame
                presentation = getattr(self, "presentation", None)
                if presentation is not None:
                    presentation.begin_sample(token, packet.frame, packet.captured_at)
                self.live.begin_sample(
                    token, packet.captured_at, action=self.controller.action
                )
                emotion_display = getattr(self, "emotion_presentation", None)
                if emotion_display is not None:
                    emotion_display.new_context()
                self._next_detection = now + self.ANALYSIS_INTERVAL
        self._poll_test_enrollment()
        if getattr(self, "_dashboard", None) is not None:
            from .dashboard import camera_status

            self._dashboard.set_camera(
                camera_status(usable, packet is not None or now - self._opened_at > 5)
            )
            if attendance_verified and attendance_verification.automatic_record:
                receipt = attendance_verification.record_automatically()
                if receipt is not None:
                    self._dashboard.attendance_view.show_record_result(receipt)
        self._draw_regions()
        self._render()
        if metrics is not None:
            metrics.duration("gui_tick", monotonic() - now)
            metrics.report()
        # Maintain a frame-period budget rather than adding 33 ms after work.
        finished = monotonic()
        delay = max(1, round((1 / 30 - (finished - now)) * 1000))
        self._next_tick_at = finished + delay / 1000
        self._timer = self.root.after(delay, self._tick)

    def current_liveness_trial(self):
        result = getattr(self, "_liveness_trial_result", None)
        if (
            result is not None
            and result.token == self.live.token
            and len(self.live.boxes) == 1
            and self.controller.state.context is not None
            and 0 <= monotonic() - self.live.sample_time <= self.MAX_AGE
        ):
            return result
        return None

    def _liveness_trial_text(self, result):
        heading = "Liveness trial · latest sample (rounded score / threshold)"
        if result is None:
            return f"{heading}\nNo current score"
        if result.error is not None:
            return f"{heading}\nScoring failed — try again."
        rows = [
            f"{sample.variant}: {sample.score:.2f} / {sample.threshold:.3f} · "
            + ("Spoof threshold met" if sample.threshold_met else "Below threshold")
            for sample in result.value
        ]
        return "\n".join([heading, *rows])

    def _display_regions(self, now):
        boxes, size, unconfirmed = self.live.visual_regions(now)
        if (
            getattr(self, "_local_workspace", False)
            and len(boxes) == 1
            and not self.presentation.region_visible(
                self.live.token, now, previous=unconfirmed
            )
        ):
            return [], None, unconfirmed
        return boxes, size, unconfirmed

    def _draw_regions(self):
        now = monotonic()
        boxes, size, unconfirmed = self._display_regions(now)
        mirrored = getattr(self, "_preview_mirrored", False)
        local = getattr(self, "_local_workspace", False)
        dashed = unconfirmed and not local
        dashboard = getattr(self, "_dashboard", None)
        accent = None
        if dashboard is not None and len(boxes) == 1:
            from .dashboard import TONES

            accent = TONES["neutral"]
            state = self.controller.state
            if not unconfirmed and state.ui_state == "preview":
                if (
                    state.recognition is not None
                    and state.recognition.accepted is False
                ):
                    accent = TONES["warning"]
                preview = self.liveness_preview
                if (
                    preview is not None
                    and preview.last_check is not None
                    and preview.last_check.decision == "spoof"
                    and preview.box == self.live.boxes[0]
                    and preview.current(self.live, now, self.MAX_AGE) is not None
                ):
                    accent = TONES["error"]
        key = (tuple(boxes), size, self.preview_size, dashed, mirrored, accent)
        previous_key = getattr(self, "_region_key", None)
        if key == previous_key:
            return
        self._region_key = key
        from .identity_overlay import preview_box

        if (
            local
            and len(boxes) == 1
            and size is not None
            and min(self.preview_size) > 0
        ):
            coordinates = preview_box(
                boxes[0], size, self.preview_size, mirrored=mirrored
            )
            if dashboard is not None:
                from .dashboard import corner_segments

                segments = corner_segments(coordinates)
                items = (
                    getattr(self, "_corner_items", ())
                    if previous_key is not None
                    else ()
                )
                if not items:
                    self.canvas.delete("region")
                    self._corner_items = tuple(
                        self.canvas.create_line(
                            *segment, fill=accent, width=2, tags="region"
                        )
                        for segment in segments
                    )
                else:
                    for item, segment in zip(items, segments):
                        self.canvas.coords(item, *segment)
                        if previous_key[-1] != accent:
                            self.canvas.itemconfigure(item, fill=accent)
                return
            item = (
                getattr(self, "_single_region_item", None)
                if previous_key is not None
                else None
            )
            if item is None:
                self.canvas.delete("region")
                self._single_region_item = self.canvas.create_rectangle(
                    *coordinates, outline="#ffce65", width=2, tags="region"
                )
            else:
                self.canvas.coords(item, *coordinates)
            return
        self._single_region_item = None
        self._corner_items = ()
        self.canvas.delete("region")
        if not boxes or size is None:
            return
        for box in boxes:
            style = (
                {"outline": "#a1a8b0", "dash": (4, 4)}
                if dashed
                else {"outline": "#ffce65"}
            )
            self.canvas.create_rectangle(
                *preview_box(box, size, self.preview_size, mirrored=mirrored),
                **style,
                width=2,
                tags="region",
            )

    def _draw_identity_overlay(self, state, view, now):
        from .identity_overlay import CanvasIdentityOverlay, identity_label, preview_box

        boxes, size, unconfirmed = self._display_regions(now)
        workspace = getattr(self, "_local_workspace", False)
        label = identity_label(
            state,
            view,
            face_count=len(boxes),
            confirmed=not unconfirmed,
            processing=self.live.status.startswith("Processing"),
            trained=getattr(self, "_trained_recognition_enabled", False),
            unverified_preview=workspace,
        )
        if label is None or size is None or min(self.preview_size) <= 0:
            if self._identity_overlay is not None:
                self._identity_overlay.hide()
            return
        if self._identity_overlay is None:
            self._identity_overlay = CanvasIdentityOverlay(self.canvas)
        box = preview_box(
            boxes[0], size, self.preview_size, mirrored=self._preview_mirrored
        )
        self._identity_overlay.update(
            label,
            box,
            self.preview_size,
            show_unverified=not workspace,
            match_box_width=workspace,
        )

    def _render(self):
        state = self.controller.state
        workspace = getattr(self, "_local_workspace", False)
        render_time = getattr(self, "_ui_time", None)
        if render_time is None:
            render_time = monotonic()
        presentation = getattr(self, "presentation", None)
        view = (
            presentation.view(state, render_time) if presentation is not None else None
        )
        emotion_display = getattr(self, "emotion_presentation", None)
        emotion_view = (
            emotion_display.view(render_time) if emotion_display is not None else None
        )
        self._last_display_view, self._last_emotion_view = view, emotion_view
        if hasattr(self, "_identity_overlay"):
            self._draw_identity_overlay(state, view, render_time)
        inspector = getattr(self, "_diagnostics_view", None)
        trial = self.current_liveness_trial()
        liveness_preview = getattr(self, "liveness_preview", None)
        liveness_reading = (
            liveness_preview.display_text() if liveness_preview is not None else None
        )
        if getattr(self, "_dashboard", None) is not None:
            sample = (
                liveness_preview.current(self.live, render_time, self.MAX_AGE)
                if liveness_preview is not None
                else None
            )
            self._dashboard.update_preview(
                state,
                view,
                emotion_view,
                detecting=self.live.status.startswith("Processing"),
                scan=liveness_preview.last_check
                if liveness_preview is not None
                else None,
                trained_liveness=liveness_preview is not None,
                trained_emotion=self._trained_emotion_enabled,
                score_error=sample is not None and sample.frame_status == "error",
            )
        profiles = getattr(self, "_trial_profiles", None)
        overlay = getattr(self, "_registration_overlay", None)
        if overlay is not None:
            guidance = (
                profiles.guidance()
                if self._developer_mode and self._preview_item is not None
                else None
            )
            overlay.update(guidance, self.preview_size, render_time)
        can_register = self._can_start_local_registration(render_time)
        registering = workspace and profiles is not None and profiles.active
        registration_requested = getattr(self, "_registration_requested", False)
        if (
            presentation is not None
            and presentation.stable_trial
            and not (trial is not None and trial.error is not None)
        ):
            trial = presentation.trial_view(render_time)
        if inspector is not None:
            inspector.refresh(self, render_time)
        diagnostics_paused = (
            inspector is not None and getattr(inspector, "paused", False) is True
        )
        raw_requested = (
            inspector is not None
            and getattr(inspector, "raw_refresh_requested", False) is True
        )
        key = (
            self.live.status,
            state,
            view,
            emotion_view,
            trial,
            liveness_reading,
            can_register,
            registering,
            registration_requested,
        )
        debug_due = (
            getattr(self, "_debug_visible", False)
            and not diagnostics_paused
            and (
                raw_requested
                or (
                    state != getattr(self, "_last_payload", None)
                    and render_time >= getattr(self, "_next_debug_update", 0)
                )
            )
        )
        if key == self._last_render and not debug_due:
            return
        self._last_render = key
        pending = self.live.status.startswith("Processing")
        missing = (
            "No face detected"
            if state.ui_state == "no_face"
            else ("Processing" if pending else "Unavailable")
        )
        emotion = state.emotion.label if state.emotion else missing
        identity = self.controller.demo_identity
        if identity == "Unavailable":
            identity = missing
        liveness = state.liveness.status if state.liveness else missing
        trained_face = (
            state.recognition is not None and state.recognition.source == "model"
        )
        identity_kind = "simulated"
        if getattr(self, "_trained_recognition_enabled", False):
            identity_kind = "trained, UNVERIFIED"
            if trained_face and state.recognition.accepted:
                identity += f" [{state.recognition.person_id}]"
        trained = state.emotion is not None and state.emotion.source == "model"
        emotion_kind = (
            "trained CNN"
            if trained
            or (
                state.emotion is None
                and getattr(self, "_trained_emotion_enabled", False)
            )
            else "simulated"
        )
        if trained:
            emotion += f" ({state.emotion.confidence:.1%} softmax)"
        status = "Checking current frame…" if pending else self.live.status
        if view is not None:
            from .presentation import CHECKING_MATCH

            status, identity, emotion, liveness = (
                view.status,
                view.identity,
                view.emotion,
                view.liveness,
            )
            if emotion_view is not None:
                emotion = emotion_view.label
            # Four fixed-position cells, changed only when their visible text changes.
            emotion_caption = (
                ("latest sample" if emotion_view.sampled else "display only ≤800 ms")
                if emotion_view is not None
                else "display-smoothed"
            )
            fields = {
                "status": status,
                "identity": f"Identity ({identity_kind}): {identity}",
                "emotion": f"Emotion ({emotion_kind}, {emotion_caption}): {emotion}",
                "liveness": f"Liveness (SIMULATED): {liveness}",
            }
            if getattr(self, "_mode_supported", False) and not self._developer_mode:
                from .developer_view import standard_fields

                fields = standard_fields(state, view, emotion_view)
            if (
                presentation.stable_trial
                and view.source_context is not None
                and identity not in ("Unavailable", CHECKING_MATCH)
            ):
                shown_identity = "Unknown person" if identity == "Unknown" else identity
                fields["identity"] = f"Identity: {shown_identity} (unverified sample)"
            if liveness_reading is not None:
                fields["liveness"] = liveness_reading
            elif getattr(self, "_liveness_trial_worker", None) is not None:
                fields["liveness"] = self._liveness_trial_text(trial)
            elif workspace:
                fields["liveness"] = "Liveness: verified model unavailable"
            for name, text in fields.items():
                cache = "_display_" + name
                if text != getattr(self, cache, None):
                    getattr(self, "display_" + name).set(text)
                    setattr(self, cache, text)
        summary = (
            f"{status}\n\n"
            f"Identity ({identity_kind}): {identity}\n"
            f"{'Verified liveness' if workspace else 'Liveness (simulated)'}: {liveness}\n"
            f"Emotion ({emotion_kind}): {emotion}"
        )
        if summary != getattr(self, "_last_summary", None):
            self.summary.set(summary)
            self._last_summary = summary
        if getattr(self, "_trained_recognition_enabled", False):
            unknown = (
                trained_face
                and state.ui_state == "live"
                and not state.recognition.accepted
            )
            prompt = (
                (
                    "Unknown? Registration options open the form only. Genuine saving is disabled."
                )
                if view is not None
                else "Unknown — Register a new person. Genuine enrollment requires trained liveness."
                if unknown
                else "Genuine registration and attendance remain disabled."
            )
            if getattr(self, "_mode_supported", False) and not self._developer_mode:
                prompt = "Unknown person? Open registration options. Saving remains disabled."
            if workspace:
                matched_name = view.identity_name if view is not None else None
                if (
                    state.ui_state == "preview"
                    and trained_face
                    and state.recognition.accepted is True
                ):
                    matched_name = state.recognition.name
                if registering:
                    prompt = "Follow the five capture steps below the camera."
                elif liveness_preview is None:
                    prompt = (
                        "Local registration requires the AENet-Full checkpoint."
                        if self._developer_mode
                        else "Registration unavailable."
                    )
                elif not registration_requested and getattr(
                    profiles, "save_notice", None
                ):
                    prompt = profiles.save_notice
                elif matched_name:
                    prompt = f"Face matches the saved profile “{matched_name}”. Show a new person to register."
                elif registration_requested:
                    prompt = "Waiting for one unregistered face. Ask the new person to face the camera."
                elif can_register:
                    prompt = "New face detected. You can start registration."
                else:
                    prompt = "Select Register new person to start."
            elif profiles is not None:
                prompt = "Capture reference photos to create a named test profile for this session."
            if prompt != getattr(self, "_last_registration_prompt", None):
                self.registration_prompt.set(prompt)
                self._last_registration_prompt = prompt
            if workspace:
                text = (
                    "Cancel registration request"
                    if registration_requested
                    else "Register new person"
                )
                button_state = (
                    "disabled" if registering or liveness_preview is None else "normal"
                )
                button_view = (text, button_state)
                if button_view != getattr(self, "_last_registration_button", None):
                    self.registration_button.configure(text=text, state=button_state)
                    self._last_registration_button = button_view
            else:
                button_state = (
                    "normal"
                    if profiles is not None or view is not None or unknown
                    else "disabled"
                )
                if button_state != getattr(self, "_last_registration_button", None):
                    self.registration_button.configure(state=button_state)
                    self._last_registration_button = button_state
        if getattr(self, "_trained_emotion_enabled", False):
            if trained:
                result = state.emotion
                architecture = getattr(self, "_emotion_architecture", "EmotionCNN")
                text = (
                    f"Selected {architecture}: {result.model_id}\nWeights: {result.model_version} • "
                    f"epoch {result.selected_epoch}\nSHA-256: {result.checkpoint_sha256}"
                )
                if text != getattr(self, "_last_model_info", None):
                    self.model_info.set(text)
                    self._last_model_info = text
        if emotion_view is not None and getattr(self, "_debug_visible", False):
            # Separate audit text: never insert historical data into SYS-02 JSON.
            max_age_ms = round(emotion_display.display_seconds * 1000)
            text = f"Emotion display: {emotion_view.reason}; source: {emotion_view.source_context}; max age {max_age_ms} ms."
            if text != getattr(self, "_last_emotion_display_info", None):
                self.emotion_display_info.set(text)
                self._last_emotion_display_info = text
        if not getattr(self, "_hint_rendered", False):
            self.hint.set(
                "Show one face to the camera."
                if workspace
                else "Show one face to the camera. Preview results are unverified."
                if getattr(self, "_mode_supported", False)
                else "Automatic sampled detection • Dashed grey box = previous detection, unconfirmed "
                "(maximum 450 ms). Previous-sample text is display-only, at most 800 ms; no authorization."
            )
            self._hint_rendered = True
        # Clearing a current result is immediate, including the diagnostic view.
        # Repeated same-context JSON changes are capped at 4 Hz; first results
        # and security-invalidating changes bypass the cap. JSON never authorizes.
        cleared = (
            state.ui_state
            in {"no_face", "multiple_faces", "spoof", "uncertain", "error"}
            or state.context is None
            or (
                state.recognition is None
                and state.liveness is None
                and state.emotion is None
            )
        )
        previous_payload = getattr(self, "_last_payload", None)
        first_result = previous_payload is None or (
            previous_payload.recognition is None
            and previous_payload.liveness is None
            and previous_payload.emotion is None
        )
        if (
            getattr(self, "_debug_visible", True)
            and not diagnostics_paused
            and (raw_requested or state != getattr(self, "_last_payload", None))
            and (
                raw_requested
                or cleared
                or first_result
                or render_time >= getattr(self, "_next_debug_update", 0)
            )
        ):
            started = monotonic()
            if inspector is not None:
                inspector.update_raw(state)
            else:
                self.payload.configure(state="normal")
                self.payload.delete("1.0", "end")
                self.payload.insert(
                    "1.0", json.dumps(asdict(state), indent=2, allow_nan=False)
                )
                self.payload.configure(state="disabled")
            self._last_payload = state
            self._next_debug_update = render_time + 0.25
            if getattr(self, "metrics", None) is not None:
                self.metrics.duration("debug_json", monotonic() - started)

    def _can_start_local_registration(self, now):
        profiles = getattr(self, "_trial_profiles", None)
        if (
            not getattr(self, "_local_workspace", False)
            or not self._developer_mode
            or profiles is None
            or profiles.active
            or profiles.liveness_threshold is None
            or len(self.live.boxes) != 1
            or self.live.sample_time is None
            or not 0 <= now - self.live.sample_time <= profiles.session.MAX_SAMPLE_AGE
        ):
            return False
        state = self.controller.state
        recognition = state.recognition
        return (
            state.ui_state == "preview"
            and state.context is not None
            and recognition is not None
            and recognition.source == "model"
            and recognition.accepted is False
        )

    def _show_registration(self):
        profiles = getattr(self, "_trial_profiles", None)
        if profiles is not None:
            if not self._local_workspace:
                profiles.show()
            elif (
                not self._developer_mode
                or profiles.active
                or profiles.liveness_threshold is None
            ):
                return
            elif getattr(self, "_registration_requested", False):
                self._registration_requested = False
            elif self._can_start_local_registration(monotonic()):
                profiles.show()
            else:
                self._registration_requested = True
            attendance_verification = getattr(self, "_attendance_verification", None)
            if attendance_verification is not None:
                attendance_verification.invalidate()
            self._render()
            return
        state = self.controller.state
        presentation = getattr(self, "presentation", None)
        if not getattr(self, "_trained_recognition_enabled", False):
            return
        if presentation is None and (
            state.ui_state != "live"
            or state.recognition is None
            or state.recognition.source != "model"
            or state.recognition.accepted
        ):
            return
        if presentation is not None:
            presentation.reset()
        emotion_display = getattr(self, "emotion_presentation", None)
        if emotion_display is not None:
            emotion_display.reset("action_change")
        self._presentation_request = None
        self.registration_form.grid()
        self.live.reset()
        self.controller.reset("registration")
        self._last_render = None
        self._render()

    def _trial_gallery_changed(self):
        # Gallery edits invalidate in-flight recognition without restarting the camera.
        self._registration_requested = False
        attendance_verification = getattr(self, "_attendance_verification", None)
        if attendance_verification is not None:
            attendance_verification.invalidate()
        self.live.reset()
        self.presentation.reset()
        liveness_preview = getattr(self, "liveness_preview", None)
        if liveness_preview is not None:
            liveness_preview.clear()
        self._presentation_request = None
        self._next_detection = 0
        self._last_render = None
        self._render()

    def _clear_presentation(
        self, timestamp, reason="invalidated", *, affect_emotion=True
    ):
        presentation = getattr(self, "presentation", None)
        if presentation is not None:
            presentation.invalidate(timestamp)
        emotion_display = getattr(self, "emotion_presentation", None)
        if affect_emotion and emotion_display is not None:
            emotion_display.clear(reason)

    def _start_test_enrollment(self):
        if (
            not getattr(self, "_controlled_registration", False)
            or self._registration_worker.busy
        ):
            return
        state = self.controller.state
        if (
            state.context is None
            or state.ui_state != "live"
            or state.recognition is None
            or state.context.action != "registration"
            or state.recognition.accepted
            or state.recognition.source != "model"
        ):
            self.registration_message.set(
                "Open Register a new person and wait for its current Unknown test result."
            )
            return
        token = self._generation, self.controller.attempt_id
        if self._registration_worker.submit(
            token, (self.registration_id.get(), self.registration_name.get())
        ):
            self._enrollment_token = token
            self.registration_message.set(
                "Processing five generated test samples… Genuine saving remains disabled."
            )

    def _poll_test_enrollment(self):
        worker = getattr(self, "_registration_worker", None)
        if worker is None:
            return
        if self.controller.state.ui_state in {
            "spoof",
            "uncertain",
            "error",
            "no_face",
            "multiple_faces",
        }:
            self._enrollment_token = None
        result = worker.take()
        if result is None:
            return
        token = self._generation, self.controller.attempt_id
        if result.token != self._enrollment_token or result.token != token:
            self.registration_message.set(
                "Test registration cancelled by reset or scene change; retry."
            )
            return
        self._enrollment_token = None
        if result.error is not None:
            self.registration_message.set(f"Test registration failed: {result.error}")
            return
        try:
            person = self.adapters.matcher.commit(result.value)
        except (ValueError, PermissionError) as error:
            self.registration_message.set(f"Test registration failed: {error}")
            return
        self.retry()  # Invalidate in-flight old-gallery matches before showing the new one.
        self.registration_message.set(
            f"Test identity {person.name} [{person.person_id}] added to memory only. "
            "Checking subsequent recognition. No genuine enrollment."
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        attendance_verification = getattr(self, "_attendance_verification", None)
        if attendance_verification is not None:
            attendance_verification.service.end_session()
            attendance_verification.sync_session()
        presentation = getattr(self, "presentation", None)
        if presentation is not None:
            presentation.reset()
        emotion_display = getattr(self, "emotion_presentation", None)
        if emotion_display is not None:
            emotion_display.reset("shutdown")
        self._presentation_request = None
        self.root.after_cancel(self._timer)
        self.live.reset()
        self.feed.close()
        self.detector.close()
        self.inference.close()
        trial_worker = getattr(self, "_liveness_trial_worker", None)
        if trial_worker is not None:
            trial_worker.close()
        profiles = getattr(self, "_trial_profiles", None)
        if profiles is not None:
            profiles.close()
        worker = getattr(self, "_registration_worker", None)
        if worker is not None:
            worker.close()
        if getattr(self, "metrics", None) is not None:
            self.metrics.report(force=True)
        inspector = getattr(self, "_diagnostics_view", None)
        if inspector is not None:
            inspector.window.destroy()
        self._completed_diagnostic = None
        self.root.destroy()


class DemoCamera:
    """Generated source for exercising the same GUI path without camera hardware."""

    def __init__(self):
        import cv2
        import numpy as np

        self.frame = np.full((360, 640, 3), (45, 33, 20), dtype=np.uint8)
        cv2.putText(
            self.frame,
            "Synthetic camera - no real face",
            (100, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 235, 235),
            1,
        )
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        return self.opened, self.frame.copy()

    def release(self):
        self.opened = False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run webcam recognition, liveness, emotion and local attendance demonstration."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--camera", type=int, default=0, help="Webcam device index (default 0)."
    )
    source.add_argument(
        "--demo-camera",
        action="store_true",
        help="Use a generated frame instead of a webcam.",
    )
    parser.add_argument(
        "--dev-scenario",
        choices=SCENARIOS,
        default=SCENARIOS[0],
        help="Development only: deterministic simulated adapter scenario.",
    )
    parser.add_argument(
        "--dev-manual",
        action="store_true",
        help="Development only: freeze and manually select source regions.",
    )
    parser.add_argument(
        "--dev-action",
        choices=("attendance", "registration"),
        default="attendance",
        help="Development manual harness only: non-persistent operation.",
    )
    parser.add_argument(
        "--detector",
        choices=("yunet", "haar"),
        default="yunet",
        help="Automatic detector (default: CPU YuNet; local model required).",
    )
    parser.add_argument(
        "--yunet-model",
        default=str(YUNET_MODEL),
        help="Local checksum-pinned YuNet ONNX file.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print aggregate performance counters every 5 s.",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="Native OpenCV CPU threads (default 1, avoid oversubscription).",
    )
    parser.add_argument(
        "--emotion-checkpoint",
        help="Local trained Emotion checkpoint (ResNet-18 or legacy EmotionCNN).",
    )
    parser.add_argument(
        "--face-checkpoint", help="Selected trained Face recognition checkpoint."
    )
    liveness_options = parser.add_mutually_exclusive_group()
    liveness_options.add_argument(
        "--liveness-checkpoint",
        metavar="PATH",
        help="AENet-Full epoch 18 checkpoint override for the local workspace. "
        "Defaults to the bundled artifacts/models/liveness/liveness-inference.pt.",
    )
    liveness_options.add_argument(
        "--liveness-trial-checkpoint",
        action="append",
        default=[],
        metavar="PATH",
        help="Experimental read-only liveness scoring. Pass once per supplied checkpoint.",
    )
    parser.add_argument(
        "--local-workspace",
        action="store_true",
        help="Open the trained webcam workspace with consented local profile registration.",
    )
    parser.add_argument(
        "--local-profile-db",
        type=Path,
        metavar="PATH",
        help="Select a separate SQLite profile database in the local workspace.",
    )
    parser.add_argument(
        "--attendance",
        action="store_true",
        help="Show attendance controls. Recording requires isolated attendance test mode.",
    )
    parser.add_argument(
        "--attendance-db",
        type=Path,
        metavar="PATH",
        help="Override the separate local SQLite attendance database.",
    )
    parser.add_argument(
        "--attendance-verification-test",
        action="store_true",
        help="Enable candidate verification only with explicitly isolated test databases.",
    )
    parser.add_argument(
        "--attendance-auto-retry-test",
        action="store_true",
        help="In isolated attendance verification, retry once after spoof using fresh samples.",
    )
    parser.add_argument(
        "--attendance-auto-record-test",
        action="store_true",
        help="Automatically record a stable three-second live match in isolated attendance testing.",
    )
    parser.add_argument(
        "--gallery",
        help="Read-only compatible schema 0.2 gallery. Never written by the webcam UI.",
    )
    parser.add_argument(
        "--controlled-registration-demo",
        action="store_true",
        help="Generated-input test enrollment into isolated memory; requires --demo-camera and --face-checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.dev_action != "attendance" and not args.dev_manual:
        parser.error(
            "--dev-action registration requires the separate --dev-manual harness"
        )
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    if args.emotion_checkpoint and args.dev_manual:
        parser.error(
            "--emotion-checkpoint is supported only in the automatic live workflow"
        )
    if args.face_checkpoint and args.dev_manual:
        parser.error("--face-checkpoint requires the automatic live workflow")
    if args.gallery and not args.face_checkpoint:
        parser.error("--gallery requires --face-checkpoint")
    if args.controlled_registration_demo and (
        not args.demo_camera or not args.face_checkpoint or args.gallery
    ):
        parser.error(
            "--controlled-registration-demo requires --demo-camera, --face-checkpoint and no external --gallery"
        )
    if args.liveness_trial_checkpoint and (
        args.dev_manual
        or args.controlled_registration_demo
        or not args.face_checkpoint
        or not args.emotion_checkpoint
    ):
        parser.error(
            "The three-model trial requires Face and Emotion checkpoints in automatic preview mode."
        )
    if args.local_workspace and (
        args.dev_manual
        or args.demo_camera
        or args.gallery
        or not args.face_checkpoint
        or not args.emotion_checkpoint
    ):
        parser.error(
            "--local-workspace requires a webcam and Face and Emotion checkpoints, with its own profile database."
        )
    if args.liveness_checkpoint and (
        not args.local_workspace or args.detector != "yunet"
    ):
        parser.error(
            "--liveness-checkpoint requires --local-workspace and the release's YuNet detector."
        )
    if (
        args.attendance or args.local_profile_db is not None
    ) and not args.local_workspace:
        parser.error("--attendance and --local-profile-db require --local-workspace.")
    if args.attendance_db is not None and not args.attendance:
        parser.error("--attendance-db requires --attendance.")
    if args.attendance_auto_retry_test and not args.attendance_verification_test:
        parser.error(
            "--attendance-auto-retry-test requires --attendance-verification-test."
        )
    if args.attendance_auto_record_test and not args.attendance_verification_test:
        parser.error(
            "--attendance-auto-record-test requires --attendance-verification-test."
        )
    if args.attendance_verification_test:
        if (
            not args.attendance
            or args.local_profile_db is None
            or args.attendance_db is None
        ):
            parser.error(
                "Candidate verification requires --attendance, --local-profile-db and --attendance-db."
            )
        test_root = LOCAL_ARTIFACTS.resolve() / "attendance-verification-test"
        try:
            if test_root.is_symlink() or any(
                not path.resolve().is_relative_to(test_root)
                or path.resolve() == test_root
                or (path.exists() and path.stat().st_nlink != 1)
                for path in (args.local_profile_db, args.attendance_db)
            ):
                parser.error(
                    "Both candidate databases must be separate files under artifacts/attendance-verification-test."
                )
        except (OSError, ValueError) as error:
            parser.error(f"Test database path is invalid: {error}")
        if args.detector != "yunet" or args.liveness_trial_checkpoint:
            parser.error(
                "Candidate verification requires the released AENet-Full checkpoint and YuNet."
            )
    liveness_checkpoint = args.liveness_checkpoint
    if (
        args.local_workspace
        and args.detector == "yunet"
        and not args.liveness_trial_checkpoint
        and liveness_checkpoint is None
    ):
        if not LOCAL_LIVENESS_CHECKPOINT.is_file():
            parser.error(
                "The bundled liveness checkpoint is missing; supply --liveness-checkpoint PATH."
            )
        liveness_checkpoint = LOCAL_LIVENESS_CHECKPOINT

    if args.attendance_verification_test and liveness_checkpoint is None:
        parser.error(
            "Candidate verification requires the released AENet-Full checkpoint."
        )

    # Construct/load trained adapters once before creating Tk, never in its event loop.
    adapters = None
    if args.face_checkpoint:
        try:
            from .recognition_adapter import TrainedPreviewAdapters

            adapters = TrainedPreviewAdapters(
                args.face_checkpoint,
                emotion_checkpoint=args.emotion_checkpoint,
                gallery_path=args.gallery,
                controlled_registration=args.controlled_registration_demo,
                scenario=args.dev_scenario,
                crop_to_detector_box=args.local_workspace,
            )
        except Exception as error:
            parser.exit(1, f"Trained integration unavailable: {error}\n")
    elif args.emotion_checkpoint:
        from .emotion_adapter import MixedEmotionAdapters

        adapters = MixedEmotionAdapters(args.dev_scenario, args.emotion_checkpoint)
    trial = None
    if args.liveness_trial_checkpoint:
        try:
            from ..liveness.checkpoint_trial import LivenessCheckpointTrial

            trial = LivenessCheckpointTrial(args.liveness_trial_checkpoint)
        except (OSError, RuntimeError, ValueError, ImportError, KeyError) as error:
            parser.exit(1, f"Liveness trial unavailable: {error}\n")
    profile_store = None
    attendance = None
    if args.local_workspace:
        from .local_profile_store import (
            LOCAL_BOX_PROFILE_DB,
            LOCAL_PROFILE_DB,
            LocalProfileStore,
        )

        profile_store = LocalProfileStore(
            args.local_profile_db or LOCAL_BOX_PROFILE_DB, crop_to_detector_box=True
        )
        if args.attendance:
            from ..attendance.service import AttendanceService
            from ..attendance.store import AttendanceStore
            from ..config import load_paths

            try:
                workspace_root = LOCAL_ARTIFACTS.parent
                attendance_path = args.attendance_db or (
                    load_paths(
                        workspace_root / "configs/project.toml", root=workspace_root
                    ).attendance
                    / "attendance.sqlite3"
                )
                store = AttendanceStore(attendance_path)
                for profile_path in (
                    profile_store.path,
                    LOCAL_PROFILE_DB.resolve(),
                    LOCAL_BOX_PROFILE_DB.resolve(),
                ):
                    if store.path == profile_path or (
                        store.path.exists()
                        and profile_path.exists()
                        and store.path.samefile(profile_path)
                    ):
                        parser.error(
                            "The attendance database must be separate from the profile database."
                        )
            except (OSError, ValueError) as error:
                parser.error(f"Attendance storage configuration is invalid: {error}")
            attendance = AttendanceService(store)

    try:
        import tkinter as tk

        import cv2
        from PIL import ImageTk  # noqa: F401 — fail before camera startup if Tk bindings are missing.
    except ImportError as error:
        parser.exit(1, f"Webcam demo dependencies unavailable: {error}\n")
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(args.opencv_threads)
    released_liveness = None
    if liveness_checkpoint is not None:
        from .liveness_preview import ReleasedLiveness

        try:
            released_liveness = ReleasedLiveness(liveness_checkpoint, args.yunet_model)
        except (
            OSError,
            RuntimeError,
            ValueError,
            ImportError,
            KeyError,
            cv2.error,
        ) as error:
            parser.exit(1, f"AENet-Full unavailable: {error}\n")
    attendance_verification = None
    if args.attendance_verification_test:
        from ..attendance.verification import AttendanceVerification

        def registered_name(person_id):
            matcher = adapters.matcher
            with matcher._lock:
                entry = matcher._gallery.get(person_id)
                return entry.name if entry is not None else None

        try:
            attendance_verification = AttendanceVerification(
                store,
                released_liveness.provenance,
                adapters.face_inference.provenance,
                registered_name,
                max_age_seconds=LiveWebcamApp.MAX_AGE,
                automatic_retry=args.attendance_auto_retry_test,
                automatic_record=args.attendance_auto_record_test,
            )
        except (ValueError, KeyError) as error:
            parser.error(f"Candidate attendance verification is unavailable: {error}")
        attendance = attendance_verification.service
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(1, f"Cannot open the webcam demo window: {error}\n")

    def open_camera():
        camera = cv2.VideoCapture(args.camera)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except (OSError, cv2.error):
                pass  # Backends may ignore/reject this best-effort latency hint.
        return camera

    factory = DemoCamera if args.demo_camera else open_camera
    source_name = (
        "Synthetic camera (no real face)"
        if args.demo_camera
        else f"Webcam {args.camera}"
    )
    if args.dev_manual:
        app = WebcamApp(
            root,
            factory,
            source_name=source_name,
            scenario=args.dev_scenario,
            action=args.dev_action,
        )
    else:
        if released_liveness is not None:
            detector = released_liveness.detect
        else:
            locator = (
                YuNetFaceDetector(args.yunet_model)
                if args.detector == "yunet"
                else HaarFaceDetector()
            )
            detector = ScaledFaceDetector(locator)
        if args.controlled_registration_demo:
            from ..recognition.gallery_matching import CONTROLLED_BOX
            from .recognition_adapter import ControlledCamera

            factory = ControlledCamera

            def detector(frame):
                return [CONTROLLED_BOX]

            source_name = (
                "GENERATED TEST INPUT • controlled regions, no physical camera"
            )
        detector_name = (
            "Controlled regions (test only)"
            if args.controlled_registration_demo
            else f"{args.detector} (CPU)"
        )
        if released_liveness is not None:
            detector_name = "YuNet (AENet-Full release, CPU)"
        try:
            app = LiveWebcamApp(
                root,
                factory,
                source_name=f"{source_name} • {detector_name}",
                scenario=args.dev_scenario,
                detector=detector,
                diagnostics=args.diagnostics,
                adapters=adapters,
                liveness_trial=trial,
                released_liveness=released_liveness,
                enable_trial_profiles=trial is not None and args.gallery is None,
                local_profile_store=profile_store,
                attendance=attendance,
                attendance_verification=attendance_verification,
                runtime_info={
                    "camera_index": None if args.demo_camera else args.camera,
                    "face_detector": detector_name,
                    "processing_device": "CPU",
                    "input": "generated" if args.demo_camera else "webcam",
                },
            )
        except ValueError as error:
            root.destroy()
            parser.exit(1, f"Cannot open workspace: {error}\n")
    root.after_idle(app.start)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
