"""Automatic Tk webcam demo plus an explicit developer selection harness."""

import argparse
from contextlib import suppress
from dataclasses import asdict
import json
from time import monotonic

from .diagnostics import Diagnostics
from .face_detection import (
    FaceDetectorUnavailable, HaarFaceDetector, ScaledFaceDetector, YuNetFaceDetector, YUNET_MODEL,
)
from .live_preview import CameraFeed, LiveController, SingleWorker, evaluate_adapters, prepare_preview_image
from .webcam_prototype import (
    SCENARIOS,
    PrototypeController,
    StubAdapters,
    frame_size,
    run_stubs,
)

AUTOMATIC = "Automatic detection"
MANUAL = "Manual selection"


class WebcamApp:
    """Explicit developer freeze/selection harness; retains the verified legacy API."""

    def __init__(self, root, open_camera, *, source_name: str,
                 scenario=SCENARIOS[0], action="attendance"):
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

        ttk.Label(panel, text="Facial Attendance", font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(panel, text=f"DEMO / SIMULATED OUTPUT • Developer manual harness • {source_name}").grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        self.canvas = tk.Canvas(panel, width=640, height=360, background="#162332", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nw", pady=(10, 6))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._draw_drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)

        controls = ttk.Frame(panel)
        controls.grid(row=2, column=0, sticky="w")
        self.freeze_button = ttk.Button(controls, text="Freeze frame", command=self.freeze)
        self.freeze_button.pack(side="left")
        self.run_button = ttk.Button(controls, text="Run stubs", command=self.evaluate)
        self.run_button.pack(side="left", padx=6)
        self.clear_button = ttk.Button(controls, text="Clear regions", command=self._clear_regions)
        self.clear_button.pack(side="left")
        ttk.Button(controls, text="Retry / resume preview", command=self.retry).pack(side="left", padx=6)
        self.hint = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.hint, wraplength=630).grid(row=3, column=0, sticky="w", pady=(6, 0))

        results = ttk.Frame(panel)
        results.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=(16, 0), pady=(10, 0))
        self.action = tk.StringVar(value=action)
        self.scenario = tk.StringVar(value=scenario)
        self.selection_mode = tk.StringVar(value=MANUAL)
        ttk.Label(results, text=f"Developer scenario: {scenario}\nOperation: {action} (non-persistent)",
                  wraplength=340).pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(results, textvariable=self.summary, wraplength=340, justify="left").pack(
            anchor="w", pady=8
        )
        ttk.Label(results, text="SYS-02 result payload").pack(anchor="w")
        self.payload = ScrolledText(results, width=45, height=14, wrap="word", state="disabled")
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
                    self._camera_error("Camera read failed. Reconnect the camera and retry.")
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
        from PIL import ImageTk

        image = prepare_preview_image(frame)
        self.preview_size = image.size
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        if getattr(self, "_preview_item", None) is None:
            self._preview_item = self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="preview")
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
            self.controller.capture_error("model_unavailable", str(error), module="detector")
        except ValueError as error:
            self.controller.capture_error("invalid_input", str(error), module="detector")
        except cv2.error:
            self.controller.capture_error(
                "inference_error", "OpenCV could not detect face regions.", module="detector"
            )
        else:
            self.boxes.extend(boxes)
            width, height = frame_size(self.frozen)
            view_width, view_height = self.preview_size
            for x1, y1, x2, y2 in boxes:
                self.canvas.create_rectangle(
                    x1 * view_width / width, y1 * view_height / height,
                    x2 * view_width / width, y2 * view_height / height,
                    outline="#ffce65", width=2, tags="region",
                )
            self.hint.set(
                f"{len(boxes)} face region(s) detected. Run stubs or switch to Manual selection."
            )
            return
        self.hint.set("Automatic detection unavailable for this frame. Switch to Manual selection.")

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
            left * width // view_width, top * height // view_height,
            (right * width + view_width - 1) // view_width,
            (bottom * height + view_height - 1) // view_height,
        )
        self.boxes.append(box)
        self.canvas.create_rectangle(left, top, right, bottom, outline="#5df6cb", width=2, tags="region")
        self.hint.set(f"{len(self.boxes)} region(s) selected. Run stubs or clear the selection.")

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
            self.hint.set("No regions selected. Resume preview and freeze again to detect faces.")

    def _clear_regions(self):
        if self.frozen is None:
            return
        if self.controller.state.ui_state in {"spoof", "uncertain"}:
            self._resume_preview()
            return
        self._reset_selection()
        self._render()

    def evaluate(self):
        if self.frozen is None or self.evaluated or self.controller.state.ui_state == "error":
            return
        self.evaluated = True
        request = self.controller.begin_capture(self.frozen, self.boxes)
        if request is not None:
            height, width, _ = request.face_crop.shape
            self.hint.set(f"Source crop {width} × {height} pixels. Resume preview for a fresh frame.")
            run_stubs(self.controller, request, StubAdapters(self.scenario.get()))
        else:
            self.hint.set("Resume preview for a fresh frame or adjust the region selection.")
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
        status = messages.get(state.ui_state, state.errors[0].message if state.errors else state.ui_state)
        emotion = state.emotion.label if state.emotion else "Unavailable"
        self.summary.set(
            f"{status}\nDemo identity: {self.controller.demo_identity}\n"
            f"Demo emotion: {emotion}\nSimulated values, not inferred from image content.\nSaving disabled in demo"
        )
        self.payload.configure(state="normal")
        self.payload.delete("1.0", "end")
        self.payload.insert("1.0", json.dumps(asdict(state), indent=2, allow_nan=False))
        self.payload.configure(state="disabled")
        self.freeze_button.configure(state="normal" if self.latest is not None and self.frozen is None else "disabled")
        can_run = self.frozen is not None and not self.evaluated and state.ui_state != "error"
        self.run_button.configure(state="normal" if can_run else "disabled")
        self.clear_button.configure(state="normal" if self.frozen is not None else "disabled")
        if self.frozen is None:
            self.hint.set("Freeze a preview frame to select a face region." if self.latest is not None else status)

    def close(self):
        self.root.after_cancel(self._timer)
        try:
            self._release()
        finally:
            self.root.destroy()


class LiveWebcamApp:
    """Default automatic attendance preview. All methods execute on the Tk thread."""

    ANALYSIS_INTERVAL = 0.3
    MAX_AGE = 1.0

    def __init__(self, root, open_camera, *, source_name, scenario=SCENARIOS[0],
                 detector=None, adapters=None, diagnostics=False):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.live = LiveController()
        self.controller = self.live.controller
        self.adapters = adapters if adapters is not None else StubAdapters(scenario)
        self.metrics = Diagnostics(diagnostics)
        self.feed = CameraFeed(open_camera, self.metrics)
        self.detector = SingleWorker(detector if detector is not None else ScaledFaceDetector(YuNetFaceDetector()),
                                     "detector", self.metrics)
        self.inference = SingleWorker(evaluate_adapters, "adapters", self.metrics)
        self._closed = False
        self._generation = 0
        self._drawn_sequence = None
        self._next_detection = 0
        self._detection_frame = None
        self._detector_failed = False
        self._opened_at = monotonic()
        self.photo = None
        self.preview_size = (0, 0)
        self._last_render = None
        self._last_summary = None
        self._last_payload = None
        self._region_key = None
        self._preview_item = None
        self._sampled_sequence = 0

        root.title("Facial Attendance | Automatic SYS-05 preview")
        root.protocol("WM_DELETE_WINDOW", self.close)
        panel = ttk.Frame(root, padding=16)
        panel.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="Facial Attendance", font=("TkDefaultFont", 20, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(panel, text="DEMO / SIMULATED OUTPUT", foreground="#a04400",
                  font=("TkDefaultFont", 13, "bold")).grid(row=0, column=1, sticky="w", padx=18)
        ttk.Label(panel, text=f"Attendance preview • {source_name}").grid(
            row=1, column=0, sticky="w", pady=(4, 10))
        self.canvas = tk.Canvas(panel, width=640, height=360, background="#162332",
                                highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nw")
        self.hint = tk.StringVar(value="Opening camera…")
        ttk.Label(panel, textvariable=self.hint, wraplength=630).grid(
            row=3, column=0, sticky="w", pady=10)
        ttk.Button(panel, text="Retry / restart camera", command=self.retry).grid(
            row=4, column=0, sticky="w")
        results = ttk.Frame(panel)
        results.grid(row=2, column=1, rowspan=3, sticky="nsew", padx=(18, 0))
        ttk.Label(results, text="Live results", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(results, textvariable=self.summary, wraplength=330, justify="left",
                  font=("TkDefaultFont", 12)).pack(anchor="w", pady=12)
        ttk.Label(results, text="Face detection uses the camera image. Identity, liveness and "
                  "emotion are deterministic simulated values, not inferred from image content.",
                  wraplength=330, justify="left").pack(anchor="w")
        ttk.Label(results, text="Saving disabled • No attendance or enrollment records",
                  wraplength=330).pack(anchor="w", pady=10)
        self.debug = ttk.Frame(results)
        ttk.Button(results, text="Show / hide SYS-02 debug JSON", command=self._toggle_debug).pack(
            anchor="w", pady=(4, 8))
        self.payload = ScrolledText(self.debug, width=43, height=13, wrap="word", state="disabled")
        self.payload.pack(fill="both", expand=True)
        self._debug_visible = False
        self._render()
        self._timer = root.after(0, self._tick)

    # Preview conversion is shared with the developer harness, not its controls.
    _draw_frame = WebcamApp._draw_frame

    def _toggle_debug(self):
        self._debug_visible = not self._debug_visible
        if self._debug_visible:
            self.debug.pack(fill="both", expand=True)
        else:
            self.debug.pack_forget()
        self._last_render = None
        self._render()

    def start(self):
        self.retry()

    def retry(self):
        if self._closed:
            return
        self.live.reset()
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
        self._render()

    def _tick(self):
        if self._closed:
            return
        now = monotonic()
        packet = self.feed.latest()
        usable = (packet is not None and packet.generation == self._generation
                  and packet.error is None and now - packet.captured_at <= self.MAX_AGE)
        if usable:
            if packet.sequence != self._drawn_sequence:
                started = monotonic()
                self._draw_frame(packet.frame)
                if getattr(self, "metrics", None) is not None:
                    self.metrics.duration("render", monotonic() - started)
                    self.metrics.duration("preview_age", monotonic() - packet.captured_at)
                    self.metrics.count("previewed")
                    self.metrics.count("preview_dropped", max(0, packet.sequence - (self._drawn_sequence or 0) - 1))
                self._drawn_sequence = packet.sequence
        elif packet is not None or now - self._opened_at > 5:
            self.live.fail("camera_unavailable", "Camera unavailable — check access and retry.")
            self.canvas.delete("all")
            self.photo = None
            self._preview_item = None
            self._region_key = None

        # Rendering itself can take time. Recheck age at acceptance time,
        # rather than trusting the timestamp from the start of the Tk tick.
        accept_time = monotonic()
        if usable and accept_time - packet.captured_at > self.MAX_AGE:
            usable = False
            self.live.fail("camera_unavailable", "Camera frame expired — waiting for a fresh frame.")
            self.canvas.delete("all")
            self.photo = self._preview_item = self._region_key = None
        # Invalidate expired context BEFORE consuming any worker completions.
        expired = self.live.expire(accept_time, self.MAX_AGE)
        metrics = getattr(self, "metrics", None)
        if metrics is not None and expired:
            metrics.count("analysis_expired")
        # Free a finished adapter slot before dispatching the next eligible
        # detection. The original token/context still governs acceptance.
        completed = self.inference.take()
        if completed is not None and usable:
            applied = self.live.completed(completed)
            if metrics is not None and not applied:
                metrics.count("adapter_discarded")
        elif completed is not None and metrics is not None:
            metrics.count("adapter_discarded")
        detected = self.detector.take()
        if detected is not None:
            if metrics is not None and (not usable or detected.token != self.live.token):
                metrics.count("detection_discarded")
            if usable:
                request = self.live.detected(detected, self._detection_frame)
                self._detector_failed = bool(
                    self.controller.state.errors
                    and self.controller.state.errors[0].module == "detector"
                )
                if request is not None:
                    submitted = self.inference.submit(detected.token, (request, self.adapters))
                    if metrics is not None and not submitted:
                        metrics.count("adapter_busy_dropped")
            self._detection_frame = None
        # Sample the newest camera frame only when the single detector slot is
        # free. Starting a sample invalidates every previous capture/context.
        if (usable and not self._detector_failed and not self.detector.busy
                and now >= self._next_detection
                and (packet.generation, packet.sequence) != self.live.token):
            token = packet.generation, packet.sequence
            if self.detector.submit(token, packet.frame):
                if metrics is not None:
                    previous = getattr(self, "_sampled_sequence", 0)
                    metrics.count("analysis_unsampled", max(0, packet.sequence - previous - 1))
                    metrics.count("analysis_started")
                    if self.live.token is not None:
                        metrics.count("analysis_superseded")
                self._sampled_sequence = packet.sequence
                self._detection_frame = packet.frame
                self.live.begin_sample(token, packet.captured_at)
                self._next_detection = now + self.ANALYSIS_INTERVAL
        self._draw_regions()
        self._render()
        if metrics is not None:
            metrics.duration("gui_tick", monotonic() - now)
            metrics.report()
        # Maintain a frame-period budget rather than adding 33 ms after work.
        delay = max(1, round((1 / 30 - (monotonic() - now)) * 1000))
        self._timer = self.root.after(delay, self._tick)

    def _draw_regions(self):
        boxes, size, unconfirmed = self.live.visual_regions(monotonic())
        key = (tuple(boxes), size, self.preview_size, unconfirmed)
        if key == getattr(self, "_region_key", None):
            return
        self._region_key = key
        self.canvas.delete("region")
        if not boxes or size is None:
            return
        width, height = size
        view_width, view_height = self.preview_size
        for x1, y1, x2, y2 in boxes:
            style = {"outline": "#a1a8b0", "dash": (4, 4)} if unconfirmed else {"outline": "#ffce65"}
            self.canvas.create_rectangle(
                x1 * view_width / width, y1 * view_height / height,
                x2 * view_width / width, y2 * view_height / height,
                **style, width=2, tags="region",
            )

    def _render(self):
        state = self.controller.state
        key = (self.live.status, state)
        if key == self._last_render:
            return
        self._last_render = key
        pending = self.live.status.startswith("Processing")
        missing = "No face detected" if state.ui_state == "no_face" else (
            "Processing" if pending else "Unavailable")
        emotion = state.emotion.label if state.emotion else missing
        identity = self.controller.demo_identity
        if identity == "Unavailable":
            identity = missing
        liveness = state.liveness.status if state.liveness else missing
        summary = (
            f"{self.live.status}\n\n"
            f"Identity (simulated): {identity}\n"
            f"Liveness (simulated): {liveness}\n"
            f"Emotion (simulated): {emotion}"
        )
        if summary != getattr(self, "_last_summary", None):
            self.summary.set(summary)
            self._last_summary = summary
        if not getattr(self, "_hint_rendered", False):
            self.hint.set("Automatic sampled detection • Dashed grey box = previous detection, unconfirmed "
                          "(maximum 450 ms). Simulated results clear on every fresh analysis and face loss.")
            self._hint_rendered = True
        if getattr(self, "_debug_visible", True) and state != getattr(self, "_last_payload", None):
            started = monotonic()
            self.payload.configure(state="normal")
            self.payload.delete("1.0", "end")
            self.payload.insert("1.0", json.dumps(asdict(state), indent=2, allow_nan=False))
            self.payload.configure(state="disabled")
            self._last_payload = state
            if getattr(self, "metrics", None) is not None:
                self.metrics.duration("debug_json", monotonic() - started)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.root.after_cancel(self._timer)
        self.live.reset()
        self.feed.close()
        self.detector.close()
        self.inference.close()
        if getattr(self, "metrics", None) is not None:
            self.metrics.report(force=True)
        self.root.destroy()


class DemoCamera:
    """Generated source for exercising the same GUI path without camera hardware."""

    def __init__(self):
        import cv2
        import numpy as np

        self.frame = np.full((360, 640, 3), (45, 33, 20), dtype=np.uint8)
        cv2.putText(self.frame, "Synthetic camera - no real face", (100, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 235, 235), 1)
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        return self.opened, self.frame.copy()

    def release(self):
        self.opened = False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the SYS-05 webcam-to-GUI stub demo.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0).")
    source.add_argument("--demo-camera", action="store_true", help="Use a generated frame instead of a webcam.")
    parser.add_argument("--dev-scenario", choices=SCENARIOS, default=SCENARIOS[0],
                        help="Development only: deterministic simulated adapter scenario.")
    parser.add_argument("--dev-manual", action="store_true",
                        help="Development only: freeze and manually select source regions.")
    parser.add_argument("--dev-action", choices=("attendance", "registration"), default="attendance",
                        help="Development manual harness only: non-persistent operation.")
    parser.add_argument("--detector", choices=("yunet", "haar"), default="yunet",
                        help="Automatic detector (default: CPU YuNet; local model required).")
    parser.add_argument("--yunet-model", default=str(YUNET_MODEL), help="Local checksum-pinned YuNet ONNX file.")
    parser.add_argument("--diagnostics", action="store_true", help="Print aggregate performance counters every 5 s.")
    parser.add_argument("--opencv-threads", type=int, default=1,
                        help="Native OpenCV CPU threads (default 1, avoid oversubscription).")
    args = parser.parse_args(argv)
    if args.dev_action != "attendance" and not args.dev_manual:
        parser.error("--dev-action registration requires the separate --dev-manual harness")
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")

    try:
        import cv2
        import tkinter as tk
        from PIL import ImageTk  # Ensure the GUI dependency is available at startup.
    except ImportError as error:
        parser.exit(1, f"Webcam demo dependencies unavailable: {error}\n")
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(args.opencv_threads)
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
    source_name = "Synthetic camera (no real face)" if args.demo_camera else f"Webcam {args.camera}"
    if args.dev_manual:
        app = WebcamApp(root, factory, source_name=source_name,
                        scenario=args.dev_scenario, action=args.dev_action)
    else:
        detector = YuNetFaceDetector(args.yunet_model) if args.detector == "yunet" else HaarFaceDetector()
        app = LiveWebcamApp(root, factory, source_name=f"{source_name} • {args.detector} (CPU)",
                            scenario=args.dev_scenario, detector=ScaledFaceDetector(detector),
                            diagnostics=args.diagnostics)
    root.after_idle(app.start)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
