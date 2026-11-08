"""Runnable Tk webcam demo for SYS-05.1 and SYS-05.2."""

import argparse
from dataclasses import asdict
import json

from .webcam_prototype import (
    SCENARIOS,
    PrototypeController,
    StubAdapters,
    frame_size,
    run_stubs,
)


class WebcamApp:
    """Preview frames and send one operator-selected region to the stub adapters."""

    def __init__(self, root, open_camera, *, source_name: str):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.open_camera = open_camera
        self.controller = PrototypeController()
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
        ttk.Label(panel, text=f"DEMO  •  {source_name}  •  Stub results").grid(
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
        settings = ttk.Frame(results)
        settings.pack(fill="x")
        self.action = tk.StringVar(value="attendance")
        self.scenario = tk.StringVar(value=SCENARIOS[0])
        action_choice = ttk.Combobox(
            settings, textvariable=self.action, values=("attendance", "registration"),
            state="readonly", width=13,
        )
        action_choice.pack(side="left")
        action_choice.bind("<<ComboboxSelected>>", self._change_action)
        ttk.Combobox(
            settings, textvariable=self.scenario, values=SCENARIOS, state="readonly", width=15
        ).pack(side="left", padx=(6, 0))
        self.summary = tk.StringVar()
        ttk.Label(results, textvariable=self.summary, wraplength=340, justify="left").pack(
            anchor="w", pady=8
        )
        ttk.Label(results, text="SYS-02 result payload").pack(anchor="w")
        self.payload = ScrolledText(results, width=45, height=14, wrap="word", state="disabled")
        self.payload.pack(fill="both", expand=True)
        self._render()
        self._timer = root.after(0, self._tick)

    def _change_action(self, _event):
        self.controller.reset(self.action.get())
        self.retry()

    def retry(self):
        self.controller.reset()
        self.frozen = self.latest = None
        self.evaluated = False
        self._clear_regions()
        self.canvas.delete("all")
        if self.camera is None:
            self.start()
        self._render()

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
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _camera_error(self, message):
        self._release()
        self.latest = self.frozen = self.photo = None
        self.boxes.clear()
        self.canvas.delete("all")
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
        import cv2
        from PIL import Image, ImageTk

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((640, 360))
        self.preview_size = image.size
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.delete("preview")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="preview")
        self.canvas.tag_lower("preview")

    def freeze(self):
        if self.latest is None or self.frozen is not None:
            return
        self.frozen = self.latest.copy()
        self.frozen.flags.writeable = False
        self.evaluated = False
        self.controller.reset()
        self.hint.set("Draw a rectangle around one face, then select Run stubs.")
        self._render()

    def _start_drag(self, event):
        if self.frozen is None or self.evaluated:
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

    def _clear_regions(self):
        self.boxes.clear()
        self.drag_start = self.drag_outline = None
        self.canvas.delete("region")
        if self.frozen is not None:
            self.hint.set("Draw a rectangle around one face, then select Run stubs.")

    def evaluate(self):
        if self.frozen is None or self.evaluated:
            return
        self.evaluated = True
        request = self.controller.begin_capture(self.frozen, self.boxes)
        if request is not None:
            height, width, _ = request.face_crop.shape
            self.hint.set(f"Source crop {width} × {height} pixels. Resume preview for a fresh frame.")
            run_stubs(self.controller, request, StubAdapters(self.scenario.get()))
        else:
            self.hint.set("Resume preview and select a face region in a fresh frame.")
        self._render()

    def _render(self):
        state = self.controller.state
        messages = {
            "idle": "Preview",
            "checking": "Checking liveness",
            "live": "Live face (stub)",
            "spoof": "Spoof (stub). Attempt blocked. Retry with a fresh frame.",
            "uncertain": "Uncertain liveness (stub). Retry with a fresh frame.",
        }
        status = messages.get(state.ui_state, state.errors[0].message if state.errors else state.ui_state)
        emotion = state.emotion.label if state.emotion else "Unavailable"
        self.summary.set(
            f"{status}\nDemo identity: {self.controller.demo_identity}\n"
            f"Demo emotion: {emotion}\nSaving disabled in demo"
        )
        self.payload.configure(state="normal")
        self.payload.delete("1.0", "end")
        self.payload.insert("1.0", json.dumps(asdict(state), indent=2, allow_nan=False))
        self.payload.configure(state="disabled")
        self.freeze_button.configure(state="normal" if self.latest is not None and self.frozen is None else "disabled")
        can_select = self.frozen is not None and not self.evaluated
        self.run_button.configure(state="normal" if can_select else "disabled")
        self.clear_button.configure(state="normal" if can_select else "disabled")
        if self.frozen is None:
            self.hint.set("Freeze a preview frame to select a face region." if self.latest is not None else status)

    def close(self):
        self.root.after_cancel(self._timer)
        try:
            self._release()
        finally:
            self.root.destroy()


class DemoCamera:
    """Generated source for exercising the same GUI path without camera hardware."""

    def __init__(self):
        import cv2
        import numpy as np

        self.frame = np.full((360, 640, 3), (45, 33, 20), dtype=np.uint8)
        cv2.rectangle(self.frame, (240, 75), (400, 285), (140, 235, 100), 2)
        cv2.putText(self.frame, "Synthetic face region", (214, 313),
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
    args = parser.parse_args(argv)

    try:
        import cv2
        import tkinter as tk
        from PIL import ImageTk  # Ensure the GUI dependency is available at startup.
    except ImportError as error:
        parser.exit(1, f"Webcam demo dependencies unavailable: {error}\n")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(1, f"Cannot open the webcam demo window: {error}\n")
    if args.demo_camera:
        app = WebcamApp(root, DemoCamera, source_name="Synthetic camera")
    else:
        app = WebcamApp(root, lambda: cv2.VideoCapture(args.camera), source_name=f"Webcam {args.camera}")
    root.after_idle(app.start)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
