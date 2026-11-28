"""Comparable generated-input native Tk and actual-model CPU benchmarks.

No physical camera, person images, accuracy or webcam-performance claims.
--source-root selects a frozen source snapshot; the benchmark harness is shared.
"""

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import monotonic, sleep


def distribution(values):
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=root)
    p.add_argument("--mode", choices=("integrated", "stub"), default="integrated")
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--thread-bench", action="store_true")
    p.add_argument("--slow-ms", type=float, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--face-checkpoint",
        type=Path,
        default=Path("artifacts/models/recognition/face-recognition-resnet18.pt"),
    )
    p.add_argument(
        "--emotion-checkpoint",
        type=Path,
        default=Path("artifacts/models/emotion/best.pt"),
    )
    p.add_argument(
        "--yunet-model",
        type=Path,
        default=Path("artifacts/detectors/face_detection_yunet_2023mar.onnx"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Refusing to overwrite a benchmark")
    sys.path.insert(0, str(args.source_root / "src"))
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    y, x = np.indices((720, 1280))
    frame = np.stack(
        ((x // 3 + y // 2) % 256, (x // 2 + y // 4) % 256, (x // 4 + y // 3) % 256),
        axis=-1,
    ).astype(np.uint8)
    frame.flags.writeable = False
    box = (320, 140, 960, 600)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "system": platform.platform(),
        "scope": "Generated 1280x720 BGR, real CPU YuNet cost plus controlled single-face regions; no webcam",
        "mode": args.mode,
        "source_root": str(args.source_root),
        "physical_webcam_tested": False,
        "duration_requested_s": args.duration,
        "opencv_threads": 1,
        "torch_threads": args.torch_threads,
        "slow_model_injection_ms": args.slow_ms,
        "debug_open": args.debug,
        "versions": {
            key: version(key)
            for key in ("torch", "torchvision", "numpy", "Pillow", "opencv-python")
        },
        "source_sha256": {
            str(f.relative_to(args.source_root)): sha256(f.read_bytes()).hexdigest()
            for f in sorted((args.source_root / "src").rglob("*.py"))
        },
    }
    adapters = None
    if args.mode == "integrated":
        import torch

        from facial_attendance.ui.recognition_adapter import TrainedPreviewAdapters

        adapters = TrainedPreviewAdapters(
            args.face_checkpoint, emotion_checkpoint=args.emotion_checkpoint
        )
        torch.set_num_threads(args.torch_threads)
        # Warm both models before measuring, like an initialized steady-state GUI.
        for _ in range(5):
            adapters.face_inference.embed(frame, box)
            adapters.emotion_adapter.inference.predict_bgr(frame[140:600, 320:960])
        if args.thread_bench:
            rows = []
            for threads in (1, 2, 4):
                torch.set_num_threads(threads)
                for _ in range(8):
                    adapters.face_inference.embed(frame, box)
                    adapters.emotion_adapter.inference.predict_bgr(
                        frame[140:600, 320:960]
                    )
                timings = defaultdict(list)
                for _ in range(50):
                    start = monotonic()
                    adapters.face_inference.embed(frame, box)
                    timings["face_ms"].append((monotonic() - start) * 1000)
                    start = monotonic()
                    adapters.emotion_adapter.inference.predict_bgr(
                        frame[140:600, 320:960]
                    )
                    timings["emotion_ms"].append((monotonic() - start) * 1000)
                rows.append(
                    {
                        "threads": threads,
                        **{k: distribution(v) for k, v in timings.items()},
                    }
                )
            report["cpu_threads_benchmark"] = rows
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(rows, indent=2))
            return 0

    import tkinter as tk

    from facial_attendance.ui.face_detection import (
        ScaledFaceDetector,
        YuNetFaceDetector,
    )
    from facial_attendance.ui.webcam_app import LiveWebcamApp

    real_detector = ScaledFaceDetector(YuNetFaceDetector(args.yunet_model))
    real_detector(frame)
    detection_count = [0]

    def detector(image):
        real_detector(image)  # Real cost; generated pattern is not a detectable face.
        detection_count[0] += 1
        dx = 4 if detection_count[0] % 2 else -4
        return [(box[0] + dx, box[1], box[2] + dx, box[3])]

    class GeneratedCamera:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            pass

    root_tk = tk.Tk()
    app = LiveWebcamApp(
        root_tk,
        GeneratedCamera,
        source_name="Generated performance comparison",
        detector=detector,
        adapters=adapters,
        diagnostics=True,
    )
    app.metrics.interval = 3600
    app.metrics.report = lambda **kwargs: None
    timings = defaultdict(list)
    counts = defaultdict(int)
    geometry = []
    errors = []
    started = [None]
    last_preview = [None]
    last_beat = [None]

    def measured(owner, name, key, delay=0):
        original = getattr(owner, name)

        def call(*a, **kw):
            if delay:
                sleep(delay)
            start = monotonic()
            try:
                return original(*a, **kw)
            finally:
                if started[0] is not None:
                    timings[key].append((monotonic() - start) * 1000)

        setattr(owner, name, call)

    if adapters is not None:
        measured(adapters.face_inference, "embed", "face_total_ms", args.slow_ms / 1000)
        measured(adapters.face_inference.model, "embed", "face_forward_ms")
        measured(adapters.emotion_adapter.inference, "predict_bgr", "emotion_total_ms")
        measured(
            adapters.emotion_adapter.inference.model, "forward", "emotion_forward_ms"
        )
    original_draw = app._draw_frame

    def draw(image):
        now = monotonic()
        if started[0] is not None:
            if last_preview[0] is not None:
                timings["preview_interval_ms"].append((now - last_preview[0]) * 1000)
            last_preview[0] = now
        return original_draw(image)

    app._draw_frame = draw
    measured(app, "_render", "results_render_ms")
    for attr in ("summary", "model_info", "registration_prompt"):
        value = getattr(app, attr, None)
        if value is not None:
            value.trace_add(
                "write",
                lambda *a, key=attr: (
                    counts.__setitem__(key + "_writes", counts[key + "_writes"] + 1)
                    if started[0] is not None
                    else None
                ),
            )
    last_fields = {}

    def field_change(key, value):
        if last_fields.get(key) != value:
            if started[0] is not None:
                counts[key + "_changes"] += 1
            last_fields[key] = value

    if hasattr(app, "display_identity"):
        for key in ("identity", "emotion", "liveness", "status"):
            var = getattr(app, "display_" + key)
            var.trace_add(
                "write", lambda *a, key=key, var=var: field_change(key, var.get())
            )
    else:

        def summary_fields(*args):
            for line in app.summary.get().splitlines():
                for key in ("identity", "emotion", "liveness"):
                    if line.startswith(key.capitalize() + " ("):
                        field_change(key, line)

        app.summary.trace_add("write", summary_fields)
    if hasattr(app, "registration_button"):
        configure = app.registration_button.configure

        def button_configure(*a, **kwargs):
            if "state" in kwargs:
                field_change("registration_state", kwargs["state"])
            return configure(*a, **kwargs)

        app.registration_button.configure = button_configure

    def sample():
        now = monotonic()
        if started[0] is not None:
            if last_beat[0] is not None:
                timings["heartbeat_delay_ms"].append(
                    max(0, (now - last_beat[0]) * 1000 - 10)
                )
            last_beat[0] = now
            geo = (
                root_tk.winfo_width(),
                root_tk.winfo_height(),
                app.canvas.winfo_x(),
                app.canvas.winfo_y(),
            )
            if not geometry or geo != geometry[-1]:
                geometry.append(geo)
        root_tk.after(10, sample)

    def begin():
        started[0] = monotonic()
        app.metrics.snapshot(force=True)
        root_tk.after(round(args.duration * 1000), finish)

    def finish():
        elapsed = monotonic() - started[0]
        report.update(
            elapsed_s=elapsed,
            metrics=app.metrics.snapshot(force=True),
            distributions_ms={k: distribution(v) for k, v in timings.items()},
            widget_writes=dict(counts),
            geometry_transitions=len(geometry) - 1,
            geometry_sequence=geometry,
            callback_errors=errors,
            can_save=app.controller.state.can_save,
        )
        if hasattr(app, "results_panel"):
            report["primary_panel_size"] = [
                app.results_panel.winfo_width(),
                app.results_panel.winfo_height(),
            ]
            # Native behavior checks outside the timed comparison window.
            original_geometry = (
                root_tk.winfo_width(),
                root_tk.winfo_height(),
                app.canvas.winfo_rootx(),
                app.canvas.winfo_rooty(),
            )

            def cell_positions():
                return tuple(
                    (
                        w.winfo_rootx(),
                        w.winfo_rooty(),
                        w.winfo_width(),
                        w.winfo_height(),
                    )
                    for w in getattr(app, "result_cells", ())
                )

            original_cells = cell_positions()
            if hasattr(app, "display_identity"):
                app.display_identity.set("Generated long test name " * 4)
            else:
                app.summary.set(
                    "Current sample • UNVERIFIED\nIdentity: "
                    + "Generated long test name " * 4
                    + "\nEmotion: Positive\nLiveness: SIMULATED"
                )
            root_tk.update_idletasks()
            long_geometry = (
                root_tk.winfo_width(),
                root_tk.winfo_height(),
                app.canvas.winfo_rootx(),
                app.canvas.winfo_rooty(),
            )
            if not app._debug_visible:
                app._toggle_debug()
            root_tk.update_idletasks()
            # This is a developer-only control; check its fit after opening
            # Developer Mode, rather than requiring it in Standard Mode.
            report["diagnostics_button_visible"] = bool(
                app.diagnostics_button.winfo_ismapped()
            )
            report["diagnostics_button_bottom"] = (
                app.diagnostics_button.winfo_y() + app.diagnostics_button.winfo_height()
            )
            debug_geometry = (
                root_tk.winfo_width(),
                root_tk.winfo_height(),
                app.canvas.winfo_rootx(),
                app.canvas.winfo_rooty(),
            )

            def children(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from children(child)

            model_var = str(app.model_info) if hasattr(app, "model_info") else None
            primary_model_labels = []
            for child in children(app.results_panel):
                try:
                    if model_var and str(child.cget("textvariable")) == model_var:
                        primary_model_labels.append(str(child))
                except tk.TclError:
                    pass
            report["layout_checks"] = {
                "long_text_geometry_stable": original_geometry == long_geometry,
                "opening_diagnostics_preserves_preview_position": original_geometry
                == debug_geometry,
                "metadata_outside_primary_results": not primary_model_labels,
                "controls_fit_panel": report["diagnostics_button_visible"]
                and report["diagnostics_button_bottom"]
                <= app.results_panel.winfo_height(),
            }
            if original_cells:
                report["layout_checks"]["separate_result_cells_stay_fixed"] = (
                    original_cells == cell_positions()
                )
        app.close()

    root_tk.report_callback_exception = lambda kind, value, trace: errors.append(
        f"{kind.__name__}: {value}"
    )
    if args.debug:
        app._toggle_debug()
    root_tk.after_idle(app.start)
    root_tk.after(10, sample)
    root_tk.after(1500, begin)
    root_tk.mainloop()
    for worker in (app.feed, app.detector, app.inference):
        worker.thread.join(timeout=2)
    report["workers_stopped"] = all(
        not w.thread.is_alive() for w in (app.feed, app.detector, app.inference)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "metrics",
                    "distributions_ms",
                    "widget_writes",
                    "geometry_transitions",
                    "callback_errors",
                    "workers_stopped",
                )
            },
            indent=2,
        )
    )
    return (
        0 if not errors and report["workers_stopped"] and not report["can_save"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
