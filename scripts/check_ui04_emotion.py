"""Verify the supplied Emotion checkpoint; optional native Tk uses generated pixels only.

No physical camera, face detection accuracy, identity or liveness claims. Refuses
to overwrite reports. The checkpoint stays outside the source tree.
"""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.emotion.inference import (  # noqa: E402 — source checkout bootstrap precedes local imports.
    EmotionInference,
    prediction_from_logits,
    preprocess_bgr_crop,
)
from facial_attendance.ui.emotion_adapter import (  # noqa: E402 — source checkout bootstrap precedes local imports.
    MixedEmotionAdapters,
    TrainedEmotionAdapter,
)
from facial_attendance.ui.live_preview import evaluate_adapters  # noqa: E402 — source checkout bootstrap precedes local imports.
from facial_attendance.ui.webcam_prototype import PrototypeController  # noqa: E402 — source checkout bootstrap precedes local imports.


def native_check(checkpoint, engine):
    import tkinter as tk

    from facial_attendance.ui.webcam_app import DemoCamera, LiveWebcamApp

    phase = ["initial"]
    one = [(160, 70, 480, 300)]
    two = one + [(30, 70, 130, 300)]

    def detector(frame):
        return [] if phase[0] == "loss" else two if phase[0] == "multiple" else one

    root = tk.Tk()
    root.withdraw()
    adapters = MixedEmotionAdapters("Live match", checkpoint)
    adapters.emotion_adapter = TrainedEmotionAdapter(checkpoint, inference=engine)
    app = LiveWebcamApp(
        root,
        DemoCamera,
        source_name="Generated UI-04 validation input",
        detector=detector,
        adapters=adapters,
    )
    result = {
        "input": "Generated BGR image and controlled regions; not a physical webcam test",
        "checks": [],
        "callback_errors": [],
    }
    started = monotonic()
    late = [None]

    def finish(error=None):
        if error:
            result["error"] = error
        app.close()

    def callback_error(kind, value, trace):
        result["callback_errors"].append(f"{kind.__name__}: {value}")
        finish("Native Tk callback failed")

    root.report_callback_exception = callback_error

    def check():
        state = app.controller.state
        if state.can_save:
            finish("Mixed mode attempted to enable saving")
            return
        if phase[0] in ("initial", "retry") and state.emotion is not None:
            app._toggle_debug() if not app._debug_visible else None
            payload = json.loads(app.payload.get("1.0", "end"))
            if (
                state.emotion.source != "model"
                or payload["emotion"]["source"] != "model"
                or "Emotion (trained CNN)" not in app.summary.get()
                or "Liveness (simulated)" not in app.summary.get()
                or "epoch 9" not in app.model_info.get()
            ):
                finish("Trained/simulated provenance display mismatch")
                return
            result["checks"].append(phase[0] + "_trained_result")
            result["current_result"] = {
                "context": payload["context"],
                "emotion": payload["emotion"],
                "can_save": payload["can_save"],
            }
            if phase[0] == "initial":
                from facial_attendance.ui.webcam_prototype import Completion

                late[0] = Completion(state.context, state.emotion)
                phase[0] = "retry"
                app.retry()
                if app.controller.receive(late[0]):
                    finish("Retry accepted a stale trained result")
                    return
                result["checks"].append("retry_rejected_stale_completion")
            else:
                phase[0] = "loss"
        elif phase[0] == "loss" and state.ui_state == "no_face":
            if any(
                getattr(state, field) is not None
                for field in ("recognition", "liveness", "emotion")
            ):
                finish("Face loss retained current results")
                return
            result["checks"].append("face_loss_cleared_results")
            phase[0] = "multiple"
        elif phase[0] == "multiple" and state.ui_state == "multiple_faces":
            if state.emotion is not None or state.recognition is not None:
                finish("Multiple faces retained current results")
                return
            result["checks"].append("multiple_faces_blocked_and_cleared")
            finish()
            return
        if monotonic() - started > 15:
            finish("Timed out waiting for native GUI transitions")
            return
        root.after(25, check)

    root.after_idle(app.start)
    root.after(25, check)
    root.mainloop()
    for worker in (app.feed, app.detector, app.inference):
        worker.thread.join(timeout=2)
    result["workers_stopped"] = all(
        not worker.thread.is_alive()
        for worker in (app.feed, app.detector, app.inference)
    )
    result["passed"] = (
        len(result["checks"]) == 5
        and result["workers_stopped"]
        and not result["callback_errors"]
        and "error" not in result
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--native-gui",
        action="store_true",
        help="Real Tk widgets with generated input, no camera",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    import numpy as np
    import torch

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("torch", "numpy", "Pillow", "opencv-python")
        },
        "physical_webcam_tested": False,
        "accuracy_measured": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for folder in ("src", "configs", "tests", "scripts")
            for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".toml", ".json")
        },
    }
    try:
        engine = EmotionInference(args.checkpoint).load()
        frame = np.full((80, 100, 3), (11, 22, 33), dtype=np.uint8)
        original = frame.copy()
        frame.flags.writeable = False
        controller = PrototypeController()
        request = controller.begin_capture(frame, [(10, 20, 50, 60)])
        tensor = preprocess_bgr_crop(request.face_crop)
        with torch.inference_mode():
            logits = engine.model(tensor)
        prediction = prediction_from_logits(logits)
        adapters = MixedEmotionAdapters("Live match", args.checkpoint)
        adapters.emotion_adapter = TrainedEmotionAdapter(
            args.checkpoint, inference=engine
        )
        completions = evaluate_adapters((request, adapters))
        accepted = [controller.receive(item) for item in completions]
        result = controller.state.emotion
        controller.reset()
        report.update(
            checkpoint=engine.provenance,
            tensor_shape=list(tensor.shape),
            tensor_dtype=str(tensor.dtype),
            logits_shape=list(logits.shape),
            logits=logits.tolist(),
            prediction=asdict(prediction),
            accepted_context_bound_completions=all(accepted),
            emotion_source=result.source,
            source_unchanged=bool(np.array_equal(frame, original)),
            stale_after_reset_rejected=not controller.receive(completions[-1]),
            can_save=controller.state.can_save,
        )
        report["passed"] = bool(
            all(accepted)
            and result.source == "model"
            and report["source_unchanged"]
            and report["stale_after_reset_rejected"]
            and not report["can_save"]
        )
        if args.native_gui:
            report["native_gui"] = native_check(args.checkpoint, engine)
            report["passed"] &= report["native_gui"]["passed"]
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
