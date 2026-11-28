"""Run real Tk widgets with controlled SYS-05 input. No webcam unless --camera is explicit.

No pixels or recordings are saved. Controlled regions are test fixtures, never
evidence of real face detection. Physical glasses/lighting checks remain manual.
"""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from facial_attendance.ui.face_detection import FaceDetectorUnavailable  # noqa: E402 — source checkout bootstrap precedes local imports.
from facial_attendance.ui.webcam_app import DemoCamera, LiveWebcamApp  # noqa: E402 — source checkout bootstrap precedes local imports.
from facial_attendance.ui.webcam_prototype import Completion, Emotion  # noqa: E402 — source checkout bootstrap precedes local imports.

ONE = [(160, 70, 480, 300)]
TWO = [(40, 50, 200, 280), (350, 50, 520, 280)]


class UnavailableCamera:
    def isOpened(self):
        return False

    def release(self):
        pass


def run_case(
    name,
    scenario="Live match",
    *,
    boxes=ONE,
    mode=None,
    visible=False,
    duration=4,
    camera=None,
):
    import tkinter as tk

    import cv2

    root = tk.Tk()
    if not visible:
        root.withdraw()
    observed = {
        "states": set(),
        "seen_live": False,
        "retried": False,
        "stale_rejected": None,
        "frames_observed": 0,
        "source_shape": None,
    }
    calls = [0]
    opens = [0]
    callbacks = []
    result = {}

    def detector(frame):
        calls[0] += 1
        if mode == "missing":
            raise FaceDetectorUnavailable("Controlled detector-unavailable fixture")
        if mode == "loss" and calls[0] >= 3:
            return []
        if mode == "multiple_after_live" and calls[0] >= 3:
            return TWO
        return boxes

    def factory():
        opens[0] += 1
        if mode == "camera_missing" or (mode == "retry" and opens[0] == 1):
            return UnavailableCamera()
        return DemoCamera()

    def open_physical():
        cap = cv2.VideoCapture(camera)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    app = LiveWebcamApp(
        root,
        open_physical if camera is not None else factory,
        source_name="Controlled native Tk smoke"
        if camera is None
        else f"Physical camera {camera}",
        scenario=scenario,
        detector=None if mode == "real_detector" or camera is not None else detector,
        diagnostics=False,
    )
    started = monotonic()

    def finish(passed, reason):
        if result:
            return
        app._toggle_debug()
        payload = json.loads(app.payload.get("1.0", "end"))
        if payload["can_save"] is not False:
            passed, reason = False, "Stub attempted to enable saving"
        result.update(
            name=name,
            passed=passed,
            reason=reason,
            elapsed_s=round(monotonic() - started, 3),
            states=sorted(observed["states"]),
            ui_state=app.controller.state.ui_state,
            errors=[asdict(error) for error in app.controller.state.errors],
            can_save=payload["can_save"],
            context=payload["context"],
            source_shape=observed["source_shape"],
            frames_observed=observed["frames_observed"],
            stale_rejected=observed["stale_rejected"],
            callback_errors=callbacks,
        )
        app.close()

    def callback_error(kind, value, trace):
        callbacks.append(f"{kind.__name__}: {value}")
        try:
            finish(False, "Tk callback raised an exception")
        except Exception:
            root.destroy()

    root.report_callback_exception = callback_error

    def check():
        state = app.controller.state
        observed["states"].add(state.ui_state)
        packet = app.feed.latest()
        if packet is not None and packet.frame is not None:
            observed["frames_observed"] = max(
                observed["frames_observed"], packet.sequence
            )
            observed["source_shape"] = list(packet.frame.shape)
        if camera is not None:
            if monotonic() - started >= duration:
                finish(
                    observed["frames_observed"] > 0 and not callbacks,
                    "Physical capture only; owner face/glasses/lighting acceptance is separate",
                )
                return
        elif mode == "retry":
            if state.errors and not observed["retried"]:
                app.retry()
                observed["retried"] = True
            elif state.ui_state == "live" and observed["retried"]:
                finish(True, "Native Retry recovered into a fresh simulated result")
                return
        elif mode == "stale":
            if state.ui_state == "live" and not observed["retried"]:
                late = Completion(state.context, Emotion(0, "Negative"))
                app.retry()
                observed["retried"] = True
                observed["stale_rejected"] = not app.controller.receive(late)
            elif state.ui_state == "live" and observed["retried"]:
                finish(
                    observed["stale_rejected"],
                    "Native reset rejected the original-context late callback",
                )
                return
        elif mode in {"loss", "multiple_after_live"}:
            observed["seen_live"] |= state.ui_state == "live"
            expected = "no_face" if mode == "loss" else "multiple_faces"
            if observed["seen_live"] and state.ui_state == expected:
                cleared = all(
                    getattr(state, field) is None
                    for field in ("recognition", "liveness", "emotion")
                )
                finish(cleared, "Detected scene invalidation cleared all model fields")
                return
        else:
            expected = (
                "error"
                if mode in {"missing", "camera_missing"} or name == "invalid_box"
                else "no_face"
                if not boxes or mode == "real_detector"
                else "multiple_faces"
                if len(boxes) > 1
                else "spoof"
                if scenario == "Spoof"
                else "uncertain"
                if scenario == "Uncertain"
                else "live"
            )
            if state.ui_state == expected:
                finish(True, "Expected native GUI state reached with controlled input")
                return
        if monotonic() - started >= duration + 2:
            finish(False, "Timed out waiting for expected GUI transition")
            return
        root.after(25, check)

    root.after_idle(app.start)
    root.after(25, check)
    root.mainloop()
    for worker in (app.feed, app.detector, app.inference):
        worker.thread.join(timeout=1)
    result["workers_stopped"] = all(
        not worker.thread.is_alive()
        for worker in (app.feed, app.detector, app.inference)
    )
    result["passed"] = bool(
        result.get("passed") and result["workers_stopped"] and not callbacks
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument(
        "--camera", type=int, help="Explicit optional physical-camera-only probe"
    )
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite an existing smoke report")
    import cv2

    cv2.setNumThreads(1)
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "opencv": version("opencv-python"),
        "scope": "Native Tk with generated input and explicit controlled regions; physical validation separate",
        "results": [],
    }
    cases = [
        ("real_yunet_no_face", "Live match", ONE, "real_detector"),
        ("live_match", "Live match", ONE, None),
        ("live_unknown", "Live unknown", ONE, None),
        ("spoof", "Spoof", ONE, None),
        ("uncertain", "Uncertain", ONE, None),
        ("multiple_faces", "Live match", TWO, None),
        ("no_face", "Live match", [], None),
        ("invalid_box", "Live match", [(-1, 0, 20, 20)], None),
        ("detector_unavailable", "Live match", ONE, "missing"),
        ("camera_unavailable", "Live match", ONE, "camera_missing"),
        ("retry_recovery", "Live match", ONE, "retry"),
        ("face_loss", "Live match", ONE, "loss"),
        ("multiple_after_live", "Live match", ONE, "multiple_after_live"),
        ("stale_after_reset", "Live match", ONE, "stale"),
    ]
    try:
        if args.camera is not None:
            report["scope"] = (
                "Physical capture probe only; owner validation required for face/glasses/lighting"
            )
            report["results"].append(
                run_case(
                    "physical_capture",
                    visible=args.visible,
                    duration=args.duration,
                    camera=args.camera,
                )
            )
        else:
            for name, scenario, boxes, mode in cases:
                report["results"].append(
                    run_case(
                        name,
                        scenario,
                        boxes=boxes,
                        mode=mode,
                        visible=args.visible,
                        duration=args.duration,
                    )
                )
    except Exception as error:
        report["unavailable"] = f"{type(error).__name__}: {error}"
    report["passed"] = (
        bool(report["results"])
        and all(row["passed"] for row in report["results"])
        and "unavailable" not in report
    )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
