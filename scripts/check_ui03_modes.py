"""Mapped native dual-mode checks on generated input and the actual checkpoints.

No physical camera or genuine identity is used. Output is local test evidence;
choose an ignored artifacts path, not a tracked diagnostic log location.
"""

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-checkpoint", required=True)
    parser.add_argument("--emotion-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    import tkinter as tk

    import cv2

    from facial_attendance.recognition.gallery_matching import CONTROLLED_BOX
    from facial_attendance.ui.recognition_adapter import (
        ControlledCamera,
        TrainedPreviewAdapters,
    )
    from facial_attendance.ui.webcam_app import LiveWebcamApp

    cv2.setNumThreads(1)
    adapters = TrainedPreviewAdapters(
        args.face_checkpoint,
        emotion_checkpoint=args.emotion_checkpoint,
        controlled_registration=True,
    )
    adapters.matcher.commit(
        adapters.matcher.prepare(("mode-fixture", "Generated Mode Person"))
    )
    opened = [0]

    def camera():
        opened[0] += 1
        return ControlledCamera()

    root = tk.Tk()
    app = LiveWebcamApp(
        root,
        camera,
        source_name="Generated mode verification",
        adapters=adapters,
        detector=lambda _: [CONTROLLED_BOX],
        diagnostics=True,
        runtime_info={
            "camera_index": None,
            "input": "generated",
            "face_detector": "controlled regions",
            "processing_device": "CPU",
        },
    )
    app.metrics.report = lambda **kw: None
    report = {
        "input": "generated only; actual supplied trained checkpoints",
        "physical_webcam_tested": False,
        "checks": [],
        "callback_errors": [],
    }

    def ids():
        return tuple(
            id(v)
            for v in (
                app.feed,
                app.detector,
                app.inference,
                app.live,
                app.controller,
                app.adapters,
                adapters.face_inference.model,
                adapters.emotion_adapter.inference.model,
                adapters.matcher,
                adapters.matcher._gallery,
            )
        )

    shared = ids()
    window_id = [None]
    geometry = [None]
    sequence = [None]
    refresh = [0]
    frozen = {}
    clipboard_before = [None]
    clipboard_changed = [False]

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        report["checks"].append(name)

    def bounds(w):
        return (w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height())

    def primary_geometry():
        return (
            bounds(root),
            bounds(app.canvas),
            tuple(bounds(w) for w in app.result_cells),
        )

    def finish(error=None):
        if error:
            report["error"] = str(error)
        if clipboard_changed[0]:
            app.debug.clipboard_clear()
            if clipboard_before[0] is not None:
                app.debug.clipboard_append(clipboard_before[0])
        report["metrics"] = app.metrics.snapshot(force=True)
        report["diagnostics_refresh_count"] = (
            app._diagnostics_view.refresh_count if app._diagnostics_view else 0
        )
        report["source_sha256"] = {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src").rglob("*.py")
        }
        app.close()

    def failed(kind, value, tb):
        report["callback_errors"].append(f"{kind.__name__}: {value}")
        finish(value)

    root.report_callback_exception = failed

    def begin():
        check(
            "standard_is_default_even_with_metrics_enabled",
            not app._developer_mode and not app._debug_visible,
        )
        check(
            "standard_hides_diagnostics_button",
            not app.diagnostics_button.winfo_ismapped(),
        )
        check("standard_keeps_mode_switch_visible", app.mode_button.winfo_ismapped())
        check(
            "preview_and_results_visible",
            app.photo is not None and app.canvas.winfo_ismapped(),
        )
        check(
            "standard_labels_are_concise",
            app.display_identity.get().startswith("Identity: ")
            and app.display_emotion.get().startswith("Emotion: "),
        )
        check(
            "standard_omits_internal_values",
            all(
                v not in app.display_emotion.get() + app.hint.get()
                for v in ("800", "450", "continuity", "confidence", "smoothed")
            ),
        )
        check(
            "standard_safety_is_explicit",
            "SIMULATED" in app.display_liveness.get()
            and not app.controller.state.can_save,
        )
        app.registration_id.set("retained-id")
        app.registration_name.set("Retained typed name")
        app._show_registration()
        geometry[0] = None
        app.mode_button.invoke()
        root.after(200, opened_debug)

    def opened_debug():
        panel = app._diagnostics_view
        root.update_idletasks()
        check("developer_mode_activated", app._developer_mode)
        check(
            "developer_shows_diagnostics_button",
            app.diagnostics_button.winfo_ismapped(),
        )
        check(
            "diagnostics_opens_as_visible_toplevel",
            app.debug.winfo_ismapped() and panel.runtime.winfo_ismapped(),
        )
        x, y, w, h = bounds(app.debug)
        check(
            "diagnostics_fits_screen",
            0 <= x
            and 0 <= y
            and x + w <= root.winfo_screenwidth()
            and y + h <= root.winfo_screenheight(),
        )
        content = json.loads(panel.runtime.get("1.0", "end"))
        check(
            "runtime_metadata_available",
            content["recognition"]["configured_threshold"]
            == adapters.face_inference.contract.cosine_threshold
            and content["runtime"]["processing_device"] == "CPU",
        )
        check(
            "no_invented_trained_liveness",
            content["liveness"]["trained_threshold"] is None
            and content["liveness"]["trained_model_metadata"] is None,
        )
        check(
            "raw_and_display_sections_separate",
            "historical_display_audit" in content["emotion"]
            and "raw_current_prediction" in content["emotion"],
        )
        panel.runtime.yview_moveto(1.0)
        check("runtime_content_scrolls", panel.runtime.yview()[0] > 0)
        panel.notebook.select(1)
        root.after(30, raw_ready)

    def raw_ready():
        if app.controller.state.emotion is None:
            root.after(30, raw_ready)
            return
        app._last_payload = None
        app._last_render = None
        app._render()
        root.update_idletasks()
        app.payload.yview_moveto(1.0)
        report["raw_scroll_probe"] = {
            "lines": len(app.payload.get("1.0", "end").splitlines()),
            "height": app.payload.winfo_height(),
            "yview": app.payload.yview(),
        }
        check(
            "raw_json_is_readonly_and_scrollable",
            app.payload.yview()[0] > 0 and app.payload.cget("state") == "disabled",
        )
        raw = json.loads(app.payload.get("1.0", "end"))
        check("raw_json_keeps_false_save", raw["can_save"] is False)
        window_id[0] = str(app.debug)
        sequence[0] = app.feed.latest().sequence
        # Invoke the actual window-manager close callback rather than destroy.
        app.debug.tk.call(app.debug.protocol("WM_DELETE_WINDOW"))
        root.after(250, closed_debug)

    def closed_debug():
        check(
            "wm_close_hides_diagnostics_only",
            not app._debug_visible
            and not app.debug.winfo_ismapped()
            and not app._closed,
        )
        check(
            "capture_continues_when_diagnostics_closed",
            app.feed.latest().sequence > sequence[0],
        )
        app.diagnostics_button.invoke()
        root.after(100, reopened)

    def reopened():
        check(
            "reopen_reuses_same_toplevel",
            str(app.debug) == window_id[0] and app.debug.winfo_ismapped(),
        )
        check(
            "only_one_diagnostics_window",
            len([w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]) == 1,
        )
        root.geometry("1120x700")
        root.after(150, minimum_size)

    def minimum_size():
        root.update_idletasks()
        geometry[0] = primary_geometry()
        x, y, w, h = bounds(app.registration_submit)
        rx, ry, rw, rh = bounds(root)
        check(
            "registration_controls_fit_minimum_window",
            rx <= x and ry <= y and x + w <= rx + rw and y + h <= ry + rh,
        )
        sequence[0] = app.feed.latest().sequence
        refresh[0] = app._diagnostics_view.refresh_count
        for _ in range(12):
            app._switch_mode()
        root.after(350, switched)

    def switched():
        check(
            "mode_switching_preserves_main_geometry", primary_geometry() == geometry[0]
        )
        check(
            "mode_switching_keeps_pipeline_models_and_gallery",
            ids() == shared and len(adapters.matcher._gallery.people) == 1,
        )
        check(
            "mode_switching_keeps_registration_input",
            app.registration_id.get() == "retained-id"
            and app.registration_name.get() == "Retained typed name",
        )
        check(
            "mode_switching_keeps_camera_session",
            opened[0] == 1 and app.feed.latest().sequence > sequence[0],
        )
        check(
            "both_modes_keep_authorization_blocked", not app.controller.state.can_save
        )
        # One uninterrupted second establishes the automatic refresh bound.
        app._diagnostics_view.notebook.select(0)
        refresh[0] = app._diagnostics_view.refresh_count
        root.after(1000, refresh_checked)

    def refresh_checked():
        check(
            "automatic_developer_refresh_is_bounded",
            app._diagnostics_view.refresh_count - refresh[0] <= 5,
        )
        app._set_mode(False)
        root.after(100, standard_again)

    def standard_again():
        check(
            "standard_mode_restores_and_hides_details",
            not app._developer_mode
            and not app._debug_visible
            and app.display_emotion.get().startswith("Emotion: "),
        )
        check(
            "standard_automatically_hides_button_and_window",
            not app.diagnostics_button.winfo_ismapped()
            and not app.debug.winfo_ismapped(),
        )
        check(
            "standard_keeps_mode_control_after_return", app.mode_button.winfo_ismapped()
        )
        check(
            "standard_hides_technical_guidance",
            all(
                v
                not in app.hint.get()
                + app.source_label.get()
                + app.display_emotion.get()
                for v in ("800", "450", "continuity", "Developer Mode", "display only")
            ),
        )
        check("unchanged_models_not_reloaded", ids() == shared)
        app.mode_button.invoke()
        root.after(100, developer_again)

    def developer_again():
        check(
            "returning_to_developer_reopens_same_window",
            app._developer_mode
            and app.diagnostics_button.winfo_ismapped()
            and app.debug.winfo_ismapped()
            and str(app.debug) == window_id[0],
        )
        check("developer_keeps_mode_switch_visible", app.mode_button.winfo_ismapped())
        app.mode_button.invoke()
        root.after(100, final_standard)

    def final_standard():
        check(
            "repeated_return_to_standard_closes_diagnostics",
            not app._developer_mode
            and not app.diagnostics_button.winfo_ismapped()
            and not app.debug.winfo_ismapped(),
        )
        app.mode_button.invoke()
        root.after(100, controls_ready)

    def controls_ready():
        if app.controller.state.emotion is None:
            root.after(30, controls_ready)
            return
        panel = app._diagnostics_view
        panel.notebook.select(1)
        app._last_payload = None
        app._last_render = None
        app._render()
        root.update_idletasks()
        check(
            "diagnostic_controls_visible_only_in_developer",
            all(
                w.winfo_ismapped()
                for w in (
                    panel.pause_button,
                    panel.copy_button,
                    panel.export_button,
                    panel.completed_button,
                )
            ),
        )
        check(
            "live_status_has_snapshot_timestamp",
            "LIVE" in panel.status.get()
            and panel._snapshot_meta[1]["timestamp"] in panel.status.get(),
        )
        # A stable JSON key remains selected even when context values change.
        start = app.payload.search("can_save", "1.0")
        app.payload.tag_add("sel", start, f"{start}+8c")
        frozen["selection"] = app.payload.get("sel.first", "sel.last")
        root.after(400, selection_ready)

    def selection_ready():
        panel = app._diagnostics_view
        check(
            "live_raw_refresh_preserves_unchanged_selection",
            app.payload.get("sel.first", "sel.last") == frozen["selection"],
        )
        panel.pause_button.invoke()
        root.update_idletasks()
        frozen["raw"] = app.payload.get("1.0", "end-1c")
        frozen["runtime"] = panel.runtime.get("1.0", "end-1c")
        frozen["timestamp"] = panel._snapshot_meta[1]["timestamp"]
        frozen["completed"] = app._completed_diagnostic
        sequence[0] = app.feed.latest().sequence
        root.after(900, paused_ready)

    def paused_ready():
        panel = app._diagnostics_view
        check(
            "pause_freezes_both_actual_text_widgets",
            panel.paused
            and app.payload.get("1.0", "end-1c") == frozen["raw"]
            and panel.runtime.get("1.0", "end-1c") == frozen["runtime"],
        )
        check(
            "paused_status_keeps_frozen_timestamp",
            "PAUSED" in panel.status.get()
            and frozen["timestamp"] in panel.status.get(),
        )
        check(
            "camera_continues_during_paused_diagnostics",
            app.feed.latest().sequence > sequence[0],
        )
        check(
            "new_inference_completes_while_paused",
            app._completed_diagnostic is not frozen["completed"],
        )
        check(
            "paused_diagnostics_do_not_authorize",
            app.controller.state.can_save is False and ids() == shared,
        )
        try:
            clipboard_before[0] = app.debug.clipboard_get()
        except tk.TclError:
            clipboard_before[0] = None
        clipboard_changed[0] = True
        panel.copy_button.invoke()
        check(
            "copy_button_copies_exact_frozen_json",
            app.debug.clipboard_get() == frozen["raw"],
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "safe.json"
            with patch(
                "tkinter.filedialog.asksaveasfilename", return_value=str(target)
            ):
                panel.export_button.invoke()
            exported = target.read_text()
            data = json.loads(exported)
            check(
                "export_is_privacy_safe",
                all(
                    value not in exported
                    for value in (
                        "Generated Mode Person",
                        "mode-fixture",
                        "/Users/",
                        "embeddings",
                        "person_id",
                    )
                ),
            )
            check(
                "export_keeps_displayed_snapshot_timestamp",
                data["snapshot"]["timestamp"] == frozen["timestamp"]
                and data["snapshot"]["historical"]
                and data["authorization"]["can_save"] is False,
            )
        expected = asdict(app._completed_diagnostic.state)
        panel.completed_button.invoke()
        root.update_idletasks()
        complete = json.loads(panel.completed.get("1.0", "end"))
        # A recognition-only batch is also a completed inference between Emotion samples.
        check(
            "completed_snapshot_is_explicitly_historical",
            panel.paused
            and "HISTORICAL" in panel.status.get()
            and complete == json.loads(json.dumps(expected))
            and complete["can_save"] is False,
        )
        frozen["historical"] = panel.completed.get("1.0", "end-1c")
        panel.pause_button.invoke()
        root.after(350, resumed_ready)

    def resumed_ready():
        panel = app._diagnostics_view
        check(
            "resume_updates_current_json_again",
            not panel.paused
            and "LIVE" in panel.status.get()
            and app.payload.get("1.0", "end-1c") != frozen["raw"],
        )
        check(
            "completed_history_is_separate_from_resumed_current",
            panel.completed.get("1.0", "end-1c") == frozen["historical"]
            and panel.notebook.index(panel.notebook.select()) == 1,
        )
        app.mode_button.invoke()
        root.after(100, controls_hidden)

    def controls_hidden():
        panel = app._diagnostics_view
        check(
            "all_diagnostic_controls_hidden_in_standard",
            not app._developer_mode
            and all(
                not w.winfo_ismapped()
                for w in (
                    panel.pause_button,
                    panel.copy_button,
                    panel.export_button,
                    panel.completed_button,
                )
            ),
        )
        finish()

    root.after_idle(app.start)
    root.after(2200, begin)
    root.after(20000, lambda: finish("Native mode check timeout"))
    root.mainloop()
    for worker in (app.feed, app.detector, app.inference, app._registration_worker):
        worker.thread.join(2)
    report["workers_stopped"] = all(
        not w.thread.is_alive()
        for w in (app.feed, app.detector, app.inference, app._registration_worker)
    )
    report["passed"] = (
        not report.get("error")
        and not report["callback_errors"]
        and report["workers_stopped"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source_sha256", "metrics")},
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
