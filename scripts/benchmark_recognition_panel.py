"""Comparable 30-second native Tk presentation fixture, no webcam/model benchmark.

Uses the real app render/control paths with deterministic scalar states. Gallery
counts are fixtures, not enrollment or accuracy evidence; the separate native
sandbox check exercises actual capture, matching, consent and reset paths.
"""

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic


def main():
    root_path = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=root_path)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.duration < 30:
        p.error("At least 30 seconds required")
    if a.output.exists():
        p.error("Refusing to overwrite evidence")
    sys.path.insert(0, str(a.source_root / "src"))
    import tkinter as tk

    from facial_attendance.recognition.contracts import FaceInferenceContract
    from facial_attendance.recognition.evaluation import SandboxDecision
    from facial_attendance.ui.recognition_sandbox import RecognitionSandboxApp

    class Inference:
        contract = FaceInferenceContract()
        provenance = {
            "model_id": contract.model_id,
            "model_version": contract.model_version,
            "threshold": contract.cosine_threshold,
        }

    class Feed:
        def latest(self):
            return None

    root = tk.Tk()
    temporary = TemporaryDirectory()
    app = RecognitionSandboxApp(
        root,
        Inference(),
        lambda: None,
        lambda frame: [],
        source_kind="generated",
        repository=temporary.name,
    )
    root.after_cancel(app._timer)
    side = (
        app.status_panel.frame
        if hasattr(app, "status_panel")
        else app.start_button.master
    )

    def descendants(w):
        for child in w.winfo_children():
            yield child
            yield from descendants(child)

    widgets = list(descendants(side))
    vars_by_name = {}
    for w in widgets:
        try:
            name = str(w.cget("textvariable"))
            if name:
                vars_by_name[name] = tk.StringVar(root, name=name)
        except tk.TclError:
            pass
    writes = defaultdict(int)
    changes = defaultdict(int)
    redundant = defaultdict(int)
    last = {k: v.get() for k, v in vars_by_name.items()}
    for k, v in vars_by_name.items():

        def trace(*_, key=k, var=v):
            value = var.get()
            writes[key] += 1
            if value == last[key]:
                redundant[key] += 1
            else:
                changes[key] += 1
            last[key] = value

        v.trace_add("write", trace)
    state_writes = [0]
    state_redundant = [0]
    for button in (app.reference_button, app.evaluation_button, app.commit_button):
        original = button.configure

        def configure(*args, button=button, original=original, **kw):
            if "state" in kw:
                state_writes[0] += 1
                if str(button.cget("state")) == kw["state"]:
                    state_redundant[0] += 1
            return original(*args, **kw)

        button.configure = configure
    snapshots = {}
    transitions = defaultdict(int)
    distinct = defaultdict(set)
    intervals = []
    costs = []
    errors = []
    frames = [0]
    started = [0.0]
    last_tick = [0.0]
    checks = {}
    states = set()
    refs = set()
    passed = [False]

    def geometry():
        return {
            str(w): (w.winfo_x(), w.winfo_y(), w.winfo_width(), w.winfo_height())
            for w in (root, side, *widgets)
        }

    def finish():
        elapsed = monotonic() - started[0]
        report = {
            "scope": "deterministic scalar presentation fixture; no webcam or model inference",
            "physical_webcam_tested": False,
            "ticks": frames[0],
            "elapsed_s": elapsed,
            "tick_hz": frames[0] / elapsed,
            "geometry_transitions": dict(transitions),
            "geometry_transition_total": sum(transitions.values()),
            "geometry_distinct_counts": {k: len(v) for k, v in distinct.items()},
            "text_writes": sum(writes.values()),
            "text_changes": sum(changes.values()),
            "redundant_text_writes": sum(redundant.values()),
            "button_state_writes": state_writes[0],
            "redundant_button_state_writes": state_redundant[0],
            "tick_ms": {"mean": sum(costs) / len(costs), "max": max(costs)},
            "event_interval_ms": {
                "mean": sum(intervals) / len(intervals),
                "max": max(intervals),
            },
            "reference_counts": sorted(refs),
            "states": sorted(states),
            "callback_errors": errors,
            "source_sha256": {
                str(f.relative_to(a.source_root)): sha256(f.read_bytes()).hexdigest()
                for f in sorted((a.source_root / "src").rglob("*.py"))
            },
        }
        checks.update(
            panel_and_children_geometry_constant=not any(transitions.values()),
            no_redundant_text_writes=not sum(redundant.values()),
            no_redundant_button_state_writes=not state_redundant[0],
            repeated_for_at_least_30_seconds=elapsed >= 30,
            references_zero_to_ten=refs == set(range(11)),
            known_unknown_checking_and_stopped_covered=len(states) == 7,
            no_callback_errors=not errors,
            can_save_always_false=not app.session.can_save,
        )
        report["checks"] = checks
        report["passed"] = all(checks.values())
        passed[0] = report["passed"]
        app.feed = None
        app.session.references.clear()
        app.close()
        temporary.cleanup()
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "source_sha256",
                        "geometry_distinct_counts",
                        "geometry_transitions",
                    )
                },
                indent=2,
            )
        )

    def tick():
        t = monotonic()
        i = frames[0]
        if i:
            intervals.append((t - last_tick[0]) * 1000)
        last_tick[0] = t
        state = (i // 8) % 7
        states.add(state)
        count = (i // 14) % 11
        refs.add(count)
        app.session.references[:] = [None] * count
        app.session.set_consent(True)
        app.feed = None if state == 0 else Feed()
        app._detected = (
            None if state in (0, 1, 5, 6) else ((1, i), None, [(0, 0, 100, 100)], t)
        )
        app._decision = (
            SandboxDecision(
                "test-010",
                "Test subject 010",
                (0.462 if state == 2 else 0.9999999),
                True,
                app.session.policy.cosine_threshold,
            )
            if state in (2, 3)
            else SandboxDecision(
                None, None, -0.000001, False, app.session.policy.cosine_threshold
            )
            if state == 4
            else None
        )
        app._tick()
        root.after_cancel(app._timer)
        root.update_idletasks()
        current = geometry()
        for k, v in current.items():
            if k in snapshots and snapshots[k] != v:
                transitions[k] += 1
            distinct[k].add(v)
        snapshots.update(current)
        costs.append((monotonic() - t) * 1000)
        frames[0] += 1
        if frames[0] >= int(a.duration / 0.033) + 1:
            finish()
        else:
            root.after(33, tick)

    def start():
        root.update_idletasks()
        started[0] = monotonic()
        tick()

    def failure(kind, value, tb):
        errors.append(f"{kind.__name__}: {value}")
        finish()

    root.report_callback_exception = failure
    root.after(250, start)
    root.mainloop()
    return 0 if passed[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
