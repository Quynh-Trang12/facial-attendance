"""Presentation-only regressions; real Tk pixel geometry is checked natively."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from facial_attendance.recognition.evaluation import (
    RecognitionEvaluation,
    SandboxDecision,
)
from facial_attendance.ui.recognition_sandbox import RecognitionSandboxApp
from facial_attendance.ui.sandbox_status import (
    SandboxStatusPanel,
    current_result_fields,
    set_enabled,
)


def decision(score=0.8, accepted=True):
    return SandboxDecision(
        "test-001" if accepted else None,
        "Test subject 001" if accepted else None,
        score,
        accepted,
        RecognitionEvaluation().policy.cosine_threshold,
    )


class Value:
    def __init__(self):
        self.value = ""
        self.writes = 0

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self.writes += 1


class SandboxStatusTests(unittest.TestCase):
    def panel(self):
        panel = SandboxStatusPanel.__new__(SandboxStatusPanel)
        panel.variables = {
            k: Value()
            for k in (
                "camera",
                "decision",
                "cosine",
                "alias",
                "references",
                "counts",
                "guide",
                "sample_progress",
                "session_summary",
            )
        }
        panel._values = {}
        return panel

    def test_repeated_identical_result_has_no_additional_writes(self):
        panel = self.panel()
        panel.render(decision(), running=True, face_count=1)
        for _ in range(1000):
            panel.render(decision(), running=True, face_count=1)
        self.assertEqual(
            [v.writes for v in panel.variables.values()], [1, 1, 1, 0, 0, 0, 0, 0, 0]
        )

    def test_score_change_updates_only_score_field(self):
        panel = self.panel()
        panel.render(decision(), running=True)
        panel.render(decision(0.9), running=True)
        self.assertEqual(panel.variables["cosine"].get(), "Cosine: 0.900000")
        self.assertEqual(panel.variables["camera"].writes, 1)
        self.assertEqual(panel.variables["decision"].writes, 1)

    def test_known_unknown_checking_never_reuses_score(self):
        panel = self.panel()
        for item, expected in (
            (decision(), "Known: Test subject 001"),
            (decision(0.1, False), "Unknown"),
            (None, "Checking current sample…"),
        ):
            panel.render(item, running=True, face_count=1)
            self.assertEqual(panel.variables["decision"].get(), expected)
        self.assertIn("no current result", panel.variables["cosine"].get())

    def test_face_loss_is_explicit_and_has_no_historical_identity(self):
        camera, result, score = current_result_fields(None, running=True, face_count=0)
        self.assertEqual(result, "No face detected")
        self.assertNotIn("001", result + score)

    def test_multiple_faces_are_not_checking_or_known(self):
        self.assertIn(
            "Multiple faces", current_result_fields(None, running=True, face_count=2)[1]
        )

    def test_camera_stopped_clears_even_a_supplied_decision(self):
        values = current_result_fields(decision(), running=False)
        self.assertIn("stopped", values[0])
        self.assertNotIn("001", " ".join(values))
        self.assertIn("no current result", values[2])

    def test_reentry_renders_new_current_decision(self):
        panel = self.panel()
        panel.render(None, running=True, face_count=0)
        panel.render(decision(0.53), running=True, face_count=1)
        self.assertEqual(panel.variables["cosine"].get(), "Cosine: 0.530000")

    def test_cosine_format_does_not_modify_raw_decision(self):
        for raw in (-1.0, -0.000001, 0.4619967254593038, 1.0):
            value = decision(raw, raw >= 0.4619967254593038)
            self.assertEqual(
                current_result_fields(value, running=True)[2], f"Cosine: {raw:.6f}"
            )
            self.assertEqual(value.score, raw)
            self.assertFalse(value.can_save)

    def test_empty_gallery_does_not_invent_zero_score(self):
        self.assertEqual(
            current_result_fields(decision(None, False), running=True)[2],
            "Cosine: N/A (empty gallery)",
        )

    def test_button_state_only_changes_when_needed(self):
        button = Mock()
        button.cget.return_value = "disabled"
        for _ in range(1000):
            set_enabled(button, False)
        button.configure.assert_not_called()
        set_enabled(button, True)
        button.configure.assert_called_once_with(state="normal")

    def test_reference_counts_zero_to_ten_do_not_rewrite_other_fields(self):
        app = RecognitionSandboxApp.__new__(RecognitionSandboxApp)
        app.session = RecognitionEvaluation(source_kind="generated")
        app._cleanup_pending = False
        app.feed = None
        app._detected = None
        app._decision = None
        app.worker = SimpleNamespace(busy=False)
        app.reference_button = Mock()
        app.evaluation_button = Mock()
        app.commit_button = Mock()
        app.progress = Value()
        app.counts = Value()
        app.status_panel = self.panel()
        for count in range(11):
            app.session.references[:] = [None] * count
            app._refresh_controls()
            self.assertEqual(
                app.status_panel.variables["references"].get(),
                f"References: {count}/5 target (3–10)",
            )
        self.assertEqual(app.status_panel.variables["alias"].writes, 1)
        self.assertEqual(app.status_panel.variables["counts"].writes, 1)
        self.assertEqual(app.counts.writes, 1)
        app.session.reset()
        app._refresh_controls()
        self.assertIn("References: 0/", app.status_panel.variables["references"].get())
        self.assertFalse(app.session.can_save)

    def test_variable_lengths_affect_only_current_field(self):
        panel = self.panel()
        panel.render(None, running=False)
        for value in (
            "Unknown",
            "Known: Test subject 010",
            "Multiple faces — capture blocked",
        ):
            panel.update(decision=value)
        self.assertEqual(panel.variables["camera"].writes, 1)
        self.assertEqual(panel.variables["cosine"].writes, 1)
        self.assertEqual(panel.variables["decision"].writes, 4)


if __name__ == "__main__":
    unittest.main()
