"""Fixed-position sandbox presentation. No history, matching or authorization."""


def current_result_fields(decision, *, running, face_count=None, display_name=None):
    """Format only the current decision; invalidated scores are never retained."""
    camera = "Camera: running" if running else "Camera: stopped"
    if not running:
        return camera, "No result yet", "Cosine: — (no current result)"
    if decision is None:
        state = (
            "No face detected"
            if face_count == 0
            else "Multiple faces — capture blocked"
            if face_count is not None and face_count > 1
            else "Checking current sample…"
        )
        return camera, state, "Cosine: — (no current result)"
    return (
        camera,
        "Known: " + (display_name or decision.test_name)
        if decision.accepted
        else "Unknown",
        "Cosine: "
        + (
            "N/A (empty gallery)"
            if decision.score is None
            else format(decision.score, ".6f")
        ),
    )


def set_enabled(button, enabled):
    """Avoid a ttk state mutation when the requested state is already applied."""
    state = "normal" if enabled else "disabled"
    if str(button.cget("state")) != state:
        button.configure(state=state)


def next_step(*, consent, running, references, profiles, target, minimum):
    if not consent:
        if profiles and not references:
            return "Profile ready. Confirm consent again and start the camera to test."
        return "Confirm consent above the preview, then start the camera."
    if not running:
        if profiles and not references:
            return "Start the camera again to test your saved profile."
        return "Start the camera and keep one face in view."
    if profiles and not references:
        return (
            "Look at the camera to check the live match. Record a test to keep a score."
        )
    if references >= target:
        return (
            "Save the test profile. Then start the camera again to compare a new photo."
        )
    if references >= minimum:
        return f"Take {target - references} more for the recommended {target}, or save this test profile now."
    return f"Keep one face in view and take {target} reference photos, one at a time."


class SandboxStatusPanel:
    """Reserve pixel-sized cells, following the main webcam GUI's layout pattern."""

    def __init__(self, parent, details, policy):
        import tkinter as tk
        from tkinter import ttk

        self.frame = ttk.Frame(parent, width=420, height=360)
        self.frame.grid(row=3, column=1, sticky="nw", padx=12, pady=8)
        self.frame.pack_propagate(False)
        self.variables = {}
        self.labels = {}
        self._values = {}
        self._cell(24, text="Live recognition", font=("TkDefaultFont", 13, "bold"))
        for name in ("camera", "decision"):
            var = tk.StringVar(master=parent)
            self.variables[name] = var
            self.labels[name] = self._cell(
                24, textvariable=var, font=("TkDefaultFont", 13)
            )
        self._cell(24, text="Next step", font=("TkDefaultFont", 13, "bold"))
        for name, height in (
            ("guide", 48),
            ("sample_progress", 24),
            ("session_summary", 24),
        ):
            var = tk.StringVar(master=parent)
            self.variables[name] = var
            self.labels[name] = self._cell(height, textvariable=var)
        self.metadata = ttk.Label(
            details,
            justify="left",
            wraplength=680,
            text=(
                f"Cosine threshold: {policy.cosine_threshold}\n"
                f"512D metric embeddings • CPU • Weights: {policy.model_version}\n"
                "UNVERIFIED • can_save=false\n"
                "A photograph may match. Recognition is not liveness."
            ),
        )
        self.metadata.pack(anchor="w", pady=(0, 8))
        for name in ("cosine", "alias", "references", "counts"):
            var = tk.StringVar(master=parent)
            self.variables[name] = var
            self.labels[name] = ttk.Label(details, textvariable=var)
            self.labels[name].pack(anchor="w")
        # Keep the main actions in fixed slots when guidance or result text changes.
        self.actions = []
        for _ in range(4):
            area = ttk.Frame(self.frame, width=420, height=42)
            area.pack(fill="x", anchor="w")
            area.pack_propagate(False)
            self.actions.append(area)

    def _cell(self, height, **options):
        from tkinter import ttk

        area = ttk.Frame(self.frame, width=420, height=height)
        area.pack(fill="x", anchor="w")
        area.pack_propagate(False)
        label = ttk.Label(area, anchor="nw", justify="left", wraplength=410, **options)
        label.pack(fill="both", expand=True)
        return label

    def update(self, **values):
        for key, value in values.items():
            if self._values.get(key) != value:
                self.variables[key].set(value)
                self._values[key] = value

    def render(self, decision, *, running, face_count=None, display_name=None):
        camera, result, cosine = current_result_fields(
            decision, running=running, face_count=face_count, display_name=display_name
        )
        self.update(camera=camera, decision=result, cosine=cosine)
