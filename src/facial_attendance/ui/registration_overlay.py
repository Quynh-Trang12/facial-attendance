"""Canvas-only guidance for the current registration reference."""

from math import sin, tau


class RegistrationOverlay:
    TAG = "registration-guidance"
    COLORS = {
        "guidance": "#89C7FF",
        "accepted": "#80D5AE",
        "retry": "#F0C17D",
        "spoof": "#FFA3A6",
    }
    ANIMATION_STEP = 0.08

    def __init__(self, canvas):
        self.canvas = canvas
        self._items = None
        self._layout_key = None
        self._animation_step = None

    def clear(self):
        if self._items is not None:
            self.canvas.delete(self.TAG)
            self._items = None
            self._layout_key = self._animation_step = None

    def update(self, guidance, size, now):
        width, height = size
        if guidance is None or width < 180 or height < 120:
            self.clear()
            return
        if self._items is None:
            self._items = (
                self.canvas.create_oval(
                    0, 0, 0, 0, width=2, dash=(5, 6), tags=self.TAG
                ),
                self.canvas.create_rectangle(
                    0, 0, 0, 0, fill="#151B24", width=1, tags=self.TAG
                ),
                self.canvas.create_line(
                    0,
                    0,
                    0,
                    0,
                    width=3,
                    arrow="last",
                    arrowshape=(8, 10, 4),
                    tags=self.TAG,
                ),
                self.canvas.create_text(
                    0,
                    0,
                    anchor="w",
                    fill="#F2F5FA",
                    font=("TkDefaultFont", 12, "bold"),
                    tags=self.TAG,
                ),
                self.canvas.create_text(
                    0, 0, anchor="w", font=("TkDefaultFont", 10), tags=self.TAG
                ),
            )
            # Keep the detector outline and name above the dashed guide and card.
            self.canvas.tag_raise(self.TAG, "preview")

        guide, card, arrow, heading, detail = self._items
        pose = guidance.pose
        direction = pose.direction if pose is not None else None
        layout_key = guidance, size
        if layout_key != self._layout_key:
            self._layout_key = layout_key
            self._animation_step = None
            accent = self.COLORS[guidance.feedback]
            center_x = width / 2
            card_width = min(384, width - 16)
            left, bottom = center_x - card_width / 2, height - 10
            top = bottom - 54
            text_left = left + (53 if direction else 14)
            self._arrow_center = left + 27, (top + bottom) / 2

            radius_x = min(width * 0.19, height * 0.26)
            self.canvas.coords(
                guide,
                center_x - radius_x,
                height * 0.12,
                center_x + radius_x,
                min(height * 0.78, top - 6),
            )
            self.canvas.itemconfigure(
                guide,
                outline=self.COLORS["guidance"],
                state="normal" if pose else "hidden",
            )
            self.canvas.coords(card, left, top, left + card_width, bottom)
            self.canvas.itemconfigure(card, outline=accent)
            self.canvas.itemconfigure(
                arrow,
                fill=self.COLORS["guidance"],
                state="normal" if direction else "hidden",
            )

            title = (
                pose.instruction
                if pose
                else (
                    "Spoof detected"
                    if guidance.feedback == "spoof"
                    else "All references captured"
                )
            )
            hint = {
                "guidance": "",
                "accepted": "Reference accepted",
                "retry": "Try again",
                "spoof": "Select Retry",
            }[guidance.feedback]
            if pose is None and guidance.feedback != "spoof":
                hint = (
                    "Select Save profile"
                    if guidance.feedback == "accepted"
                    else "Check the registration form"
                )
            progress = f"{guidance.captured}/{guidance.total}"
            self.canvas.coords(heading, text_left, top + 16)
            self.canvas.itemconfigure(
                heading, text=title, width=left + card_width - text_left - 10
            )
            self.canvas.coords(detail, text_left, top + 38)
            self.canvas.itemconfigure(
                detail, text=f"{progress} · {hint}" if hint else progress, fill=accent
            )

        # The GUI tick drives the animation. Only the arrow moves, at about 12 FPS.
        if direction is not None:
            step = int(now / self.ANIMATION_STEP)
            if step != self._animation_step:
                self._animation_step = step
                x, y = self._arrow_center
                dx, dy = direction
                shift = 5 * sin(step * self.ANIMATION_STEP * tau)
                self.canvas.coords(
                    arrow,
                    x + dx * (shift - 11),
                    y + dy * (shift - 11),
                    x + dx * (shift + 11),
                    y + dy * (shift + 11),
                )
