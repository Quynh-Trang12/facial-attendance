import unittest
from unittest.mock import Mock

from facial_attendance.ui.registration_overlay import RegistrationOverlay
from facial_attendance.ui.trial_profiles import RegistrationGuidance, TrialProfiles


class RegistrationOverlayTests(unittest.TestCase):
    def setUp(self):
        self.canvas = Mock()
        self.canvas.create_oval.return_value = "guide"
        self.canvas.create_rectangle.return_value = "card"
        self.canvas.create_line.return_value = "arrow"
        self.canvas.create_text.side_effect = ("heading", "detail", "heading", "detail")
        self.overlay = RegistrationOverlay(self.canvas)

    def test_arrows_point_in_the_users_screen_direction(self):
        for index, signs in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1)), start=1):
            with self.subTest(pose=index):
                pose = TrialProfiles.POSES[index]
                self.overlay.update(
                    RegistrationGuidance(pose, index, 5, "guidance"), (640, 360), 0
                )
                item, x1, y1, x2, y2 = self.canvas.coords.call_args.args
                self.assertEqual(item, "arrow")
                self.assertEqual((x2 - x1, y2 - y1), (22 * signs[0], 22 * signs[1]))
                self.canvas.itemconfigure.assert_any_call(
                    "heading", text=pose.instruction, width=321
                )
        self.canvas.tag_raise.assert_called_once_with(
            RegistrationOverlay.TAG, "preview"
        )

    def test_resize_and_animation_move_existing_items_and_cleanup_stops_updates(self):
        guidance = RegistrationGuidance(TrialProfiles.POSES[1], 1, 5, "accepted")
        self.overlay.update(guidance, (640, 360), 0)
        original_arrow = self.canvas.coords.call_args
        self.canvas.coords.reset_mock()
        self.canvas.itemconfigure.reset_mock()
        self.overlay.update(guidance, (640, 360), 0.04)
        self.canvas.coords.assert_not_called()
        self.canvas.itemconfigure.assert_not_called()

        self.overlay.update(guidance, (640, 360), 0.16)
        self.assertNotEqual(self.canvas.coords.call_args, original_arrow)
        self.assertEqual(self.canvas.coords.call_count, 1)
        self.overlay.update(guidance, (840, 460), 0.16)
        self.canvas.coords.assert_any_call("card", 228, 396, 612, 450)
        self.canvas.create_line.assert_called_once()

        self.overlay.update(None, (840, 460), 0.17)
        self.canvas.delete.assert_called_once_with(RegistrationOverlay.TAG)
        self.canvas.coords.reset_mock()
        self.overlay.update(None, (840, 460), 0.32)
        self.canvas.coords.assert_not_called()

    def test_straight_pose_uses_a_dashed_guide_and_spoof_replaces_the_guidance(self):
        self.overlay.update(
            RegistrationGuidance(TrialProfiles.POSES[0], 0, 5, "guidance"),
            (640, 360),
            0,
        )
        self.assertEqual(self.canvas.create_oval.call_args.kwargs["dash"], (5, 6))
        self.canvas.itemconfigure.assert_any_call(
            "guide", outline="#89C7FF", state="normal"
        )
        self.canvas.itemconfigure.assert_any_call(
            "arrow", fill="#89C7FF", state="hidden"
        )
        self.overlay.update(RegistrationGuidance(None, 0, 5, "spoof"), (640, 360), 0.1)
        self.canvas.itemconfigure.assert_any_call(
            "guide", outline="#89C7FF", state="hidden"
        )
        self.canvas.itemconfigure.assert_any_call(
            "detail", text="0/5 · Select Retry", fill="#FFA3A6"
        )
