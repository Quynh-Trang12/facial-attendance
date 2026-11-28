"""Protect source-bound crop diagnostics and their non-destructive boundary."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facial_attendance.liveness.audit.crop_content_scan import (
    inspect_crop,
    reviewed_crop_disposition,
    safe_input,
    supplied_crop,
)
from facial_attendance.liveness.datasets.source_dataset import read_reproduction_face


class SourceCropAuditTests(unittest.TestCase):
    def fixture(self, root, box="20 20 100 100 0.99"):
        image = root / "sample.png"
        Image.new("RGB", (224, 224), (20, 50, 100)).save(image)
        bbox = root / "sample_BB.txt"
        bbox.write_text(box)
        record = {
            "relative_path": image.name,
            "audit_split": "train",
            "binary_label": "0",
            "source_type": "0",
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "bbox_sha256": hashlib.sha256(bbox.read_bytes()).hexdigest(),
        }
        return image, bbox, record

    def test_geometry_matches_the_existing_reader_including_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for text in (
                "1.9 2.9 3.9 4.9 0.99",
                "12 -37 199 241 0.99",
                "78  52 78 84 0.99",
                "",
                "161 16 16 21 0.99",
            ):
                with self.subTest(box=text):
                    image, bbox, _ = self.fixture(root, text)
                    box, _ = supplied_crop(text, (224, 224))
                    with Image.open(image) as original:
                        self.assertEqual(
                            original.crop(box).tobytes(),
                            read_reproduction_face(image, bbox).tobytes(),
                        )

    def test_wrong_subject_is_flagged_while_matching_box_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, bbox, record = self.fixture(root, "170 10 15 20 0.99")
            before = (image.read_bytes(), bbox.read_bytes(), record.copy())
            detector = root / "detector.onnx"
            with patch(
                "facial_attendance.liveness.audit.crop_content_scan.detector_boxes",
                return_value=[(20, 20, 140, 180)],
            ):
                result = inspect_crop(root, record, detector_model=detector)
            self.assertIsNone(result["error"])
            self.assertIn("detector_no_overlap", result["review_flags"])
            self.assertIn("larger_disagreeing_face", result["review_flags"])
            self.assertEqual(result["size_band"], "lt_16")
            self.assertEqual((image.read_bytes(), bbox.read_bytes(), record), before)
            with patch(
                "facial_attendance.liveness.audit.crop_content_scan.detector_boxes",
                return_value=[(170, 10, 185, 30)],
            ):
                counterpart = inspect_crop(root, record, detector_model=detector)
            self.assertNotIn("detector_no_overlap", counterpart["review_flags"])
            self.assertEqual(counterpart["best_detector_iou"], 1)
            # Tiny size is preserved as a diagnostic, never silently accepted,
            # excluded, relabelled or enlarged into new source evidence.
            self.assertEqual(counterpart["native_min_edge"], 15)
            with patch(
                "facial_attendance.liveness.audit.crop_content_scan.detector_boxes",
                return_value=[(170, 10, 185, 30), (20, 20, 140, 180)],
            ):
                competing = inspect_crop(root, record, detector_model=detector)
            self.assertEqual(competing["best_detector_iou"], 1)
            self.assertIn("larger_disagreeing_face", competing["review_flags"])
            self.assertEqual(competing["proposed_disposition"], "review_multiple_faces")
            self.assertEqual(
                counterpart["proposed_disposition"], "review_single_face_crop"
            )

    def test_face_counts_only_request_review_for_both_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, bbox, record = self.fixture(root)
            source_bytes = (image.read_bytes(), bbox.read_bytes())
            cases = (
                ([], "review_no_detection"),
                ([(20, 20, 120, 120)], "review_single_face_crop"),
                ([(20, 20, 120, 120), (150, 20, 200, 80)], "review_multiple_faces"),
            )
            for label in ("0", "1"):
                for boxes, expected in cases:
                    with self.subTest(label=label, faces=len(boxes)):
                        record["binary_label"] = label
                        record["source_type"] = "0" if label == "0" else "1"
                        before = record.copy()
                        with patch(
                            "facial_attendance.liveness.audit.crop_content_scan.detector_boxes",
                            return_value=boxes,
                        ):
                            result = inspect_crop(
                                root, record, detector_model=root / "detector.onnx"
                            )
                        self.assertEqual(result["proposed_disposition"], expected)
                        self.assertEqual(result["detector_candidates"], len(boxes))
                        self.assertNotIn("quarantine", result["proposed_disposition"])
                        self.assertEqual(record, before)
                        self.assertEqual(
                            (image.read_bytes(), bbox.read_bytes()), source_bytes
                        )

    def test_confirmed_wrong_crop_needs_no_detector_confirmation(self):
        review = {
            "decision": "confirmed_wrong_target",
            "finding": "Supplied crop selects a background picture",
            "review_method": "Source image/crop visual inspection",
            "evidence": "page-01.png",
        }
        for count in (0, 1, 3):
            for label in (0, 1):
                candidate = {
                    **review,
                    "detector_candidates": count,
                    "binary_label": label,
                }
                self.assertEqual(reviewed_crop_disposition(candidate), "quarantine")
        with self.assertRaisesRegex(ValueError, "review evidence"):
            reviewed_crop_disposition({"decision": "confirmed_wrong_roi"})

    def test_valid_and_unresolved_crops_remain_distinct(self):
        self.assertEqual(
            reviewed_crop_disposition({"decision": "partial_roi_target_unresolved"}),
            "review",
        )
        self.assertEqual(
            reviewed_crop_disposition({"decision": "confirmed_valid_roi"}), "review"
        )
        accepted = {
            "decision": "confirmed_valid_usable_roi",
            "finding": "Correct target and usable crop",
            "review_method": "Content review",
            "evidence": "page-02.png",
            "detector_candidates": 0,
        }
        self.assertEqual(reviewed_crop_disposition(accepted), "retain")
        rejected = {**accepted, "decision": "confirmed_unusable_crop"}
        self.assertEqual(reviewed_crop_disposition(rejected), "quarantine")

    def test_source_changes_and_invalid_geometry_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, bbox, record = self.fixture(root)
            image.write_bytes(b"changed")
            result = inspect_crop(root, record)
            self.assertIn("checksum mismatch", result["error"])
            _, _, record = self.fixture(root, "0 0 -1 20 0.99")
            self.assertIn("invalid supplied crop", inspect_crop(root, record)["error"])

    def test_escape_and_symlink_aliases_do_not_change_input_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside.png"
            outside.touch()
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_input(root, "../outside.png")
            try:
                (root / "alias.png").symlink_to(outside)
            except OSError as error:
                self.skipTest(f"Symlink creation unavailable: {error}")
            for name in ("../outside.png", "alias.png"):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    safe_input(root, name)


if __name__ == "__main__":
    unittest.main()
