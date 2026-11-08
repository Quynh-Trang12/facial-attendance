"""Unit tests for the shared UI-01.2 / UI-01.4 gallery; no webcam required."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facial_attendance.ui.gallery import (
    DuplicatePersonIDError,
    FaceGallery,
    GalleryFormatError,
    GalleryStorageError,
    IncompatibleGalleryError,
    UnknownPersonIDError,
)


class FaceGalleryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "artifacts" / "gallery" / "gallery.json"
        self.gallery = FaceGallery(model_id="DEMO_MODEL", model_version="DEMO_V1", embedding_dim=3)

    def register(self, person_id="EMP001", name="Example User", embedding=None, *, n_images=3):
        return self.gallery.register(
            person_id, name, [1.0, 0.0, 0.0] if embedding is None else embedding,
            n_images=n_images, created_at="2026-11-05T09:00:00+07:00",
        )

    def load(self):
        return FaceGallery.load(
            self.path, model_id="DEMO_MODEL", model_version="DEMO_V1", embedding_dim=3
        )

    def test_new_identity_and_duplicate_id(self):
        original = self.register()
        with self.assertRaises(DuplicatePersonIDError):
            self.register(name="Another Person")
        self.assertEqual(self.gallery.get("EMP001"), original)

    def test_casefold_duplicate_id_is_rejected_and_lookup_is_consistent(self):
        original = self.register()
        self.assertEqual(self.gallery.get("emp001"), original)
        with self.assertRaises(DuplicatePersonIDError):
            self.register("emp001", "Different Person")
        self.assertEqual(len(self.gallery.people), 1)

    def test_equal_names_with_distinct_ids_are_allowed(self):
        self.register()
        self.register("EMP002", "Example User", [0.0, 1.0, 0.0])
        self.assertEqual(len(self.gallery.people), 2)

    def test_explicit_template_update_and_unknown_id(self):
        self.register()
        updated = self.gallery.update_template("emp001", [0.0, 1.0, 0.0], n_images=5)
        self.assertEqual(updated.embeddings, ((0.0, 1.0, 0.0),))
        self.assertEqual(updated.n_images, 5)
        with self.assertRaises(UnknownPersonIDError):
            self.gallery.update_template("MISSING", [1.0, 0.0, 0.0], n_images=3)

    def test_invalid_embedding_does_not_mutate_records(self):
        self.register()
        before = self.gallery.to_dict()
        for bad in ([1.0, 0.0], [float("nan"), 0.0, 0.0], [True, 0.0, 0.0],
                    ["1.0", 0.0, 0.0], [0.2, 0.3, 0.4], [[1.0, 0.0, 0.0]]):
            with self.subTest(embedding=bad), self.assertRaises(ValueError):
                self.gallery.update_template("EMP001", bad, n_images=3)
        self.assertEqual(self.gallery.to_dict(), before)

    def test_numpy_float32_vectors_are_accepted(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed in the test environment")
        identity = self.register(embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(identity.embeddings, ((1.0, 0.0, 0.0),))
        self.gallery.update_template("EMP001", [np.float32(0.0), np.float32(1.0), np.float32(0.0)], n_images=4)
        self.assertEqual(self.gallery.get("emp001").n_images, 4)

    def test_round_trip_and_averaged_template_count(self):
        self.register(n_images=5)
        self.gallery.save(self.path)
        reloaded = self.load()
        self.assertEqual(reloaded.to_dict(), self.gallery.to_dict())
        self.assertEqual(reloaded.to_dict()["people"][0]["n_images"], 5)
        self.assertEqual(len(reloaded.get("EMP001").embeddings), 1)
        reloaded.register("EMP002", "Another User", [0.0, 1.0, 0.0], n_images=3)
        reloaded.save(self.path)
        self.assertEqual(len(self.load().people), 2)

    def test_missing_file_is_empty_gallery_and_does_not_create_directory(self):
        loaded = self.load()
        self.assertEqual(loaded.people, ())
        self.assertFalse(self.path.parent.exists())

    def test_mismatched_model_or_dimensions_are_rejected(self):
        self.register()
        self.gallery.save(self.path)
        with self.assertRaises(IncompatibleGalleryError):
            FaceGallery.load(self.path, model_id="DIFFERENT", model_version="DEMO_V1", embedding_dim=3)
        with self.assertRaises(IncompatibleGalleryError):
            FaceGallery.load(self.path, model_id="DEMO_MODEL", model_version="DIFFERENT", embedding_dim=3)
        with self.assertRaises(IncompatibleGalleryError):
            FaceGallery.load(self.path, model_id="DEMO_MODEL", model_version="DEMO_V1", embedding_dim=512)

    def test_corruption_and_case_variant_duplicate_ids_on_disk_are_rejected(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{invalid json", encoding="utf-8")
        with self.assertRaises(GalleryFormatError):
            self.load()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{invalid json")
        self.register()
        duplicate = self.gallery.to_dict()
        other = dict(duplicate["people"][0])
        other["person_id"] = "emp001"
        duplicate["people"].append(other)
        self.path.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaises(GalleryFormatError):
            self.load()

    def test_old_or_invalid_schema_is_rejected_without_migration(self):
        self.register()
        payload = self.gallery.to_dict()
        payload["schema_version"] = "0.1"
        with self.assertRaises(GalleryFormatError):
            FaceGallery.from_dict(payload)
        self.assertEqual(len(self.gallery.people), 1)

    def test_multiple_raw_embeddings_on_disk_are_rejected(self):
        self.register()
        payload = self.gallery.to_dict()
        payload["people"][0]["embeddings"].append([0.0, 1.0, 0.0])
        with self.assertRaises(GalleryFormatError):
            FaceGallery.from_dict(payload)

    def test_atomic_replace_failure_restores_file_and_in_memory_snapshot(self):
        self.register()
        self.gallery.save(self.path)
        original_bytes = self.path.read_bytes()
        self.register("EMP002", "Another User", [0.0, 1.0, 0.0])
        with patch("facial_attendance.ui.gallery.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(GalleryStorageError):
                self.gallery.save(self.path)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertIsNone(self.gallery.get("EMP002"))
        self.gallery.save(self.path)
        self.assertEqual(len(self.load().people), 1)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_first_save_failure_rolls_back_all_unsaved_records(self):
        self.register()
        with patch("facial_attendance.ui.gallery.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(GalleryStorageError):
                self.gallery.save(self.path)
        self.assertEqual(self.gallery.people, ())
        self.assertFalse(self.path.exists())

    def test_identity_text_image_count_and_timestamp_require_validation(self):
        for invalid in ("", " EMP001", "EMP001 "):
            with self.subTest(person_id=invalid), self.assertRaises(ValueError):
                self.gallery.register(invalid, "A", [1.0, 0.0, 0.0], n_images=3)
        for invalid in (0, 2, 11, 3.0, True):
            with self.subTest(n_images=invalid), self.assertRaises(ValueError):
                self.gallery.register("EMP001", "A", [1.0, 0.0, 0.0], n_images=invalid)
        with self.assertRaises(ValueError):
            self.gallery.register("EMP001", "A", [1.0, 0.0, 0.0], n_images=3, created_at="invalid")
        with self.assertRaises(ValueError):
            self.gallery.register("EMP001", "A", [1.0, 0.0, 0.0], n_images=3, created_at="2026-11-05T09:00:00")

    def test_timestamp_is_saved_in_utc(self):
        self.register()
        self.assertEqual(self.gallery.get("EMP001").created_at, "2026-11-05T02:00:00Z")


if __name__ == "__main__":
    unittest.main()
