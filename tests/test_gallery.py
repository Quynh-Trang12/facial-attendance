"""Isolated gallery tests: synthetic vectors, no webcam or ML dependencies."""

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

    def register(self, person_id="EMP001", name="Example User"):
        return self.gallery.register(
            person_id, name, [0.1, 0.2, -0.3], created_at="2026-11-05T09:00:00+07:00"
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

    def test_equal_names_with_distinct_ids_are_allowed(self):
        self.register()
        self.register("EMP002", "Example User")
        self.assertEqual(len(self.gallery.people), 2)

    def test_explicit_embedding_update_and_unknown_id(self):
        self.register()
        self.gallery.add_embedding("EMP001", [-0.5, 0.0, 0.8])
        self.assertEqual(len(self.gallery.get("EMP001").embeddings), 2)
        with self.assertRaises(UnknownPersonIDError):
            self.gallery.add_embedding("MISSING", [0.1, 0.2, 0.3])

    def test_invalid_embedding_does_not_mutate_records(self):
        self.register()
        before = self.gallery.to_dict()
        for bad in ([0.1, 0.2], [float("nan"), 0.2, 0.3], [True, 0.2, 0.3], ["0.1", 0.2, 0.3]):
            with self.subTest(embedding=bad), self.assertRaises(ValueError):
                self.gallery.add_embedding("EMP001", bad)
        self.assertEqual(self.gallery.to_dict(), before)

    def test_round_trip_without_retraining(self):
        self.register()
        self.gallery.add_embedding("EMP001", [0.2, -0.4, 0.6])
        self.gallery.save(self.path)
        reloaded = self.load()
        self.assertEqual(reloaded.to_dict(), self.gallery.to_dict())
        reloaded.register("EMP002", "Another User", [0.0, 0.3, 0.2])
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
            FaceGallery.load(self.path, model_id="DEMO_MODEL", model_version="DEMO_V1", embedding_dim=128)

    def test_corruption_and_duplicate_ids_on_disk_are_rejected(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{invalid json", encoding="utf-8")
        with self.assertRaises(GalleryFormatError):
            self.load()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{invalid json")
        self.register()
        duplicate = self.gallery.to_dict()
        duplicate["people"].append(dict(duplicate["people"][0]))
        self.path.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaises(GalleryFormatError):
            self.load()

    def test_invalid_or_unknown_schema_is_rejected(self):
        self.register()
        payload = self.gallery.to_dict()
        payload["schema_version"] = "99.0"
        with self.assertRaises(GalleryFormatError):
            FaceGallery.from_dict(payload)
        self.assertEqual(len(self.gallery.people), 1)

    def test_atomic_replace_failure_preserves_original_file(self):
        self.register()
        self.gallery.save(self.path)
        original_bytes = self.path.read_bytes()
        self.gallery.register("EMP002", "Another User", [0.0, 0.1, 0.2])
        with patch("facial_attendance.ui.gallery.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(GalleryStorageError):
                self.gallery.save(self.path)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_identity_text_and_timestamp_require_validation(self):
        for invalid in ("", " EMP001", "EMP001 "):
            with self.subTest(person_id=invalid), self.assertRaises(ValueError):
                self.gallery.register(invalid, "A", [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            self.gallery.register("EMP001", "A", [0.1, 0.2, 0.3], created_at="invalid")


if __name__ == "__main__":
    unittest.main()
