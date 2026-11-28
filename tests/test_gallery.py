"""Synthetic storage tests; NumPy compatibility cases use the existing dependency."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

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
            person_id, name, [0.6, 0.8, 0.0], n_images=5,
            created_at="2026-11-05T09:00:00+07:00"
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

    def test_explicit_template_update_and_unknown_id(self):
        original = self.register()
        updated = self.gallery.update_template("emp001", [0.0, 0.0, 1.0], n_images=3)
        self.assertEqual(updated.embeddings, ((0.0, 0.0, 1.0),))
        self.assertEqual(updated.n_images, 3)
        self.assertEqual((updated.person_id, updated.name, updated.created_at),
                         (original.person_id, original.name, original.created_at))
        with self.assertRaises(UnknownPersonIDError):
            self.gallery.update_template("MISSING", [1.0, 0.0, 0.0], n_images=3)

    def test_invalid_embedding_does_not_mutate_records(self):
        self.register()
        before = self.gallery.to_dict()
        for bad in (
            [0.1, 0.2], [1.0, 0.0, 0.0, 0.0], [], None, 1.0, "100",
            [float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0],
            [float("-inf"), 0.0, 0.0], [True, 0.0, 0.0], ["1", 0.0, 0.0],
            [1j, 0.0, 0.0], [[1.0], [0.0], [0.0]],
            {1.0, 0.0, -1.0}, {1.0: 0, 0.0: 0, -1.0: 0},
            [0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [1.0001, 0.0, 0.0],
        ):
            with self.subTest(embedding=bad):
                with self.assertRaises(ValueError):
                    self.gallery.update_template("EMP001", bad, n_images=5)
                with self.assertRaises(ValueError):
                    self.gallery.register("EMP002", "B", bad, n_images=5)
        self.assertEqual(self.gallery.to_dict(), before)

    def test_round_trip_without_retraining(self):
        self.register()
        self.gallery.update_template("EMP001", [0.0, -0.6, 0.8], n_images=10)
        self.gallery.save(self.path)
        reloaded = self.load()
        self.assertEqual(reloaded.to_dict(), self.gallery.to_dict())
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
            FaceGallery.load(self.path, model_id="DEMO_MODEL", model_version="DEMO_V1", embedding_dim=128)
        with self.assertRaises(IncompatibleGalleryError):
            FaceGallery.load(self.path, model_id="DEMO_MODEL", model_version="DEMO_V2", embedding_dim=3)

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
        self.register("EMP002", "Another User")
        with patch("facial_attendance.ui.gallery.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(GalleryStorageError):
                self.gallery.save(self.path)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        self.assertIsNone(self.gallery.get("EMP002"))
        # The actual reported bug: a later successful enrollment must not save
        # the previously failed identity as a side effect.
        self.register("EMP003", "Later User")
        self.gallery.save(self.path)
        reloaded = self.load()
        self.assertEqual([person.person_id for person in reloaded.people], ["EMP001", "EMP003"])
        self.assertIsNone(reloaded.get("EMP002"))
        # Explicit recovery is possible by registering the failed ID again.
        self.register("emp002", "Retried User")
        self.gallery.save(self.path)
        self.assertEqual(self.load().get("EMP002").name, "Retried User")

    def test_identity_text_and_timestamp_require_validation(self):
        for invalid in ("", " EMP001", "EMP001 "):
            with self.subTest(person_id=invalid), self.assertRaises(ValueError):
                self.gallery.register(invalid, "A", [1.0, 0.0, 0.0], n_images=5)
        with self.assertRaises(ValueError):
            self.gallery.register("EMP001", "A", [1.0, 0.0, 0.0], n_images=5, created_at="invalid")

    def test_casefold_duplicate_registration_and_lookup_preserve_spelling(self):
        original = self.register()
        with self.assertRaises(DuplicatePersonIDError):
            self.register("emp001", "Other Person")
        self.assertEqual(self.gallery.get("eMp001"), original)
        self.assertEqual(self.gallery.to_dict()["people"][0]["person_id"], "EMP001")
        self.register("Straße", "Other User")
        with self.assertRaises(DuplicatePersonIDError):
            self.register("STRASSE", "Another User")

    def test_case_variant_duplicates_on_disk_rejected_without_changing_file(self):
        self.register()
        payload = self.gallery.to_dict()
        duplicate = dict(payload["people"][0], person_id="emp001")
        payload["people"].append(duplicate)
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaises(GalleryFormatError):
            self.load()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(self.gallery.people), 1)

    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_numpy_arrays_scalars_and_sequences(self):
        vector = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        for embedding in (vector, list(vector), tuple(vector), vector.astype(np.float64),
                          np.array([1, 0, 0], dtype=np.int64)):
            with self.subTest(embedding=embedding):
                gallery = FaceGallery(model_id="M", model_version="V", embedding_dim=3)
                person = gallery.register("A", "A", embedding, n_images=np.int64(3))
                self.assertEqual(person.embeddings[0], tuple(float(x) for x in embedding))
                self.assertTrue(all(type(x) is float for x in person.embeddings[0]))
                gallery.update_template("a", vector, n_images=5)

    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_invalid_numpy_inputs_are_rejected_without_mutation(self):
        self.register()
        before = self.gallery.to_dict()
        for bad in (
            np.array(1.0), np.array([[1.0, 0.0, 0.0]]), np.array([[1.0], [0.0], [0.0]]),
            np.ones((3, 1, 1)), np.ones(2), np.array([True, False, False]),
            [np.bool_(True), 0.0, 0.0], np.array([True, 0.0, 0.0], dtype=object),
            np.array(["1", "0", "0"]), np.array([1j, 0, 0]),
            np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0]),
        ):
            with self.subTest(embedding=bad), self.assertRaises(ValueError):
                self.gallery.update_template("EMP001", bad, n_images=5)
        self.assertEqual(self.gallery.to_dict(), before)

    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_512d_float32_template_round_trip_preserves_values(self):
        vector = np.arange(1, 513, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        gallery = FaceGallery(model_id="run-1", model_version="abcdef012345", embedding_dim=512)
        gallery.register("emp-0001", "Quế An", vector, n_images=5)
        gallery.save(self.path)
        loaded = FaceGallery.load(self.path, model_id="run-1", model_version="abcdef012345", embedding_dim=512)
        person = loaded.get("EMP-0001")
        self.assertEqual(person.embeddings[0], tuple(float(x) for x in vector))
        self.assertEqual(person.n_images, 5)
        with self.assertRaises(ValueError):
            gallery.register("B", "B", vector[:-1], n_images=5)

    def test_image_count_is_required_and_validated_for_registration_and_update(self):
        self.register()
        before = self.gallery.to_dict()
        with self.assertRaises(TypeError):
            self.gallery.register("B", "B", [1.0, 0.0, 0.0])
        for bad in (None, True, False, 2, 11, 3.0, "5"):
            with self.subTest(n_images=bad):
                with self.assertRaises(ValueError):
                    self.gallery.register("B", "B", [1.0, 0.0, 0.0], n_images=bad)
                with self.assertRaises(ValueError):
                    self.gallery.update_template("EMP001", [1.0, 0.0, 0.0], n_images=bad)
        self.assertEqual(self.gallery.to_dict(), before)

    def test_name_limits_and_utc_timestamp(self):
        person = self.register(name="Đặng Thị Dung")
        self.assertEqual(person.created_at, "2026-11-05T02:00:00+00:00")
        self.register("B", "A" * 64)
        for bad in ("", "A" * 65, " A", "A "):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                self.register("C", bad)
        with self.assertRaises(ValueError):
            self.gallery.register("C", "C", [1.0, 0.0, 0.0], n_images=5,
                                  created_at="2026-11-05T09:00:00")

    def test_legacy_file_requires_explicit_migration_and_is_preserved(self):
        self.register()
        payload = self.gallery.to_dict()
        payload["schema_version"] = "0.1"
        del payload["people"][0]["n_images"]
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(GalleryFormatError, "migration or re-enrollment"):
            self.load()
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_append_api_is_rejected_without_mutation(self):
        self.register()
        before = self.gallery.to_dict()
        with self.assertRaisesRegex(ValueError, "update_template"):
            self.gallery.add_embedding("EMP001", [1.0, 0.0, 0.0])
        self.assertEqual(self.gallery.to_dict(), before)

    def test_invalid_template_records_on_disk_are_rejected(self):
        self.register()
        original = self.gallery.to_dict()
        for field, bad in (
            ("embeddings", []), ("embeddings", [[1.0, 0.0, 0.0]] * 2),
            ("embeddings", [[0.1, 0.2, 0.3]]), ("embeddings", [[True, 0.0, 0.0]]),
            ("embeddings", [[float("nan"), 0.0, 0.0]]),
            ("n_images", 2), ("n_images", True), ("n_images", 11),
            ("created_at", None), ("created_at", "2026-11-05T09:00:00"),
            ("name", "A" * 65),
        ):
            payload = deepcopy(original)
            payload["people"][0][field] = bad
            with self.subTest(field=field, value=bad), self.assertRaises(GalleryFormatError):
                FaceGallery.from_dict(payload)
        for field in ("n_images", "created_at"):
            payload = deepcopy(original)
            del payload["people"][0][field]
            with self.subTest(missing=field), self.assertRaises(GalleryFormatError):
                FaceGallery.from_dict(payload)

    def test_norm_tolerance_does_not_silently_normalize(self):
        person = self.gallery.register("A", "A", [0.9995, 0.0, 0.0], n_images=3)
        self.assertEqual(person.embeddings[0], (0.9995, 0.0, 0.0))
        with self.assertRaises(ValueError):
            self.gallery.register("B", "B", [0.998, 0.0, 0.0], n_images=3)

    def test_first_save_failure_discards_pending_registration(self):
        self.register()
        with patch("facial_attendance.ui.gallery.tempfile.mkstemp", side_effect=OSError("full")):
            with self.assertRaises(GalleryStorageError):
                self.gallery.save(self.path)
        self.assertEqual(self.gallery.people, ())
        self.assertFalse(self.path.exists())
        self.gallery.save(self.path)
        self.assertEqual(self.load().people, ())

    def test_save_failures_roll_back_loaded_records_and_updates(self):
        self.register()
        self.gallery.save(self.path)
        before = self.path.read_bytes()
        for target in ("Path.mkdir", "tempfile.mkstemp", "os.fsync", "os.replace", "json.dumps"):
            with self.subTest(stage=target):
                gallery = self.load()
                baseline = gallery.to_dict()
                gallery.update_template("emp001", [1.0, 0.0, 0.0], n_images=3)
                gallery.register("EMP002", "B", [0.0, 1.0, 0.0], n_images=5)
                error = ValueError("serialization failure") if target == "json.dumps" else OSError("disk failure")
                with patch("facial_attendance.ui.gallery." + target, side_effect=error):
                    with self.assertRaisesRegex(GalleryStorageError, "rolled back"):
                        gallery.save(self.path)
                self.assertEqual(gallery.to_dict(), baseline)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
                gallery.save(self.path)
                self.assertEqual(self.load().to_dict(), baseline)

    def test_from_dict_establishes_validated_rollback_baseline(self):
        self.register()
        baseline = self.gallery.to_dict()
        gallery = FaceGallery.from_dict(baseline)
        gallery.register("B", "B", [1.0, 0.0, 0.0], n_images=3)
        with patch("facial_attendance.ui.gallery.os.replace", side_effect=OSError("failed")):
            with self.assertRaises(GalleryStorageError):
                gallery.save(self.path)
        self.assertEqual(gallery.to_dict(), baseline)


if __name__ == "__main__":
    unittest.main()
