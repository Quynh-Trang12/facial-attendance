"""Generated-image fixtures test loader behavior; they are not Kaggle data."""

from collections import Counter
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from facial_attendance.recognition.data_loader import (
    DEFAULT_EXCLUSIONS,
    ClassificationFaces,
    FaceLayout,
    PairFaces,
    VerificationPairs,
    sample_training_pairs,
)
from facial_attendance.recognition.download import COMPETITION, download_face_data


class FaceDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="face-loader-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for split in ("train_data", "val_data", "test_data"):
            for identity in ("10", "2"):
                for index in range(3 if split == "train_data" else 1):
                    path = (
                        self.root
                        / "classification_data"
                        / split
                        / identity
                        / f"{index}.PNG"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (9, 7), color=50 + index).save(path)
        verification = self.root / "verification_data"
        verification.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            Image.new("RGB", (7, 9), color=(20, 40, 60)).save(verification / name)
        self.pairs_file = self.root / "verification_pairs_val.txt"
        self.pairs_file.write_text(
            "verification_data/a.png verification_data/b.png 1\na.png c.png 0\n"
        )
        self.layout = FaceLayout.discover(self.root, exclusions=None)

    def test_discovers_enclosing_directory(self):
        wrapped = self.root / "wrapped"
        wrapped.mkdir()
        for name in [
            "classification_data",
            "verification_data",
            "verification_pairs_val.txt",
        ]:
            (self.root / name).rename(wrapped / name)
        self.assertEqual(FaceLayout.discover(self.root, exclusions=None).root, wrapped.resolve())

    def test_rejects_incomplete_layout(self):
        self.pairs_file.unlink()
        with self.assertRaises(ValueError):
            FaceLayout.discover(self.root)

    def test_classification_converts_grayscale_to_rgb_and_maps_ids(self):
        dataset = ClassificationFaces(self.layout, "train_data")
        self.assertEqual(dataset.class_to_idx, {"10": 0, "2": 1})
        self.assertEqual(len(dataset), 6)
        image, target = dataset[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (9, 7))
        self.assertEqual(target, 0)

    def test_validation_reuses_training_mapping(self):
        train = ClassificationFaces(self.layout, "train_data")
        val = ClassificationFaces(
            self.layout, "val_data", class_to_idx=train.class_to_idx
        )
        self.assertEqual([sample.target for sample in val.samples], [0, 1])
        with self.assertRaises(ValueError):
            ClassificationFaces(self.layout, "val_data")

    def test_boolean_targets_are_not_accepted_as_integer_class_ids(self):
        with self.assertRaisesRegex(ValueError, "integer targets"):
            ClassificationFaces(
                self.layout, "train_data", class_to_idx={"10": False, "2": True}
            )

    def test_unknown_validation_identity_is_rejected(self):
        train = ClassificationFaces(self.layout, "train_data")
        (self.root / "classification_data/val_data/2").rename(
            self.root / "classification_data/val_data/new"
        )
        with self.assertRaises(ValueError):
            ClassificationFaces(
                self.layout, "val_data", class_to_idx=train.class_to_idx
            )

    def test_empty_identity_is_rejected(self):
        (self.root / "classification_data/train_data/empty").mkdir()
        with self.assertRaises(ValueError):
            ClassificationFaces(self.layout, "train_data")

    def test_pairs_preserve_order_labels_and_both_path_styles(self):
        dataset = VerificationPairs(self.layout)
        self.assertEqual([p.label for p in dataset.pairs], [1, 0])
        left, right, label = dataset[0]
        self.assertEqual((left.mode, right.mode, label), ("RGB", "RGB", 1))

    def test_missing_pair_image_is_rejected(self):
        self.pairs_file.write_text("missing.png b.png 1\n")
        with self.assertRaisesRegex(ValueError, "line 1"):
            VerificationPairs(self.layout)

    def test_invalid_pair_label_is_rejected(self):
        self.pairs_file.write_text("a.png b.png 2\n")
        with self.assertRaisesRegex(ValueError, "label 0/1"):
            VerificationPairs(self.layout)

    def test_wrong_column_count_is_rejected(self):
        self.pairs_file.write_text("a.png b.png\n")
        with self.assertRaises(ValueError):
            VerificationPairs(self.layout)

    def test_path_traversal_is_rejected(self):
        self.pairs_file.write_text(
            "../classification_data/train_data/10/0.PNG b.png 1\n"
        )
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            VerificationPairs(self.layout)

    def test_absolute_pair_path_is_rejected(self):
        self.pairs_file.write_text(f"{self.root}/verification_data/a.png b.png 1\n")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            VerificationPairs(self.layout)

    def test_corrupt_image_fails_at_decode(self):
        (self.root / "verification_data/a.png").write_text("not an image")
        dataset = VerificationPairs(self.layout)
        with self.assertRaisesRegex(ValueError, "Cannot decode"):
            dataset[0]

    def test_sampling_is_balanced_reproducible_and_identity_correct(self):
        train = ClassificationFaces(self.layout, "train_data")
        pairs = sample_training_pairs(train, count_per_label=10, seed=42)
        self.assertEqual(
            pairs, sample_training_pairs(train, count_per_label=10, seed=42)
        )
        self.assertEqual(Counter(p.label for p in pairs), {0: 10, 1: 10})
        identities = {sample.path: sample.identity for sample in train.samples}
        for pair in pairs:
            self.assertNotEqual(pair.left, pair.right)
            self.assertEqual(
                identities[pair.left] == identities[pair.right], bool(pair.label)
            )

    def test_sampling_does_not_use_validation(self):
        train = ClassificationFaces(self.layout, "train_data")
        val = ClassificationFaces(
            self.layout, "val_data", class_to_idx=train.class_to_idx
        )
        with self.assertRaises(ValueError):
            sample_training_pairs(val, count_per_label=1)

    def test_sampled_pairs_load_actual_images(self):
        train = ClassificationFaces(self.layout, "train_data")
        records = sample_training_pairs(train, count_per_label=3, seed=42)
        dataset = PairFaces(records)
        self.assertEqual(len(dataset), 6)
        left, right, label = dataset[0]
        self.assertEqual((left.mode, right.mode), ("RGB", "RGB"))
        self.assertEqual(label, records[0].label)

    def exclude(self, *relative) -> FaceLayout:
        exclusions = self.root / "exclusions.tsv"
        exclusions.write_text(
            "# comment\n" + "".join(f"{path}\tverification_data\n" for path in relative)
        )
        return FaceLayout.discover(self.root, exclusions=exclusions)

    def test_listed_images_are_not_loaded_and_targets_do_not_change(self):
        layout = self.exclude("classification_data/train_data/10/0.PNG")
        full = ClassificationFaces(self.layout, "train_data")
        train = ClassificationFaces(layout, "train_data")
        self.assertEqual((len(full), len(train), train.excluded_count), (6, 5, 1))
        self.assertEqual(train.class_to_idx, full.class_to_idx)
        self.assertNotIn(
            (self.root / "classification_data/train_data/10/0.PNG").resolve(),
            {sample.path for sample in train.samples},
        )

    def test_sampled_pairs_never_use_a_listed_image(self):
        relative = "classification_data/train_data/10/0.PNG"
        train = ClassificationFaces(self.exclude(relative), "train_data")
        pairs = sample_training_pairs(train, count_per_label=200, seed=42)
        drawn = {path for pair in pairs for path in (pair.left, pair.right)}
        self.assertNotIn((self.root / relative).resolve(), drawn)

    def test_held_out_identity_may_lose_its_images(self):
        layout = self.exclude("classification_data/val_data/10/0.PNG")
        train = ClassificationFaces(layout, "train_data")
        val = ClassificationFaces(layout, "val_data", class_to_idx=train.class_to_idx)
        self.assertEqual((len(val), val.excluded_count), (1, 1))
        self.assertEqual(val.class_to_idx, train.class_to_idx)

    def test_training_identity_may_not_lose_all_images(self):
        layout = self.exclude(
            *(f"classification_data/train_data/2/{i}.PNG" for i in range(3))
        )
        with self.assertRaisesRegex(ValueError, "empty training identities"):
            ClassificationFaces(layout, "train_data")

    def test_list_must_match_this_dataset(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.exclude("classification_data/train_data/10/missing.PNG")

    def test_official_trial_images_can_not_be_listed(self):
        with self.assertRaisesRegex(ValueError, "not a classification image"):
            self.exclude("verification_data/a.png")

    def test_loading_defaults_to_the_committed_list(self):
        self.assertEqual(
            inspect.signature(FaceLayout.discover).parameters["exclusions"].default,
            DEFAULT_EXCLUSIONS,
        )
        self.assertEqual(
            inspect.signature(download_face_data).parameters["exclusions"].default,
            DEFAULT_EXCLUSIONS,
        )

    def test_empty_pair_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            PairFaces(())

    def test_download_calls_assigned_competition_and_restores_cache_environment(self):
        api = Mock(return_value=str(self.root))
        with patch.dict(
            "sys.modules", {"kagglehub": SimpleNamespace(competition_download=api)}
        ):
            with patch.dict(os.environ, {"KAGGLEHUB_CACHE": "existing-cache"}):
                layout = download_face_data(
                    cache_dir=self.root / "cache", exclusions=None
                )
                self.assertEqual(layout.root, self.root.resolve())
                self.assertEqual(os.environ["KAGGLEHUB_CACHE"], "existing-cache")
        api.assert_called_once_with(COMPETITION)

    def test_download_failure_preserves_original_error(self):
        api = Mock(side_effect=RuntimeError("authentication required"))
        with patch.dict(
            "sys.modules", {"kagglehub": SimpleNamespace(competition_download=api)}
        ):
            with self.assertRaisesRegex(RuntimeError, "authentication required"):
                download_face_data(cache_dir=self.root / "cache")


if __name__ == "__main__":
    unittest.main()
