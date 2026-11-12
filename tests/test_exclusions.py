"""Applying an exclusion list, on generated-image fixtures; they are not Kaggle data."""

from hashlib import sha256
import inspect
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from facial_attendance.recognition.data_loader import (
    ClassificationFaces,
    FaceLayout,
    sample_training_pairs,
)
from facial_attendance.recognition.exclusions import (
    COLUMNS,
    DEFAULT_EXCLUSIONS,
    load_face_layout,
    write_exclusions,
)

KEPT = "verification_data/a.png"


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="face-exclusion-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for split in ("train_data", "val_data", "test_data"):
            for identity in ("10", "2"):
                for index in range(3 if split == "train_data" else 1):
                    path = self.root / "classification_data" / split / identity / f"{index}.PNG"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (9, 7), color=50 + index).save(path)
        verification = self.root / "verification_data"
        verification.mkdir()
        Image.new("RGB", (7, 9)).save(verification / "a.png")
        (self.root / "verification_pairs_val.txt").write_text("a.png a.png 1\n")
        self.file = self.root / "exclusions.tsv"

    def entry(self, relative: str, kept: str = KEPT) -> tuple[str, str, str, str]:
        digest = sha256((self.root / relative).read_bytes()).hexdigest()
        return relative, "verification_data", digest, kept

    def exclude(self, *entries) -> FaceLayout:
        write_exclusions(self.file, list(entries), ["fixture list"])
        return load_face_layout(self.root, exclusions=self.file)

    def test_discover_is_unchanged_and_loads_the_published_data(self):
        self.assertEqual(FaceLayout.discover(self.root).excluded, frozenset())
        self.assertEqual(load_face_layout(self.root, exclusions=None).excluded, frozenset())

    def test_listed_images_are_not_loaded_and_targets_do_not_change(self):
        layout = self.exclude(self.entry("classification_data/train_data/10/0.PNG"))
        full = ClassificationFaces(FaceLayout.discover(self.root), "train_data")
        train = ClassificationFaces(layout, "train_data")
        self.assertEqual((len(full), len(train), train.excluded_count), (6, 5, 1))
        self.assertEqual(train.class_to_idx, full.class_to_idx)
        self.assertNotIn(
            (self.root / "classification_data/train_data/10/0.PNG").resolve(),
            {sample.path for sample in train.samples},
        )

    def test_sampled_pairs_never_use_a_listed_image(self):
        relative = "classification_data/train_data/10/0.PNG"
        train = ClassificationFaces(self.exclude(self.entry(relative)), "train_data")
        pairs = sample_training_pairs(train, count_per_label=200, seed=42)
        drawn = {path for pair in pairs for path in (pair.left, pair.right)}
        self.assertNotIn((self.root / relative).resolve(), drawn)

    def test_held_out_identity_may_lose_its_images(self):
        layout = self.exclude(self.entry("classification_data/val_data/10/0.PNG"))
        train = ClassificationFaces(layout, "train_data")
        val = ClassificationFaces(layout, "val_data", class_to_idx=train.class_to_idx)
        self.assertEqual((len(val), val.excluded_count), (1, 1))
        self.assertEqual(val.class_to_idx, train.class_to_idx)

    def test_training_identity_may_not_lose_all_images(self):
        layout = self.exclude(
            *(self.entry(f"classification_data/train_data/2/{i}.PNG") for i in range(3))
        )
        with self.assertRaisesRegex(ValueError, "empty training identities"):
            ClassificationFaces(layout, "train_data")

    def test_list_must_match_this_dataset(self):
        missing = ("classification_data/train_data/10/missing.PNG", "x", "0" * 64, KEPT)
        changed = self.entry("classification_data/train_data/10/0.PNG")[:2] + ("0" * 64, KEPT)
        no_kept = self.entry("classification_data/train_data/10/0.PNG", "verification_data/z.png")
        trial = ("verification_data/a.png", "x", "0" * 64, KEPT)
        for entry, message in (
            (missing, "not a classification image"),
            (changed, "listed content hash"),
            (no_kept, "kept copy"),
            (trial, "not a classification image"),
        ):
            self.file.unlink(missing_ok=True)
            with self.subTest(message), self.assertRaisesRegex(ValueError, f"line 3: .*{message}"):
                self.exclude(entry)

    def test_rows_need_every_column(self):
        self.file.write_text("classification_data/train_data/10/0.PNG\tverification_data\n")
        with self.assertRaisesRegex(ValueError, "line 1: expected 4"):
            load_face_layout(self.root, exclusions=self.file)

    def test_a_written_list_is_never_overwritten(self):
        self.exclude(self.entry("classification_data/train_data/10/0.PNG"))
        with self.assertRaises(FileExistsError):
            write_exclusions(self.file, [], ["again"])
        self.assertEqual(COLUMNS, ("path", "reason", "sha256", "kept_path"))

    def test_loading_defaults_to_the_committed_list(self):
        default = inspect.signature(load_face_layout).parameters["exclusions"].default
        self.assertEqual(default, DEFAULT_EXCLUSIONS)
        self.assertEqual(DEFAULT_EXCLUSIONS.parent.name, "configs")


if __name__ == "__main__":
    unittest.main()
