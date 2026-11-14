"""Seeded val_data pair list on generated images; they are not Kaggle data."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from facial_attendance.recognition.data_loader import ClassificationFaces, FaceLayout
from facial_attendance.recognition.validation_pairs import (
    build_validation_pairs,
    read_pairs,
    write_pairs,
)


def build_layout(root: Path, val_counts: dict[str, int]) -> FaceLayout:
    for identity, count in val_counts.items():
        for split, images in (("train_data", 3), ("val_data", count), ("test_data", 1)):
            for index in range(images):
                path = root / "classification_data" / split / identity / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(index * 40, len(identity), 0)).save(path)
    (root / "verification_data").mkdir()
    Image.new("RGB", (8, 8)).save(root / "verification_data" / "a.png")
    (root / "verification_pairs_val.txt").write_text("a.png a.png 1\n")
    return FaceLayout.discover(root)


class ValidationPairTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="val-pairs-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Like the filtered val_data: most identities keep two images, some only one.
        counts = {f"n{i:03d}": 1 if i % 4 == 0 else 2 for i in range(20)}
        self.layout = build_layout(self.root, counts)
        train = ClassificationFaces(self.layout, "train_data")
        self.val = ClassificationFaces(self.layout, "val_data", class_to_idx=train.class_to_idx)
        self.train = train

    def identity(self, path: Path) -> str:
        return path.parent.name

    def test_one_positive_per_two_image_identity_and_as_many_negatives(self):
        pairs = build_validation_pairs(self.val)
        positives = [p for p in pairs if p.label == 1]
        negatives = [p for p in pairs if p.label == 0]
        self.assertEqual(len(positives), 15)
        self.assertEqual(len(negatives), 15)
        self.assertEqual(len({self.identity(p.left) for p in positives}), 15)
        for pair in positives:
            self.assertEqual(self.identity(pair.left), self.identity(pair.right))
            self.assertNotEqual(pair.left, pair.right)
        for pair in negatives:
            self.assertNotEqual(self.identity(pair.left), self.identity(pair.right))
        self.assertEqual(len({(p.left, p.right) for p in negatives}), 15)

    def test_seeded_and_val_only(self):
        self.assertEqual(build_validation_pairs(self.val), build_validation_pairs(self.val))
        self.assertNotEqual(build_validation_pairs(self.val), build_validation_pairs(self.val, seed=7))
        with self.assertRaises(ValueError):
            build_validation_pairs(self.train)

    def test_file_round_trip_uses_root_relative_paths_and_stable_hash(self):
        pairs = build_validation_pairs(self.val)
        file = self.root / "out" / "val-pairs.txt"
        digest = write_pairs(file, self.layout, pairs)
        self.assertTrue(file.read_text().startswith("classification_data/val_data/"))
        self.assertEqual(read_pairs(file, self.layout), pairs)
        self.assertEqual(write_pairs(self.root / "again.txt", self.layout, pairs), digest)


if __name__ == "__main__":
    unittest.main()
