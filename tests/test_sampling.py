"""P×K triplet batches on generated-image fixtures; they are not Kaggle data."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from facial_attendance.recognition.data_loader import ClassificationFaces, FaceLayout
from facial_attendance.recognition.sampling import (
    IDENTITIES_PER_BATCH,
    IMAGES_PER_IDENTITY,
    IdentityBatchSampler,
)


def build_dataset(root: Path, images: dict[str, int]):
    for split in ("train_data", "val_data", "test_data"):
        for identity, count in images.items():
            for index in range(count if split == "train_data" else 1):
                path = root / "classification_data" / split / identity / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), color=(index, len(identity), 0)).save(path)
    (root / "verification_data").mkdir()
    Image.new("RGB", (4, 4)).save(root / "verification_data" / "a.png")
    (root / "verification_pairs_val.txt").write_text("a.png a.png 1\n")
    return FaceLayout.discover(root)


class IdentityBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="pk-sampler-fixture-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Uneven identity sizes, as in the real training split.
        self.layout = build_dataset(
            self.root, {f"n{i:03d}": 4 + i % 5 for i in range(12)}
        )
        self.train = ClassificationFaces(self.layout, "train_data")

    def sampler(self, **options):
        options = {"identities_per_batch": 5, "images_per_identity": 3} | options
        return IdentityBatchSampler(self.train, **options)

    def test_defaults_follow_the_experiment_settings(self):
        self.assertEqual((IDENTITIES_PER_BATCH, IMAGES_PER_IDENTITY), (64, 4))

    def test_each_batch_has_p_identities_and_k_distinct_files_each(self):
        sampler = self.sampler()
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            self.assertEqual(len(batch), sampler.batch_size)
            identities = Counter(self.train.samples[i].identity for i in batch)
            self.assertEqual(len(identities), 5)
            self.assertEqual(set(identities.values()), {3})
            # Distinct files, so every anchor–positive pair compares two different images.
            self.assertEqual(len({self.train.samples[i].path for i in batch}), len(batch))

    def test_targets_separate_positives_from_negatives(self):
        batch = next(iter(self.sampler()))
        for a in batch:
            for b in batch:
                same = self.train.samples[a].identity == self.train.samples[b].identity
                self.assertEqual(
                    same, self.train.samples[a].target == self.train.samples[b].target
                )

    def test_epoch_length_covers_one_pass_over_the_images(self):
        sampler = self.sampler()
        self.assertEqual(len(sampler), len(self.train) // 15)

    def test_same_seed_and_epoch_repeat_and_epochs_differ(self):
        first, again = self.sampler(), self.sampler()
        self.assertEqual(list(first), list(again))
        again.set_epoch(1)
        self.assertNotEqual(list(first), list(again))
        first.set_epoch(1)
        self.assertEqual(list(first), list(again))
        self.assertNotEqual(list(self.sampler(seed=7)), list(self.sampler()))

    def test_draws_only_from_the_indexed_samples(self):
        # A dropped image must be absent from every batch, whatever filter removed it.
        dropped = self.train.samples[0]
        self.train.samples = self.train.samples[1:]
        for batch in self.sampler():
            self.assertNotIn(dropped.path, {self.train.samples[i].path for i in batch})

    def test_identity_with_fewer_than_k_images_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fewer than 5 images"):
            self.sampler(images_per_identity=5)

    def test_too_few_identities_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only 12 identities"):
            self.sampler(identities_per_batch=13)

    def test_degenerate_batch_shapes_are_rejected(self):
        for options in ({"identities_per_batch": 1}, {"images_per_identity": 1}):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                self.sampler(**options)

    def test_held_out_splits_are_rejected(self):
        val = ClassificationFaces(
            self.layout, "val_data", class_to_idx=self.train.class_to_idx
        )
        with self.assertRaisesRegex(ValueError, "only be sampled from train_data"):
            IdentityBatchSampler(val, identities_per_batch=2, images_per_identity=2)

    def test_works_as_a_dataloader_batch_sampler(self):
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            self.skipTest("torch is not installed in this environment")
        from facial_attendance.recognition.data_loader import image_transform

        train = ClassificationFaces(
            self.layout, "train_data", transform=image_transform(8)
        )
        sampler = IdentityBatchSampler(train, identities_per_batch=4, images_per_identity=2)
        images, targets = next(iter(DataLoader(train, batch_sampler=sampler)))
        self.assertEqual(tuple(images.shape), (8, 3, 8, 8))
        self.assertEqual(sorted(Counter(targets.tolist()).values()), [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
