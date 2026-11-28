"""Identity-balanced P×K batches for triplet loss with online mining."""

from collections import defaultdict
import random

from .data_loader import ClassificationFaces

# Metric-learning batch shape agreed in the experiment settings: 64 identities × 4 images.
IDENTITIES_PER_BATCH = 64
IMAGES_PER_IDENTITY = 4


class IdentityBatchSampler:
    """Yield batches of dataset indices: P distinct identities, K distinct images each.

    Pass it to `torch.utils.data.DataLoader(dataset, batch_sampler=...)`. Triplets are
    mined inside each batch during training: any two images of one identity form an
    anchor–positive pair, and every other identity in the batch supplies negatives.
    Only `dataset.samples` is read, so images the loader dropped can never be drawn.
    """

    def __init__(
        self,
        dataset: ClassificationFaces,
        *,
        identities_per_batch: int = IDENTITIES_PER_BATCH,
        images_per_identity: int = IMAGES_PER_IDENTITY,
        seed: int = 42,
    ):
        # Sampling from held-out splits would mix evaluation data into training.
        if dataset.split != "train_data":
            raise ValueError("Triplet batches may only be sampled from train_data")
        # One identity gives no negatives, and one image per identity gives no positives.
        if identities_per_batch < 2 or images_per_identity < 2:
            raise ValueError("A triplet batch needs at least 2 identities and 2 images each")
        by_identity = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            by_identity[sample.identity].append(index)
        short = sorted(i for i, idx in by_identity.items() if len(idx) < images_per_identity)
        if short:
            raise ValueError(
                f"{len(short)} training identities have fewer than {images_per_identity} "
                f"images, for example {short[:5]}"
            )
        if len(by_identity) < identities_per_batch:
            raise ValueError(
                f"Only {len(by_identity)} identities for {identities_per_batch} per batch"
            )
        self.by_identity = {k: by_identity[k] for k in sorted(by_identity)}
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.epoch = 0
        # One epoch draws about as many images as one pass over the training set.
        self.batches = max(1, len(dataset.samples) // self.batch_size)

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.images_per_identity

    def set_epoch(self, epoch: int):
        """Change the draw per epoch while keeping any epoch reproducible after a resume."""
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        # A local RNG keyed on (seed, epoch) leaves the model-training RNG untouched.
        rng = random.Random(f"{self.seed}:{self.epoch}")
        identities = list(self.by_identity)
        for _ in range(self.batches):
            batch = []
            # Identities are drawn uniformly so large identity folders do not dominate.
            for identity in rng.sample(identities, self.identities_per_batch):
                batch += rng.sample(self.by_identity[identity], self.images_per_identity)
            yield batch
