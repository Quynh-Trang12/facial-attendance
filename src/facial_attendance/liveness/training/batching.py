"""Choose and replay training rows without changing model initialization RNG."""

import random
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def sample_rows(dataset):
    """Return inventory rows in dataset order, including the order imposed by Subset."""
    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[i] for i in dataset.indices]
    return dataset.samples


def training_subset(dataset, budget, seed, epoch, live_budget=None):
    """Match exposure across data policies without consuming initialization RNG."""
    if budget is None:
        if live_budget is not None:
            raise ValueError("Class budget requires an epoch sample budget")
        return dataset
    if not 1 <= budget <= len(dataset):
        raise ValueError("Epoch sample budget must fit the training pool")
    generator = torch.Generator().manual_seed(seed + epoch)
    if live_budget is None:
        indices = torch.randperm(len(dataset), generator=generator)[:budget].tolist()
    else:
        rows = sample_rows(dataset)
        indices = []
        for label, count in ((0, live_budget), (1, budget - live_budget)):
            pool = [i for i, row in enumerate(rows) if row.binary_label == label]
            if not 0 <= count <= len(pool):
                raise ValueError("Class budget must fit the training pool")
            order = torch.randperm(len(pool), generator=generator)[:count].tolist()
            indices.extend(pool[i] for i in order)
    return Subset(dataset, indices)


def make_loader(dataset, config, *, training=False, epoch=0):
    """Build a seeded batch loader; validation order stays fixed and is never shuffled."""
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=False,
    )


def seed_training(seed):
    """Reset Python, NumPy and Torch RNG streams before matched model/epoch work."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_training_config(config, epoch):
    """Apply a declared batch-size change only at its starting epoch."""
    batch = (
        config.large_batch_size
        if config.large_batch_size is not None
        and epoch >= config.large_batch_start_epoch
        else config.batch_size
    )
    return replace(config, batch_size=batch)


class SeededTrainingImages(Dataset):
    """Make each image augmentation reproducible from seed, epoch and row index."""

    def __init__(self, source, seed):
        self.source, self.seed, self.epoch = source, seed, 1
        self.samples = source.samples

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        # ColorJitter consumes the CPU torch RNG. A per-image seed makes its
        # output independent of worker prefetch and replay position.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * len(self.source) + index)
            return self.source[index]


def epoch_loader(dataset, config, epoch, start_sample=0):
    """Recreate the seeded image permutation and omit already completed samples.

    start_sample is the number of training images completed in this epoch; it must
    be a batch boundary or the full dataset length. Only the remaining rows are read.
    """
    if not 0 <= start_sample <= len(dataset):
        raise ValueError(
            "Resume sample cursor is outside the training split assignment"
        )
    if start_sample != len(dataset) and start_sample % config.batch_size:
        raise ValueError("Resume cursor must be at a completed batch boundary")
    dataset.epoch = epoch
    generator = torch.Generator().manual_seed(config.seed + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=order[start_sample:],
        num_workers=config.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed + epoch),
        persistent_workers=False,
    )
