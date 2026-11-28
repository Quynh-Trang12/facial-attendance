"""Align semantic annotations and balanced cohorts with selected image rows."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.models.multitask_model import (
    SOURCE_TYPES,
    annotation_targets,
)


def read_annotation_targets(dataset, annotation_file, readiness):
    """Read checksum-bound semantic labels in exactly the selected image-row order.

    A mismatched source type or binary label rejects the annotation release.
    """
    annotation_file = Path(annotation_file).resolve()
    receipt = json.loads(Path(readiness).read_text(encoding="utf-8"))
    expected = receipt["source_bindings"].get(str(annotation_file))
    if not expected or file_sha256(annotation_file) != expected:
        raise ValueError("Auxiliary annotations are not bound to the data release")
    raw = json.loads(annotation_file.read_text(encoding="utf-8"))
    targets = []
    for row in dataset.samples:
        attributes = raw[row.relative_path]
        target = annotation_targets(attributes)
        if attributes[40] != row.source_type or target["binary"] != row.binary_label:
            raise ValueError(f"Auxiliary/inventory label mismatch: {row.relative_path}")
        targets.append(target)
    return targets


def balanced_cohort(dataset, targets, total, seed):
    """Half live; half equally allocated to six attacks × four illuminations."""
    if total < 48 or total % 48:
        raise ValueError("Source/illumination budget must be a positive multiple of 48")
    if len(targets) != len(dataset):
        raise ValueError("Targets and inventory must have identical coverage")
    groups = defaultdict(list)
    for index, row in enumerate(dataset.samples):
        groups[(row.source_type, targets[index]["illumination"])].append(index)
    generator = torch.Generator().manual_seed(seed)
    cells = [((0, 0), total // 2)]
    cells += [
        ((source, illumination), total // 48)
        for source in SOURCE_TYPES[1:]
        for illumination in range(1, 5)
    ]
    indices = []
    for key, count in cells:
        pool = groups[key]
        if len(pool) < count:
            raise ValueError(f"Insufficient source/illumination stratum: {key}")
        order = torch.randperm(len(pool), generator=generator)[:count].tolist()
        indices.extend(pool[index] for index in order)
    return indices


class MultitaskImages(Dataset):
    """Pair selected RGB images with aligned semantic targets and a supplied transform."""

    def __init__(self, dataset, indices, targets, transform):
        self.dataset, self.indices = dataset, tuple(indices)
        self.targets, self.transform = targets, transform
        self.samples = tuple(dataset.samples[index] for index in self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image, binary = self.dataset[source_index]
        target = self.targets[source_index]
        if binary != target["binary"]:
            raise ValueError("Binary/auxiliary sample alignment failed")
        return self.transform(image), target
