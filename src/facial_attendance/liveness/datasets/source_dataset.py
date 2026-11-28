"""All-eleven-type source data for the declared AENet reproduction experiment.

Uses explicit annotation split assignment and the existing per-file inventory hashes.
It does not apply the LIV-03 print/screen selection to a paper reproduction, nor
modify that selection. Project-derived validation remains attributed as such.
"""

from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset

from facial_attendance.liveness.dataset_files import file_sha256, local_path, read_json
from facial_attendance.liveness.datasets.face_crop import bbox_path
from facial_attendance.liveness.datasets.image_inventory import read_inventory
from facial_attendance.liveness.models.multitask_model import (
    ALL_SOURCE_TYPES,
    annotation_targets,
)

SOURCE_CROP_RECIPE = "celeba-spoof-client-integer-coordinates-ratio-first-v1"


def read_reproduction_face(image, box):
    """Follow the source client's full-image fallback for unparseable boxes.

    CelebA-Spoof intra_dataset_code/client.py retains its original image when
    unpacking/converting box coordinates fails. Keep this reproduction-only:
    boundary clipping is allowed by the source client. Empty geometry and
    changed/missing files still fail; LIV-03's strict crop contract is unchanged.
    """
    values = box.read_text(encoding="utf-8").splitlines()
    values = values[0].strip().split(" ") if values else []
    try:
        if len(values) != 5:
            raise ValueError("Unparseable source box")
        # The source parses the four coordinates, without using its score.
        x, y, width, height = (int(float(value)) for value in values[:4])
    except (ValueError, OverflowError):
        with Image.open(image) as source:
            source.load()
            return source.convert("RGB")
    with Image.open(image) as source:
        source.load()
        image_width, image_height = source.size
        # Match both integer conversion stages and operation order in client.py.
        # int(v * (size / 224)) can differ by one pixel from int(v * size / 224).
        x, y = int(x * (image_width / 224)), int(y * (image_height / 224))
        width = int(width * (image_width / 224))
        height = int(height * (image_height / 224))
        left, top = max(0, x), max(0, y)
        right = image_width if left + width > image_width else x + width
        bottom = image_height if top + height > image_height else y + height
        if width <= 0 or height <= 0 or left >= right or top >= bottom:
            raise ValueError("Empty or invalid source bounding box")
        crop = (left, top, right, bottom)
        return source.convert("RGB").crop(crop)


class ReproductionImages(Dataset):
    """Read explicit split rows in sorted path order, with labels bound to the image inventory."""

    def __init__(self, root, inventory, annotations):
        self.root = Path(root).resolve()
        self.inventory = Path(inventory).resolve()
        self.annotations = Path(annotations).resolve()
        raw = read_json(self.annotations)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Expected a nonempty explicit annotation mapping")
        records = {row.relative_path: row for row in read_inventory(self.inventory)}
        self.samples, self.targets = [], []
        for name in sorted(raw):
            if name not in records:
                raise ValueError(
                    f"Source image is absent from the integrity inventory: {name}"
                )
            row = records[name]
            target = annotation_targets(raw[name], source_types=ALL_SOURCE_TYPES)
            if target["binary"] != row.binary_label or raw[name][40] != row.source_type:
                raise ValueError(
                    f"Source labels disagree with the integrity inventory: {name}"
                )
            self.samples.append(row)
            self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image = local_path(self.root, row.relative_path)
        box = local_path(self.root, str(bbox_path(PurePosixPath(row.relative_path))))
        if file_sha256(image) != row.sha256 or file_sha256(box) != row.bbox_sha256:
            raise ValueError(
                f"Source image/box changed since inventory: {row.relative_path}"
            )
        return read_reproduction_face(image, box), self.targets[index]

    def bindings(self):
        """Return source paths, content hashes and crop recipe for cache and run binding."""
        return {
            "root": str(self.root),
            "inventory": str(self.inventory),
            "inventory_sha256": file_sha256(self.inventory),
            "annotations": str(self.annotations),
            "annotations_sha256": file_sha256(self.annotations),
            "samples": len(self),
            "crop_recipe": SOURCE_CROP_RECIPE,
        }


def check_reproduction_partitions(train, validation):
    """Reject overlapping train/validation identities or image paths before training."""
    train_subjects = {row.subject_id for row in train.samples}
    validation_subjects = {row.subject_id for row in validation.samples}
    if train_subjects & validation_subjects:
        raise ValueError("Reproduction train/validation subjects overlap")
    train_names = {row.relative_path for row in train.samples}
    if train_names & {row.relative_path for row in validation.samples}:
        raise ValueError("Reproduction train/validation images overlap")


class GeometricTrainingImages(Dataset):
    """Pair each source crop with teacher maps at the same original cache row."""

    def __init__(self, source, maps, transform):
        if len(source) != len(maps):
            raise ValueError("Source and geometric cache split assignment differ")
        self.source, self.maps, self.transform = source, maps, transform
        self.samples = source.samples

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, source_target = self.source[index]
        # Copy the selected small mmap row before tensor conversion; workers
        # must not expose mutable tensors backed by a read-only cache mapping.
        maps = torch.from_numpy(self.maps[index].astype("float32", copy=True))
        target = {**source_target, "depth_map": maps[:1], "reflection_map": maps[1:]}
        return self.transform(image, target)


class BinaryReproductionImages(Dataset):
    """Expose binary labels and the checkpoint transform over a source dataset."""

    def __init__(self, source, transform):
        self.source, self.transform, self.samples = source, transform, source.samples

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, targets = self.source[index]
        return self.transform(image), targets["binary"]
