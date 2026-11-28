"""Read an explicit reviewed row selection without copying immutable face caches."""

import bisect
import hashlib
import json
import operator
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from facial_attendance.liveness.dataset_files import file_sha256


def row_order_sha256(samples):
    """Bind row identities and labels so caches cannot silently use a different ordering."""
    digest = hashlib.sha256()
    for row in samples:
        digest.update(
            f"{row.relative_path}\0{row.binary_label}\0{row.source_type}\n".encode()
        )
    return digest.hexdigest()


def bound_file(root, name, digest):
    """Resolve frozen-view metadata inside its root and require the recorded checksum."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != digest:
        raise ValueError("Cache-view file escapes its directory or has changed")
    return path


class _ScoredRows:
    """Expose archived score identities to the same selection gate as images."""

    def __init__(self, rows, data, split="validation"):
        self.samples = [
            SimpleNamespace(
                relative_path=row["relative_path"],
                binary_label=int(row["binary_label"]),
                source_type=int(row["source_type"]),
            )
            for row in rows
        ]
        self.data, self.split = data, split

    def __len__(self):
        return len(self.samples)

    def bindings(self):
        return {
            "inventory_sha256": self.data["inventory_sha256"],
            "annotations_sha256": self.data[self.split]["annotations_sha256"],
        }


def reviewed_prediction_indices(rows, data, directory, split):
    """Bind saved predictions to the same retained indices as image caches."""
    return ReviewedCacheView(_ScoredRows(rows, data, split), directory, split).indices


def reviewed_validation_models(models, directory):
    """Use the image-cache row gate for every saved-score table and plot."""
    selected = {}
    for arm, model in models.items():
        data = model["receipt"]["inference_metadata"]["data"]
        indices = reviewed_prediction_indices(
            model["rows"], data, directory, "validation"
        )
        selected[arm] = {
            **model,
            "rows": [model["rows"][int(i)] for i in indices],
            "labels": model["labels"][indices],
            "scores": model["scores"][indices],
            "correct": model["correct"][indices],
        }
    return selected


class IndexedRows:
    """Apply the same source indices to geometric maps, without a full copy."""

    def __init__(self, source, indices):
        self.source, self.indices = source, indices
        self.shape = (len(indices), *source.shape[1:])
        self.dtype = source.dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


class ReviewedCacheView:
    """Filter source rows while preserving image, label and teacher-map alignment.

    Retained rows are the original split assignment minus confirmed quarantines, not a
    claim of complete semantic clearance. Unresolved reviewed cases block normal
    use unless a caller explicitly requests diagnostic access.
    """

    def __init__(self, source, directory, split, *, allow_pending_review=False):
        directory = Path(directory)
        inventory_path = directory / "cache-view.json"
        inventory = json.loads(inventory_path.read_text())
        if inventory.get("schema") != "liveness-reviewed-cache-view-v1":
            raise ValueError("Unsupported reviewed cache view")
        record = inventory["splits"][split]
        binding = source.bindings()
        if (
            binding["inventory_sha256"] != inventory["inventory_sha256"]
            or binding["annotations_sha256"] != record["annotations_sha256"]
            or len(source) != record["base_samples"]
            or row_order_sha256(source.samples) != record["base_row_order_sha256"]
        ):
            raise ValueError(
                "Cache view differs from source split, labels or row order"
            )
        if record["pending_review"] and not allow_pending_review:
            raise ValueError(
                "Unresolved reviewed cases require explicit diagnostic access"
            )
        indices = np.load(
            bound_file(
                directory, record["keep_indices"], record["keep_indices_sha256"]
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            indices.dtype != np.int64
            or indices.ndim != 1
            or len(indices) != record["retained_samples"]
            or (len(indices) and (indices[0] < 0 or indices[-1] >= len(source)))
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("Invalid reviewed row-index mapping")
        self.source, self.indices = source, indices
        self.samples = [source.samples[int(index)] for index in indices]
        self.view_sha256 = file_sha256(inventory_path)
        self.split = split

    def __len__(self):
        return len(self.indices)

    def base_index(self, index):
        """Map a retained sample position back to its original image/teacher-map row."""
        return int(self.indices[operator.index(index)])

    def __getitem__(self, index):
        return self.source[self.base_index(index)]

    def select_maps(self, maps):
        """Apply the image selection to teacher maps lazily, without copying the full array."""
        if len(maps) != len(self.source):
            raise ValueError(
                "Teacher maps differ from the original source split assignment"
            )
        return IndexedRows(maps, self.indices)

    def bindings(self):
        """Retain original input bindings and add the exact reviewed-selection identity."""
        return {
            **self.source.bindings(),
            "samples": len(self),
            "base_samples": len(self.source),
            "reviewed_cache_view_sha256": self.view_sha256,
            "selection_scope": "Confirmed exclusions from a bounded review; not full dataset clearance",
        }


class TransportFaces:
    """Read existing lossless PNG/TAR cache rows without unpacking full arrays.

    A touched shard is SHA-verified before use. Original source annotation order
    supplies labels; cache pixels are returned unchanged. This does not certify
    source crop semantics or establish whole-array parity from sampled access.
    """

    def __init__(self, source, directory):
        self.source, self.samples = source, source.samples
        self.directory = Path(directory)
        path = self.directory / "face-cache-shards.json"
        self.receipt = json.loads(path.read_text())
        data, binding = self.receipt["data"], source.bindings()
        if self.receipt.get("schema") != "liveness-face-cache-transport-v1":
            raise ValueError("Unsupported transport cache")
        for key in ("inventory_sha256", "annotations_sha256", "samples"):
            if data[key] != binding[key]:
                raise ValueError("Transport cache source assignment differs")
        if (
            self.receipt["shape"] != [len(source), 224, 224, 3]
            or self.receipt["dtype"] != "uint8"
        ):
            raise ValueError("Transport cache has unsupported pixels")
        self.shards = self.receipt["shards"]
        previous = 0
        for shard in self.shards:
            if shard["first_row"] != previous or shard["stop_row"] <= previous:
                raise ValueError("Transport shard row ranges overlap or have gaps")
            previous = shard["stop_row"]
        if previous != len(source):
            raise ValueError("Transport shard coverage differs from source")
        self.stops = [item["stop_row"] for item in self.shards]
        self.verified = {}
        self.receipt_sha256 = file_sha256(path)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = self.shards[bisect.bisect_right(self.stops, index)]
        path = (self.directory / shard["file"]).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Transport shard escapes its cache")
        stat = path.stat()
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if self.verified.get(str(path)) != identity:
            if stat.st_size != shard["bytes"] or file_sha256(path) != shard["sha256"]:
                raise ValueError("Transport shard checksum mismatch")
            self.verified[str(path)] = identity
        with tarfile.open(path, mode="r:") as archive:
            member = archive.getmember(f"{index:09d}.png")
            if not member.isfile():
                raise ValueError("Transport row is not a regular PNG file")
            with archive.extractfile(member) as stream, Image.open(stream) as image:
                image.load()
                if image.mode != "RGB" or image.size != (224, 224):
                    raise ValueError("Transport face is not 224×224 RGB")
                pixels = image.copy()
        return pixels, self.source.targets[index]

    def bindings(self):
        """Add the transport receipt checksum without changing original source bindings."""
        return {
            **self.source.bindings(),
            "transport_inventory_sha256": self.receipt_sha256,
        }
