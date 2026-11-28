"""Store cropped RGB face images before training-time colour augmentation.

The uint8 array has shape [images, 224, 224, 3], with channels red, green and blue.
The JSON receipt records source-file checksums and the assigned sample order.
The loader reads from a memory-mapped file on SSD or a caller-selected RAM disk;
colour augmentation runs after a face image has been read.
"""

import hashlib
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize

FACE_SHAPE = (224, 224, 3)
_source = _images = None
_verified_arrays = {}


def _array_identity(path):
    info = path.stat()
    return (
        str(path.resolve()),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _initialize_builder(source, images):
    global _source, _images
    _source, _images = source, images


def _prepare_row(index):
    image, _ = _source[index]
    _images[index] = np.asarray(Resize((224, 224))(image), dtype=np.uint8)


def build_decoded_cache(source, directory, *, workers=12, progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "decoded-faces.json"
    if receipt_path.exists():
        return DecodedFaces(source, directory)
    array_path = directory / "decoded-faces.npy"
    if array_path.exists():
        raise FileExistsError(
            "Incomplete decoded cache; remove it explicitly before rebuilding"
        )

    required = len(source) * int(np.prod(FACE_SHAPE))
    if workers < 1 or shutil.disk_usage(directory).free < required + 2**30:
        raise ValueError("Insufficient cache capacity or invalid worker count")

    images = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.uint8, shape=(len(source), *FACE_SHAPE)
    )

    # Bound queued tasks, rather than retaining one Future per source image.
    # Fork shares the existing source metadata and writable mmap without
    # serializing hundreds of thousands of annotation tensors to each worker.
    if "fork" in multiprocessing.get_all_start_methods():
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_initialize_builder,
            initargs=(source, images),
        )
    else:
        _initialize_builder(source, images)
        pool = ThreadPoolExecutor(max_workers=workers)

    with pool:
        for first in range(0, len(source), workers * 4):
            list(
                pool.map(
                    _prepare_row, range(first, min(first + workers * 4, len(source)))
                )
            )
            if progress is not None:
                progress(min(first + workers * 4, len(source)), len(source))

    images.flush()
    _initialize_builder(None, None)
    del images

    with array_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _verified_arrays[_array_identity(array_path)] = digest

    receipt = {
        "schema": "liveness-decoded-faces-v1",
        "data": source.bindings(),
        "shape": [len(source), *FACE_SHAPE],
        "dtype": "uint8",
        "sha256": digest,
        "array_identity": list(_array_identity(array_path)),
        "verification": "Source image and box SHA checked before decode; RGB crop and PIL bilinear Resize(224,224) only",
    }

    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(receipt_path)
    return DecodedFaces(source, directory)


class DecodedFaces(Dataset):
    """Read a cached RGB face image and its source annotation by row index.

    Listed replacement rows use face-pixels.npy; other rows use decoded-faces.npy.
    The returned PIL image is 224 by 224 pixels. Augmentation is applied by the
    training dataset wrapper, not by this cache reader.
    """

    def __init__(self, source, directory):
        self.source = source
        self.samples = source.samples
        directory = Path(directory)
        receipt = json.loads((directory / "decoded-faces.json").read_text())
        if receipt["data"] != source.bindings():
            raise ValueError(
                "Decoded cache split assignment or source inventory changed"
            )

        array_path = directory / "decoded-faces.npy"
        identity = _array_identity(array_path)
        digest = _verified_arrays.get(identity)
        # A receipt is written only after the complete content SHA succeeds.
        # Inode/size/mtime/ctime bind that verification to this unchanged RAM
        # file across process resumes. A copied or modified file is rehashed.
        if digest is None and receipt.get("array_identity") == list(identity):
            digest = receipt["sha256"]
        if digest is None:
            with array_path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            _verified_arrays[identity] = digest
        if digest != receipt["sha256"]:
            raise ValueError("Decoded cache content changed")

        self.images = np.load(array_path, mmap_mode="r")
        if self.images.dtype != np.uint8 or self.images.shape != (
            len(source),
            *FACE_SHAPE,
        ):
            raise ValueError("Invalid decoded cache shape or dtype")

        self.correction_lookup = {}
        self.correction_pixels = None
        correction = receipt.get("corrections")
        if correction is not None:
            if correction.get("schema") != "liveness-face-row-corrections-v1":
                raise ValueError("Unsupported face correction schema")
            if set(correction.get("files", {})) != {
                "row-indices.npy",
                "face-pixels.npy",
            }:
                raise ValueError("Face correction needs checksums for rows and pixels")
            for name, digest in correction["files"].items():
                if Path(name).name != name:
                    raise ValueError("Correction filename escapes its cache")
                with (directory / name).open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                        raise ValueError("Face correction checksum mismatch")
            indices = np.load(directory / "row-indices.npy")
            self.correction_pixels = np.load(
                directory / "face-pixels.npy", mmap_mode="r"
            )
            if (
                indices.ndim != 1
                or indices.dtype != np.int64
                or len(indices) != correction["rows"]
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(source))
                or self.correction_pixels.shape != (len(indices), *FACE_SHAPE)
                or self.correction_pixels.dtype != np.uint8
            ):
                raise ValueError("Invalid face correction rows or pixels")
            self.correction_lookup = {
                int(row): position for position, row in enumerate(indices)
            }

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        replacement = self.correction_lookup.get(index)
        pixels = (
            self.images[index]
            if replacement is None
            else self.correction_pixels[replacement]
        )
        return Image.fromarray(pixels), self.source.targets[index]

    def bindings(self):
        return self.source.bindings()
