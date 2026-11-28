"""Pack cropped face pixels into PNG files inside uncompressed tar shards.

The pack operation reads extracted source images and face boxes. The materialize
operation reconstructs a uint8 [images, 224, 224, 3] RGB array on runtime storage.
PNG encoding preserves the face pixels; it does not preserve the original JPEG
file bytes. Shard checksums and sample order are recorded in a JSON inventory.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from torchvision.transforms import Resize

from facial_attendance.liveness.datasets.decoded_cache import (
    FACE_SHAPE,
    DecodedFaces,
    _array_identity,
)
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages

SCHEMA = "liveness-face-cache-transport-v1"
CONTENT_BINDINGS = ("inventory_sha256", "annotations_sha256", "samples", "crop_recipe")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_hasher(samples):
    # Match open_memmap's NPY v1 header, allowing comparison with historical
    # whole-array checksums without allocating that array during packing.
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.dtype(np.uint8).str,
            "fortran_order": False,
            "shape": (samples, *FACE_SHAPE),
        },
    )
    digest = hashlib.sha256()
    digest.update(header.getvalue())
    return digest


def shard_path(directory, record, staging=None):
    name = record["file"]
    if Path(name).name != name or not name.endswith(".tar"):
        raise ValueError("Invalid shard filename")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Shard path escapes its cache directory")
    if path.stat().st_size != record["bytes"]:
        raise ValueError("Shard checksum or size mismatch: " + name)
    if staging is not None:
        # Read a remote shard once. Decode its verified local copy instead of
        # performing many small PNG reads through a mounted cloud filesystem.
        target = Path(staging) / name
        digest = hashlib.sha256()
        with path.open("rb") as source, target.open("xb") as destination:
            while chunk := source.read(8 * 2**20):
                digest.update(chunk)
                destination.write(chunk)
        actual = digest.hexdigest()
        path = target
    else:
        actual = sha256(path)
    if path.stat().st_size != record["bytes"] or actual != record["sha256"]:
        raise ValueError("Shard checksum or size mismatch: " + name)
    return path


def decode_png(index, encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        if image.mode != "RGB" or image.size != (224, 224):
            raise ValueError("Expected lossless RGB 224x224 cache pixels")
        image.load()
        return index, np.asarray(image, dtype=np.uint8).copy()


def decoded_rows(path, first, stop, workers=4):
    index = first
    pending = deque()
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tarfile.open(path, mode="r|") as archive,
    ):
        for member in archive:
            if (
                index >= stop
                or not member.isfile()
                or member.name != f"{index:09d}.png"
            ):
                raise ValueError("Unexpected shard member or row order")
            if not 0 < member.size <= int(np.prod(FACE_SHAPE)) + 65536:
                raise ValueError("Invalid encoded face size")
            with archive.extractfile(member) as handle:
                encoded = handle.read()
            pending.append(pool.submit(decode_png, index, encoded))
            # Preserve row order while bounding decoded and encoded buffers.
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
            index += 1
        while pending:
            yield pending.popleft().result()
    if index != stop:
        raise ValueError("Shard row coverage is incomplete")


def encode_face(source, index):
    image, _ = source[index]
    face = Resize((224, 224))(image)
    if face.mode != "RGB":
        raise ValueError("Source crop must be RGB")
    output = io.BytesIO()
    face.save(output, format="PNG", compress_level=1)
    return index, output.getvalue(), face.tobytes()


def pack_faces(source, output, *, workers=4, shard_mib=128, expected_sha256=None):
    """Bound encoding work in flight; reuse only checksum-verified finished shards."""
    if workers < 1 or shard_mib < 1 or not len(source):
        raise ValueError(
            "Positive workers, shard capacity and split assignment required"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "face-cache-shards.json"
    inventory = {
        "schema": SCHEMA,
        "data": source.bindings(),
        "shape": [len(source), *FACE_SHAPE],
        "dtype": "uint8",
        "encoding": "Uncompressed USTAR; individual lossless RGB PNG faces",
        "row_order": "Lexicographically sorted annotation paths",
        "expected_historical_array_sha256": expected_sha256,
        "producer_sha256": sha256(Path(__file__)),
        "shards": [],
        "completed": False,
    }
    if inventory_path.exists():
        previous = json.loads(inventory_path.read_text())
        for key in (
            "schema",
            "data",
            "shape",
            "dtype",
            "expected_historical_array_sha256",
            "producer_sha256",
        ):
            if previous[key] != inventory[key]:
                raise ValueError(
                    "Cannot mix another source or producer into this cache"
                )
        inventory = previous
    digest, start = array_hasher(len(source)), 0
    for record in inventory["shards"]:
        if record["first_row"] != start or not start < record["stop_row"] <= len(
            source
        ):
            raise ValueError("Shard ranges must cover consecutive source rows")
        for _, pixels in decoded_rows(
            shard_path(output, record), start, record["stop_row"]
        ):
            digest.update(pixels.tobytes())
        start = record["stop_row"]
    if inventory["completed"]:
        if start != len(source) or digest.hexdigest() != inventory["array_sha256"]:
            raise ValueError("Completed face cache changed")
        return inventory
    write_json(inventory_path, inventory)
    archive = None
    partial = None
    first_row = start
    limit = shard_mib * 2**20

    def finish(stop_row):
        nonlocal archive
        archive.close()
        archive = None
        target = partial.with_suffix("")
        if target.exists():
            raise FileExistsError("Unattributed shard already exists: " + str(target))
        partial.replace(target)
        inventory["shards"].append(
            {
                "file": target.name,
                "first_row": first_row,
                "stop_row": stop_row,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
        write_json(inventory_path, inventory)
        print(
            json.dumps({"packed_rows": stop_row, "total_rows": len(source)}), flush=True
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Two requests per worker bound futures and encoded pixel buffers.
            for batch in range(start, len(source), workers * 2):
                indexes = range(batch, min(batch + workers * 2, len(source)))
                for index, encoded, pixels in pool.map(
                    lambda i: encode_face(source, i), indexes
                ):
                    member_bytes = 512 + ((len(encoded) + 511) // 512) * 512
                    if member_bytes + 10240 > limit:
                        raise ValueError("One face exceeds the chosen shard capacity")
                    if (
                        archive is not None
                        and archive.offset + member_bytes + 10240 > limit
                    ):
                        finish(index)
                    if archive is None:
                        if shutil.disk_usage(output).free < limit + 2**30:
                            raise ValueError(
                                "Insufficient space for another bounded shard"
                            )
                        first_row = index
                        partial = (
                            output / f"faces-{len(inventory['shards']):05d}.tar.partial"
                        )
                        if partial.exists():
                            raise FileExistsError(
                                "Inspect incomplete owned shard before retrying: "
                                + str(partial)
                            )
                        archive = tarfile.open(
                            partial, mode="w", format=tarfile.USTAR_FORMAT
                        )
                    member = tarfile.TarInfo(f"{index:09d}.png")
                    member.size = len(encoded)
                    archive.addfile(member, io.BytesIO(encoded))
                    digest.update(pixels)
        if archive is not None:
            finish(len(source))
    finally:
        if archive is not None:
            archive.close()
    inventory["array_sha256"] = digest.hexdigest()
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(
            "Reconstructed array would differ from the historical face cache"
        )
    inventory["completed"] = True
    write_json(inventory_path, inventory)
    return inventory


def materialize_faces(source, transport, output):
    """Build the existing trainer's mmap cache without original images or ZIPs."""
    transport, output = Path(transport).resolve(), Path(output).resolve()
    inventory_path = transport / "face-cache-shards.json"
    inventory = json.loads(inventory_path.read_text())
    binding = source.bindings()
    if (
        inventory["schema"] != SCHEMA
        or not inventory["completed"]
        or inventory["shape"] != [len(source), *FACE_SHAPE]
        or inventory["dtype"] != "uint8"
        or any(
            inventory["data"].get(key) != binding.get(key) for key in CONTENT_BINDINGS
        )
    ):
        raise ValueError(
            "Transport does not match this completed source split assignment"
        )
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "decoded-faces.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != inventory["array_sha256"]:
            raise ValueError("Existing decoded cache differs from the transport")
        return DecodedFaces(source, output)
    partial_path = output / "decoded-faces-building.json"
    array_path = output / "decoded-faces.npy"
    transport_sha = sha256(inventory_path)
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        if partial["transport_sha256"] != transport_sha or partial["data"] != binding:
            raise ValueError(
                "Partial array belongs to another transport or split assignment"
            )
        start = partial["completed_rows"]
        images = np.load(array_path, mmap_mode="r+")
        if images.shape != tuple(inventory["shape"]) or images.dtype != np.uint8:
            raise ValueError("Invalid partial cache shape or dtype")
    else:
        if array_path.exists():
            raise FileExistsError("Unattributed decoded array exists")
        required = len(source) * int(np.prod(FACE_SHAPE)) + 2**30
        if shutil.disk_usage(output).free < required:
            raise ValueError("Insufficient space for the decoded uint8 array")
        images = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.uint8, shape=tuple(inventory["shape"])
        )
        start = 0
        partial = {
            "transport_sha256": transport_sha,
            "data": binding,
            "completed_rows": 0,
        }
        write_json(partial_path, partial)
    cursor = 0
    for record in inventory["shards"]:
        stop = record["stop_row"]
        if record["first_row"] != cursor or not cursor < stop <= len(source):
            raise ValueError("Invalid shard split assignment ranges")
        if stop > start:
            if cursor != start:
                raise ValueError("Partial progress must end at a complete shard")
            # Only one transport shard coexists with the mmap output on SSD.
            with tempfile.TemporaryDirectory(
                dir=output.parent, prefix="face-cache-shard-"
            ) as staging:
                local_shard = shard_path(transport, record, staging=staging)
                for index, pixels in decoded_rows(local_shard, cursor, stop):
                    images[index] = pixels
            images.flush()
            start = stop
            partial["completed_rows"] = stop
            write_json(partial_path, partial)
            print(
                json.dumps({"materialized_rows": stop, "total_rows": len(source)}),
                flush=True,
            )
        cursor = stop
    if cursor != len(source):
        raise ValueError("Transport omits source rows")
    images.flush()
    del images
    digest = sha256(array_path)
    if digest != inventory["array_sha256"]:
        raise ValueError(
            "Materialized pixels differ from the producer's whole-array SHA"
        )
    receipt = {
        "schema": "liveness-decoded-faces-v1",
        "data": binding,
        "shape": inventory["shape"],
        "dtype": "uint8",
        "sha256": digest,
        "array_identity": list(_array_identity(array_path)),
        "transport_manifest_sha256": transport_sha,
        "verification": "Lossless PNG shards; verified source split assignment, shard SHA and whole-array SHA",
    }
    write_json(receipt_path, receipt)
    return DecodedFaces(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "materialize"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transport", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-mib", type=int, default=128)
    parser.add_argument("--expected-array-sha256")

    args = parser.parse_args()
    source = ReproductionImages(args.data_root, args.inventory, args.annotations)
    if args.mode == "pack":
        if args.expected_array_sha256 is None:
            parser.error(
                "pack requires --expected-array-sha256 from the historical cache receipt"
            )
        pack_faces(
            source,
            args.output_dir,
            workers=args.workers,
            shard_mib=args.shard_mib,
            expected_sha256=args.expected_array_sha256,
        )
    else:
        if args.transport is None:
            parser.error("materialize requires --transport")
        materialize_faces(source, args.transport, args.output_dir)


if __name__ == "__main__":
    main()
