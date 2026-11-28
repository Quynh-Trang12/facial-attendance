"""Generate and verify depth/reflection training targets for each face image.

The float32 array has shape [images, 4, 14, 14]. Index 0 on the second axis is
relative depth; indices 1, 2 and 3 are the predicted reflection image's RGB grids.
The receipt records the source inventory, split assignment, teacher weights and
array checksum so the targets can be matched to the same training images.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import Resize, ToTensor

from facial_attendance.liveness.dataset_files import (
    compatible_sources,
    file_sha256,
    source_path,
)
from facial_attendance.liveness.models.depth_teacher import (
    PRNetDepthTeacher,
    prnet_face_input,
    rasterize_depth,
    restore_prnet_vertices,
)
from facial_attendance.liveness.models.reflection_teacher import ReflectionTeacher


def checked_teacher_map(value, channels, name):
    """Check a teacher grid and resize it to 14 rows by 14 columns.

    Input axes are [channels, height, width]. Values must be finite floats in [0, 1];
    channels is 1 for depth or 3 for the reflection image's red/green/blue values.
    """
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] != channels
        or min(value.shape[1:]) < 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or (value < 0).any()
        or (value > 1).any()
    ):
        raise ValueError(f"{name} must be a finite CHW float teacher map in [0,1]")
    return F.interpolate(
        value.float().unsqueeze(0), size=(14, 14), mode="bilinear", align_corners=False
    ).squeeze(0)


def geometric_targets(
    binary_label, *, depth_prediction=None, reflection_prediction=None
):
    """Build depth and RGB reflection targets for one labelled face crop.

    For a live image, use the PRNet depth estimate and three zero reflection grids.
    For a spoof image, use one zero depth grid and the reflection teacher's RGB grids.
    A required teacher estimate must be supplied and already aligned to the crop.
    """
    if type(binary_label) is not int or binary_label not in (0, 1):
        raise ValueError("Expected binary source label 0=live or 1=spoof")
    if binary_label == 0:
        depth = checked_teacher_map(depth_prediction, 1, "PRNet depth")
        if not torch.count_nonzero(depth):
            raise ValueError("Live depth teacher returned an all-zero map")
        return {"depth_map": depth, "reflection_map": depth.new_zeros((3, 14, 14))}
    reflection = checked_teacher_map(reflection_prediction, 3, "Reflection prediction")
    # Many spoof images have no reflection, as noted in the source paper. Zero
    # is valid when a real teacher produced it; a missing prediction is not.
    return {
        "depth_map": reflection.new_zeros((1, 14, 14)),
        "reflection_map": reflection,
    }


MAP_SHAPE = (4, 14, 14)
MAP_DTYPE = "<f4"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cache_bindings(dataset, depth_checkpoint, reflection_checkpoint, mesh_directory):
    return {
        "schema": "binary-geometric-cache-v1",
        "data": dataset.bindings(),
        "depth_checkpoint_sha256": file_sha256(depth_checkpoint),
        "reflection_checkpoint_sha256": file_sha256(reflection_checkpoint),
        "mesh_sha256": {
            name: file_sha256(mesh_directory / name)
            for name in ("face_ind.txt", "triangles.txt")
        },
        "shape": [len(dataset), *MAP_SHAPE],
        "dtype": MAP_DTYPE,
        "row_order": "lexicographically sorted annotation paths",
        "projection": "PRNet supplied-box 1.6 crop; min/max z; source mean-triangle z-buffer projected directly to 14x14",
        "reflection": "Source BGR teacher; RGB 224x224 face input; clamp/uint8 quantization then bilinear 14x14",
        "source_sha256": {
            name: file_sha256(source_path(name))
            for name in (
                "datasets/geometric_cache.py",
                "models/depth_teacher.py",
                "models/reflection_teacher.py",
                "datasets/source_dataset.py",
            )
        },
    }


def compatible_cache_bindings(previous, current):
    return (
        previous.keys() == current.keys()
        and all(
            previous[key] == current[key] for key in previous if key != "source_sha256"
        )
        and compatible_sources(previous["source_sha256"], current["source_sha256"])
    )


@torch.inference_mode()
def generate_cache(
    dataset,
    depth_checkpoint,
    reflection_checkpoint,
    mesh_directory,
    output,
    *,
    flush_images=64,
    resume=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; select a GPU runtime. No CPU fallback.")
    if flush_images < 1:
        raise ValueError("Cache progress interval must be positive")

    torch.set_num_threads(2)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = cache_bindings(
        dataset, depth_checkpoint, reflection_checkpoint, mesh_directory
    )
    receipt_path = output / "geometric-cache.json"
    array_file = output / "geometric-targets.npy"

    if resume is not None:
        previous = Path(resume).resolve()
        previous_receipt = json.loads(
            (previous / "geometric-cache.json").read_text(encoding="utf-8")
        )
        if not compatible_cache_bindings(previous_receipt["bindings"], binding):
            raise ValueError(
                "Resume cache belongs to another source/teacher/projection"
            )
        if receipt_path.exists() or array_file.exists():
            raise FileExistsError("Resume destination must be a new per-job output")
        required = (previous / "geometric-targets.npy").stat().st_size + 512 * 1024**2
        if shutil.disk_usage(output).free < required:
            raise ValueError(
                "Need storage for a new writable copy of the prior job cache"
            )
        shutil.copyfile(previous / "geometric-targets.npy", array_file)
        atomic_json(receipt_path, previous_receipt)
        atomic_json(
            output / "cache-resume.json",
            {
                "source_directory": str(previous),
                "source_inventory_sha256": file_sha256(
                    previous / "geometric-cache.json"
                ),
                "first_resumed_row": previous_receipt["completed_rows"],
            },
        )

    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not compatible_cache_bindings(receipt["bindings"], binding):
            raise ValueError(
                "Cannot resume cache after source/teacher/projection changes"
            )
        if receipt["completed"]:
            if file_sha256(array_file) != receipt["array_sha256"]:
                raise ValueError("Completed geometric cache changed")
            return receipt
        if receipt["bindings"] != binding:
            receipt["source_relocation"] = {
                "previous_source_sha256": receipt["bindings"]["source_sha256"],
                "continued_source_sha256": binding["source_sha256"],
                "completed_rows_before_relocation": receipt["completed_rows"],
            }
            receipt["bindings"] = binding
            relocated = output / "source-continued"
            relocated.mkdir(exist_ok=False)
            for name in binding["source_sha256"]:
                (relocated / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path(name), relocated / name)
        start = receipt["completed_rows"]
        maps = np.load(array_file, mmap_mode="r+")
        if (
            maps.shape != tuple(binding["shape"])
            or maps.dtype.str != MAP_DTYPE
            or not 0 <= start <= len(dataset)
        ):
            raise ValueError("Partial cache shape/dtype/progress mismatch")
    else:
        if array_file.exists():
            raise ValueError(
                "Unattributed cache array exists without its source receipt"
            )
        required = (
            len(dataset) * np.prod(MAP_SHAPE) * np.dtype(MAP_DTYPE).itemsize
            + 512 * 1024**2
        )
        if shutil.disk_usage(output).free < required:
            raise ValueError(
                f"Insufficient storage for source-bound pseudo-label cache: need {int(required)} free bytes"
            )
        maps = np.lib.format.open_memmap(
            array_file, mode="w+", dtype=MAP_DTYPE, shape=tuple(binding["shape"])
        )
        start = 0
        source_output = output / "source"
        source_output.mkdir()
        for name in binding["source_sha256"]:
            (source_output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(name), source_output / name)
        atomic_json(
            receipt_path, {"bindings": binding, "completed_rows": 0, "completed": False}
        )

    depth = PRNetDepthTeacher().cuda().eval()
    depth.load_state_dict(
        torch.load(depth_checkpoint, map_location="cpu", weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    reflection = ReflectionTeacher().cuda().eval()
    reflection.load_state_dict(
        torch.load(reflection_checkpoint, map_location="cpu", weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    depth.requires_grad_(False)
    reflection.requires_grad_(False)

    face_indices = torch.tensor(
        np.loadtxt(mesh_directory / "face_ind.txt").astype(np.int64), device="cuda"
    )
    triangles = torch.tensor(
        np.loadtxt(mesh_directory / "triangles.txt").astype(np.int64), device="cuda"
    )
    to_tensor, resize = ToTensor(), Resize((224, 224))

    started = time.monotonic()
    for index in range(start, len(dataset)):
        image, labels = dataset[index]
        if labels["binary"] == 0:
            image_tensor = to_tensor(image).cuda()
            cropped, alignment = prnet_face_input(image_tensor)
            position = depth(cropped)[0]
            vertices = restore_prnet_vertices(position, face_indices, alignment)
            target = geometric_targets(
                0, depth_prediction=rasterize_depth(vertices, triangles)
            )
        else:
            image_tensor = to_tensor(resize(image)).unsqueeze(0).cuda()
            prediction = reflection.pseudo_reflection(image_tensor)[0]
            target = geometric_targets(1, reflection_prediction=prediction)
        packed = torch.cat((target["depth_map"], target["reflection_map"]), dim=0)
        maps[index] = packed.cpu().numpy().astype(MAP_DTYPE)
        if (index + 1) % flush_images == 0 or index + 1 == len(dataset):
            maps.flush()
            receipt = {
                "bindings": binding,
                "completed_rows": index + 1,
                "completed": False,
                "current_execution_seconds": time.monotonic() - started,
                "current_execution_rows": index + 1 - start,
            }
            atomic_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "cache": output.name,
                        "completed_rows": index + 1,
                        "total": len(dataset),
                        "images_per_second": (index + 1 - start)
                        / (time.monotonic() - started),
                    }
                ),
                flush=True,
            )

    maps.flush()
    receipt.update(completed=True, array_sha256=file_sha256(array_file))
    atomic_json(receipt_path, receipt)
    return receipt


def load_geometric_cache(directory, source):
    """Verify the saved target rows against the supplied face dataset.

    Require all rows to be complete, matching source/split records and a valid whole
    array checksum. Return a read-only [images, 4, 14, 14] float32 array and receipt.
    """
    directory = Path(directory).resolve()
    receipt = json.loads(
        (directory / "geometric-cache.json").read_text(encoding="utf-8")
    )
    if not receipt["completed"] or receipt["completed_rows"] != len(source):
        raise ValueError("Pseudo-label generation is incomplete")
    if receipt["bindings"]["data"] != source.bindings():
        raise ValueError(
            "Geometric cache belongs to another source inventory or split assignment"
        )
    array_file = directory / "geometric-targets.npy"
    if file_sha256(array_file) != receipt["array_sha256"]:
        raise ValueError("Geometric cache array changed")
    maps = np.load(array_file, mmap_mode="r")
    if maps.shape != (len(source), *MAP_SHAPE) or maps.dtype.str != MAP_DTYPE:
        maps._mmap.close()
        raise ValueError("Geometric cache shape/dtype mismatch")
    return maps, receipt
