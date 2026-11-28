"""Build source-bound row corrections instead of regenerating entire face caches.

Original arrays remain immutable. The correction contains replacement face
pixels, affected row indices and real teacher predictions for changed train
crops. Apply creates separate arrays and receipts for the corrected recipe.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torchvision.transforms import Resize

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.datasets.decoded_cache import _array_identity
from facial_attendance.liveness.datasets.geometric_cache import generate_cache
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages

ANNOTATIONS = {
    "train": "reviewed_project_train.json",
    "validation": "project_validation.json",
    "test": "metas/intra_test/test_label.json",
}


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build(args):
    output = args.output or Path(os.environ["GPU_JOB_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "corrections.json").exists():
        raise FileExistsError("A completed correction already exists")

    audit = json.loads(args.audit.read_text())
    inventory = args.data_root / "audits/current-schema-2026-11-12/inventory.csv"
    report = {
        "audit_sha256": file_sha256(args.audit),
        "splits": {},
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).parents[2]
                / "src/facial_attendance/liveness/source_dataset.py",
            )
        },
    }

    selected_train = None
    for split, annotation in ANNOTATIONS.items():
        annotations = args.data_root / annotation
        split_assignment = json.loads(annotations.read_text())
        names = sorted(split_assignment)
        indices = {name: index for index, name in enumerate(names)}
        changed = [row["relative_path"] for row in audit[split]["changes"]]
        if len(changed) != len(set(changed)) or any(
            name not in split_assignment for name in changed
        ):
            raise ValueError(
                "Correction rows differ from the declared split assignment"
            )
        changed.sort(key=indices.__getitem__)
        source = ReproductionImages(args.data_root, inventory, annotations)
        directory = output / split
        directory.mkdir(exist_ok=True)
        pixels = np.lib.format.open_memmap(
            directory / "face-pixels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(changed), 224, 224, 3),
        )

        def decode(name):
            image, _ = source[indices[name]]
            return np.asarray(Resize((224, 224))(image), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for first in range(0, len(changed), 64):
                for index, value in enumerate(
                    pool.map(decode, changed[first : first + 64]), start=first
                ):
                    pixels[index] = value
        pixels.flush()
        del pixels
        np.save(
            directory / "row-indices.npy",
            np.array([indices[name] for name in changed], dtype=np.int64),
        )
        record = {
            "data": source.bindings(),
            "changed_rows": len(changed),
            "paths": changed,
            "files": {
                name: file_sha256(directory / name)
                for name in ("face-pixels.npy", "row-indices.npy")
            },
        }
        report["splits"][split] = record
        print(
            json.dumps({"split": split, "corrected_face_rows": len(changed)}),
            flush=True,
        )
        if split == "train":
            selected_train = {name: split_assignment[name] for name in changed}

    # Reuse the published generator and teachers, but process only changed rows.
    subset_file = output / "changed-training-annotations.json"
    write_json(subset_file, selected_train)
    subset = ReproductionImages(args.data_root, inventory, subset_file)
    assets = args.pretrained
    geometry = output / "geometry"
    receipt = generate_cache(
        subset,
        assets / "depth-teacher.pt",
        assets / "reflection-teacher.pt",
        assets / "prnet-tensorflow",
        geometry,
    )
    report["geometry_array_sha256"] = receipt["array_sha256"]
    report["geometry_manifest_sha256"] = file_sha256(geometry / "geometric-cache.json")
    write_json(output / "corrections.json", report)
    print("Source-bound face and teacher corrections complete.", flush=True)


def apply(args):
    correction = json.loads((args.corrections / "corrections.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        record = correction["splits"][split]
        directory = args.corrections / split
        for name, digest in record["files"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("Correction bytes changed: " + name)
        source = args.face_cache / split
        receipt = json.loads((source / "decoded-faces.json").read_text())
        for key in ("inventory_sha256", "annotations_sha256", "samples"):
            if receipt["data"][key] != record["data"][key]:
                raise ValueError("Correction and base split assignment differ")
        base = source / "decoded-faces.npy"
        # Reuse a SHA verification bound to this unchanged local inode, exactly
        # as the decoded loader does. A transferred or modified file is hashed.
        if (
            receipt.get("array_identity") != list(_array_identity(base))
            and file_sha256(base) != receipt["sha256"]
        ):
            raise ValueError("Base face cache changed")
        target = args.output / "faces" / split
        target.mkdir(parents=True, exist_ok=True)
        if (target / "decoded-faces.json").exists():
            previous = json.loads((target / "decoded-faces.json").read_text())
            if previous.get("correction_sha256") != file_sha256(
                args.corrections / "corrections.json"
            ):
                raise ValueError("Existing output uses another correction")
            if file_sha256(target / "decoded-faces.npy") != previous["sha256"]:
                raise ValueError("Existing corrected face array changed")
            continue
        partial = target / "decoded-faces.npy.partial"
        if getattr(args, "sparse_faces", False):
            # The base stays immutable. Linking avoids a second 69 GiB local
            # copy; Drive publication can copy the same base server-side.
            if not partial.exists():
                os.link(base, partial)
            elif not os.path.samefile(base, partial):
                raise ValueError(
                    "Existing sparse base is not the verified original array"
                )
        else:
            shutil.copyfile(base, partial)
        values = np.load(partial, mmap_mode="r+")
        indices = np.load(directory / "row-indices.npy")
        replacements = np.load(directory / "face-pixels.npy", mmap_mode="r")
        if (
            values.shape != tuple(receipt["shape"])
            or len(indices) != record["changed_rows"]
            or replacements.shape != (len(indices), 224, 224, 3)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(values))
        ):
            raise ValueError("Invalid correction row alignment or shape")
        if not getattr(args, "sparse_faces", False):
            values[indices] = replacements
            values.flush()
        del values
        partial.replace(target / "decoded-faces.npy")
        corrected_sha = (
            receipt["sha256"]
            if getattr(args, "sparse_faces", False)
            else file_sha256(target / "decoded-faces.npy")
        )
        receipt.update(
            data=record["data"],
            sha256=corrected_sha,
            correction_sha256=file_sha256(args.corrections / "corrections.json"),
            base_array_sha256=receipt["sha256"],
        )
        receipt["array_identity"] = list(_array_identity(target / "decoded-faces.npy"))
        if getattr(args, "sparse_faces", False):
            for name in record["files"]:
                shutil.copyfile(directory / name, target / name)
            receipt["corrections"] = {
                "schema": "liveness-face-row-corrections-v1",
                "files": record["files"],
                "rows": len(indices),
                "base_array_scope": "Unchanged historical pixels; replacements applied before augmentation",
            }
        write_json(target / "decoded-faces.json", receipt)
        print(
            json.dumps(
                {
                    "split": split,
                    "corrected_rows": len(indices),
                    "array_sha256": receipt["sha256"],
                }
            ),
            flush=True,
        )

    source = args.geometric_cache
    receipt = json.loads((source / "geometric-cache.json").read_text())
    base = source / "geometric-targets.npy"
    if file_sha256(base) != receipt["array_sha256"]:
        raise ValueError("Base geometric cache changed")

    delta = args.corrections / "geometry"
    if (
        file_sha256(delta / "geometric-targets.npy")
        != correction["geometry_array_sha256"]
    ):
        raise ValueError("Corrected teacher values changed")
    teacher = json.loads((delta / "geometric-cache.json").read_text())
    for key in (
        "depth_checkpoint_sha256",
        "reflection_checkpoint_sha256",
        "mesh_sha256",
    ):
        if receipt["bindings"][key] != teacher["bindings"][key]:
            raise ValueError("Teacher changes require a separate correction")

    target = args.output / "geometry"
    target.mkdir(exist_ok=True)
    if (target / "geometric-cache.json").exists():
        raise FileExistsError("Corrected geometric cache already exists")
    partial = target / "geometric-targets.npy.partial"
    shutil.copyfile(base, partial)
    values = np.load(partial, mmap_mode="r+")
    indices = np.load(args.corrections / "train/row-indices.npy")
    values[indices] = np.load(delta / "geometric-targets.npy", mmap_mode="r")
    values.flush()
    del values
    partial.replace(target / "geometric-targets.npy")

    receipt["bindings"]["data"] = correction["splits"]["train"]["data"]
    receipt["bindings"]["source_sha256"] = teacher["bindings"]["source_sha256"]
    receipt.update(
        array_sha256=file_sha256(target / "geometric-targets.npy"),
        base_array_sha256=receipt["array_sha256"],
        correction_sha256=file_sha256(args.corrections / "corrections.json"),
    )
    write_json(target / "geometric-cache.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    for name in ("data-root", "audit", "pretrained"):
        build_parser.add_argument("--" + name, type=Path, required=True)
    build_parser.add_argument("--output", type=Path)
    apply_parser = commands.add_parser("apply")
    for name in ("corrections", "face-cache", "geometric-cache", "output"):
        apply_parser.add_argument("--" + name, type=Path, required=True)
    apply_parser.add_argument(
        "--sparse-faces",
        action="store_true",
        help="Reuse immutable base pixels and store only replacement rows",
    )

    args = parser.parse_args()
    (build if args.command == "build" else apply)(args)


if __name__ == "__main__":
    main()
