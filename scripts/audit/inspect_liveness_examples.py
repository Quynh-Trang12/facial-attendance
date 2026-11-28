"""Inspect selected validation examples with complete, verified multitask weights.

Weights and example identities are verified; replay must preserve archived
binary decisions. Numeric drift across CUDA runtimes is reported separately. This is bounded CUDA inference, without training or threshold changes.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import base64
import hashlib
import io
import json
import os
from datetime import UTC, datetime

import numpy as np
import torch

from facial_attendance.liveness.dataset_files import file_sha256  # noqa: E402
from facial_attendance.liveness.datasets.geometric_cache import generate_cache
from facial_attendance.liveness.datasets.reviewed_cache import ReviewedCacheView
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages
from facial_attendance.liveness.models.binary_model import (
    AENET_ARCHITECTURE,
    build_model,
    face_transform,
    state_version,
)
from facial_attendance.liveness.models.multitask_model import (
    ALL_SOURCE_TYPES,
    MultitaskLivenessModel,
)


class SelectedImages:
    """Keep selected rows in inventory order while reusing verified source decoding."""

    def __init__(self, source, names, positions):
        self.source = source
        self.indices = [positions[name] for name in names]
        self.names = names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]

    def bindings(self):
        return {
            **self.source.bindings(),
            "samples": len(self),
            "selected_paths_sha256": hashlib.sha256(
                json.dumps(self.names).encode()
            ).hexdigest(),
        }


def inspect_examples(project, inventory, target_output, *, checkpoint_root=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU inference fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = project / "data/raw/liveness/celeba-spoof-full"
    source = ReproductionImages(
        root,
        root / "audits/current-schema-2026-11-12/inventory.csv",
        root / "project_validation.json",
    )
    positions = {row.relative_path: i for i, row in enumerate(source.samples)}
    names = inventory["examples"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Examples must be nonempty unique validation paths")
    if inventory.get("cache_view"):
        view_path = project / inventory["cache_view"]
        if file_sha256(view_path) != inventory["reviewed_cache_view_sha256"]:
            raise ValueError("Example selection uses a changed cache view")
        view = ReviewedCacheView(source, view_path.parent, "validation")
        if not set(names) <= {row.relative_path for row in view.samples}:
            raise ValueError("Example gallery includes an excluded validation image")
    transform = face_transform(architecture=AENET_ARCHITECTURE)
    images, examples = [], {}
    for name in names:
        image, target = source[positions[name]]
        image = transform.transforms[0](image)
        pixels = io.BytesIO()
        image.save(pixels, format="PNG")
        images.append(transform(image))
        examples[name] = {
            "crop_png": base64.b64encode(pixels.getvalue()).decode("ascii"),
            "true_binary": target["binary"],
            "true_source_type": ALL_SOURCE_TYPES[target["spoof_type"]],
            "true_illumination": target["illumination"],
            "true_face_attributes": target["face_attributes"].tolist(),
            "models": {},
        }
    inputs = torch.stack(images).cuda()
    annotation_sha = file_sha256(source.annotations)
    inventory_sha = file_sha256(source.inventory)
    provenance = {}
    for arm, record in inventory["models"].items():
        checkpoint = (
            checkpoint_root / (arm + ".pt")
            if checkpoint_root
            else project / record["checkpoint"]
        )
        if file_sha256(checkpoint) != record["training_sha256"]:
            raise ValueError("Complete training checkpoint changed: " + arm)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        meta, state = saved["meta"], saved["state_dict"]
        run = saved["run"]
        if (
            meta["epoch"] != record["epoch"]
            or run["arm"] != arm
            or not 0 <= record["threshold"] <= 1
            or run["data"]["validation"]["annotations_sha256"] != annotation_sha
            or run["data"]["inventory_sha256"] != inventory_sha
        ):
            raise ValueError("Checkpoint/example provenance mismatch: " + arm)
        geometry = "depth_map.weight" in state
        model = MultitaskLivenessModel(
            build_model(AENET_ARCHITECTURE),
            source_types=ALL_SOURCE_TYPES,
            geometry=geometry,
        )
        model.load_state_dict(state, strict=True)
        if state_version(model.binary_model.state_dict()) != record["model_version"]:
            raise ValueError("Multitask binary weights differ from selection: " + arm)
        model.cuda().eval()
        tasks = tuple(run["auxiliary_tasks"])
        with torch.inference_mode():
            output = model.forward_tasks(inputs, tasks=tasks)
            scores = output["binary"].float().sigmoid().cpu()
        expected = torch.tensor([record["scores"][name] for name in names])
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite replay score: " + arm)
        # CUDA builds/devices can change FP32 kernels. Preserve archived scores
        # and record numeric drift; require the illustrated decisions to agree.
        replay_decisions = scores >= record["threshold"]
        archived_decisions = expected >= record["threshold"]
        if not torch.equal(replay_decisions, archived_decisions):
            raise ValueError("Replay changed an illustrated binary decision: " + arm)
        provenance[arm] = {
            "training_sha256": record["training_sha256"],
            "inference_sha256": record["inference_sha256"],
            "model_version": record["model_version"],
            "epoch": meta["epoch"],
            "threshold": record["threshold"],
            "tasks": list(tasks),
            "max_binary_score_difference": float((scores - expected).abs().max()),
        }
        for i, name in enumerate(names):
            result = {"spoof_score": float(scores[i])}
            for task in tasks:
                value = output[task][i].float().detach().cpu()
                if task == "face_attributes":
                    value = value.sigmoid()
                elif task in ("spoof_type", "illumination"):
                    value = value.softmax(dim=0)
                result[task] = value.tolist()
            examples[name]["models"][arm] = result
        del model, saved, state, output
        torch.cuda.empty_cache()
    teachers = project / "artifacts/pretrained"
    training_geometry = json.loads(
        (project / "data/liveness/face-cache/geometry/geometric-cache.json").read_text()
    )
    depth_teacher = teachers / "depth-teacher.pt"
    reflection_teacher = teachers / "reflection-teacher.pt"
    for teacher, key in (
        (depth_teacher, "depth_checkpoint_sha256"),
        (reflection_teacher, "reflection_checkpoint_sha256"),
    ):
        if file_sha256(teacher) != training_geometry["bindings"][key]:
            raise ValueError("Teacher differs from the training target generator")
    target_receipt = generate_cache(
        SelectedImages(source, names, positions),
        depth_teacher,
        reflection_teacher,
        teachers / "prnet-tensorflow",
        target_output,
    )
    targets = np.load(target_output / "geometric-targets.npy", mmap_mode="r")
    for index, name in enumerate(names):
        examples[name]["pseudo_depth"] = targets[index, :1].tolist()
        examples[name]["pseudo_reflection"] = targets[index, 1:].tolist()
    return {
        "schema": "liveness-selected-examples-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(),
        "precision": "float32",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "split": "validation",
        "selection": inventory["selection"],
        "reviewed_cache_view_sha256": inventory.get("reviewed_cache_view_sha256"),
        "annotations_sha256": file_sha256(source.annotations),
        "inventory_sha256": file_sha256(source.inventory),
        "geometric_reference": "Teacher-derived pseudo-targets recomputed for selected validation examples; not physical ground truth",
        "pseudo_target_receipt": target_receipt,
        "models": provenance,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--inventory", "--manifest", dest="inventory", type=Path, required=True
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    output = args.output or (
        Path(os.environ["GPU_JOB_OUTPUT"]) / "selected-examples.json"
        if "GPU_JOB_OUTPUT" in os.environ
        else None
    )

    if output is None:
        parser.error("--output is required unless GPU_JOB_OUTPUT is set")
    if output.exists():
        raise FileExistsError("Preserve the existing report; choose a new output path")

    result = inspect_examples(
        args.project.resolve(),
        json.loads(args.inventory.read_text()),
        output.parent / "example-pseudo-targets",
        checkpoint_root=args.checkpoint_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "examples": len(result["examples"]),
                "models": list(result["models"]),
                "device": result["device"],
            }
        )
    )


if __name__ == "__main__":
    main()
