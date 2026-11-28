"""Assemble the bounded semantic comparison and its recorded cohort."""

import csv
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    snapshot_sources,
    source_hashes,
    write_json,
)
from facial_attendance.liveness.datasets.image_augmentation import CaptureStressView
from facial_attendance.liveness.datasets.selected_dataset import LivenessImages
from facial_attendance.liveness.datasets.semantic_dataset import (
    MultitaskImages,
    balanced_cohort,
    read_annotation_targets,
)
from facial_attendance.liveness.models.binary_model import (
    RESNET_ARCHITECTURE,
    RESNET_INITIALIZATION_SHA256,
    build_model,
    face_transform,
    state_version,
)
from facial_attendance.liveness.models.multitask_model import MultitaskLivenessModel
from facial_attendance.liveness.training.batching import seed_training
from facial_attendance.liveness.training.multitask_training import (
    train_multitask_comparison,
)


@dataclass(frozen=True)
class MultitaskTrainingConfig:
    """Set the bounded semantic experiment budget and cohort/optimizer settings."""

    epochs: int = 6
    batch_size: int = 64
    workers: int = 2
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0001
    patience: int = 3
    seed: int = 42
    epoch_samples: int = 3840
    validation_samples: int = 960
    max_seconds: int = 900

    def validate(self):
        if (
            min(self.epochs, self.batch_size, self.patience, self.max_seconds) < 1
            or self.workers < 0
        ):
            raise ValueError("Invalid training/resource budget")
        if any(n < 48 or n % 48 for n in (self.epoch_samples, self.validation_samples)):
            raise ValueError("Image budgets must be positive multiples of 48")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not 0 <= self.momentum < 1
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid SGD hyperparameters")


def write_split_assignment(path, train, validation, train_indices, validation_indices):
    """Record selected source-row identities so matched cohorts can be audited later."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["role", "epoch", "relative_path", "binary_label", "source_type"]
        )
        for role, epoch, dataset, indices in [
            ("validation", 0, validation, validation_indices)
        ] + [
            ("train", epoch, train, indices)
            for epoch, indices in enumerate(train_indices, 1)
        ]:
            for index in indices:
                row = dataset.samples[index]
                writer.writerow(
                    [role, epoch, row.relative_path, row.binary_label, row.source_type]
                )


def run_multitask_training(
    root,
    inventory,
    readiness,
    train_annotations,
    validation_annotations,
    imagenet_weights,
    output,
    config,
):
    """Build matched binary-control and semantic configurations on the same cohorts.

    Initialization, row selection and augmentations are seeded identically.
    Validation alone selects each configuration; this experiment never opens test.
    """
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; select a GPU runtime. No CPU fallback.")
    if file_sha256(imagenet_weights) != RESNET_INITIALIZATION_SHA256:
        raise ValueError("Expected official torchvision ResNet18 IMAGENET1K_V1 weights")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training-config.json").exists():
        raise FileExistsError(output / "training-config.json")
    if shutil.disk_usage(output).free < 600 * 1024**2:
        raise ValueError("Need at least 600 MiB free for the paired checkpoints")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False

    train = LivenessImages(root, inventory, "train", audit_file=readiness)
    validation = LivenessImages(root, inventory, "validation", audit_file=readiness)
    train_targets = read_annotation_targets(train, train_annotations, readiness)
    validation_targets = read_annotation_targets(
        validation, validation_annotations, readiness
    )
    validation_indices = balanced_cohort(
        validation, validation_targets, config.validation_samples, config.seed + 1000
    )
    train_indices = [
        balanced_cohort(train, train_targets, config.epoch_samples, config.seed + epoch)
        for epoch in range(1, config.epochs + 1)
    ]
    clean = MultitaskImages(
        validation,
        validation_indices,
        validation_targets,
        face_transform(architecture=RESNET_ARCHITECTURE),
    )
    stress = MultitaskImages(
        validation,
        validation_indices,
        validation_targets,
        CaptureStressView(RESNET_ARCHITECTURE),
    )

    run = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "training_config": asdict(config),
        "initialization_sha256": file_sha256(imagenet_weights),
        "data": {
            "root": str(root.resolve()),
            "inventory": str(inventory.resolve()),
            "inventory_sha256": file_sha256(inventory),
            "readiness": str(readiness.resolve()),
            "readiness_sha256": file_sha256(readiness),
            "training_pool": len(train),
            "train_used_per_epoch": config.epoch_samples,
            "validation_pool": len(validation),
            "validation_used": len(validation_indices),
            "validation_views_per_image": 2,
            "train_annotations": str(train_annotations.resolve()),
            "train_annotations_sha256": file_sha256(train_annotations),
            "validation_annotations": str(validation_annotations.resolve()),
            "validation_annotations_sha256": file_sha256(validation_annotations),
        },
        "protocol": {
            "paper": "https://arxiv.org/abs/2007.12342",
            "comparison": "Matched ResNet18 binary-only vs binary + 40 face attributes + 7 scoped attacks + 5 illuminations",
            "loss": "CE(binary) + 1*BCE(face attributes) + .1*CE(attack) + .01*CE(illumination)",
            "optimization": "FP32 SGD; constant learning rate; no skipped updates; all parameters trainable",
            "sampling": "Same seeded image order/augmentation per arm; 50% live, 50% balanced attack/illumination cells",
            "selection": "Minimum pooled clean+stress validation EER ACER; tie by binary CE then earlier epoch; exact arm tie keeps control",
            "stopping": "Stop both arms together at patience, epoch cap or resource budget; never use test or physical clips for selection",
            "adaptations": "Bounded project print/screen subset; audited source boxes; camera augmentation; ImageNet normalization; no geometric/temporal branches",
            "test_exposure": "Existing experimental test split previously evaluated; this run does not open it",
        },
        "runtime": {
            "python": os.sys.version,
            "torch": str(torch.__version__),
            "torchvision": version("torchvision"),
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "gpu": torch.cuda.get_device_name(),
        },
        "source_sha256": source_hashes(),
    }
    write_json(output / "training-config.json", run)
    source = output / "source"
    snapshot_sources(source, run["source_sha256"])
    write_split_assignment(
        output / "split-assignment.csv",
        train,
        validation,
        train_indices,
        validation_indices,
    )

    states = {}
    for name, auxiliary in (
        ("binary-classification", False),
        ("binary-semantic", True),
    ):
        seed_training(config.seed)
        binary = build_model(
            RESNET_ARCHITECTURE, imagenet_weights=imagenet_weights
        ).cuda()
        model = MultitaskLivenessModel(binary).cuda()
        directory = output / name
        directory.mkdir()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        states[name] = {
            "model": model,
            "optimizer": optimizer,
            "auxiliary": auxiliary,
            "initial_state_version": state_version(model.state_dict()),
            "initial_auxiliary_version": state_version(
                {
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith("binary_model.")
                }
            ),
            "best_key": (float("inf"), float("inf")),
            "stale": 0,
            "updates": 0,
            "history": [],
        }
    if len({state["initial_state_version"] for state in states.values()}) != 1:
        raise RuntimeError("Paired models have different initial weights")

    return train_multitask_comparison(
        states,
        train,
        train_targets,
        train_indices,
        clean,
        stress,
        config,
        run,
        output,
        started,
    )
