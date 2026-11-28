"""Assemble the 15-epoch AENet comparison and continuation to epoch 25."""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    snapshot_sources,
    source_hashes,
    write_json,
)
from facial_attendance.liveness.datasets.decoded_cache import DecodedFaces
from facial_attendance.liveness.datasets.geometric_cache import load_geometric_cache
from facial_attendance.liveness.datasets.image_augmentation import (
    DeviceColorParameters,
    GeometricColorAugmentation,
)
from facial_attendance.liveness.datasets.reviewed_cache import ReviewedCacheView
from facial_attendance.liveness.datasets.source_dataset import (
    BinaryReproductionImages,
    GeometricTrainingImages,
    ReproductionImages,
    check_reproduction_partitions,
)
from facial_attendance.liveness.models.binary_model import (
    AENET_ARCHITECTURE,
    RESNET_INITIALIZATION_SHA256,
    build_model,
    face_transform,
)
from facial_attendance.liveness.models.multitask_model import (
    ALL_SOURCE_TYPES,
    GEOMETRIC_TASKS,
    SEMANTIC_TASKS,
    AuxiliaryLossWeights,
    MultitaskLivenessModel,
)
from facial_attendance.liveness.training.aenet_training import train_aenet
from facial_attendance.liveness.training.batching import (
    SeededTrainingImages,
    epoch_training_config,
    seed_training,
)
from facial_attendance.liveness.training.loader_workers import choose_loader_workers

SUPERVISION_VARIANTS = {
    "binary-classification": (),
    "aenet-full": SEMANTIC_TASKS + GEOMETRIC_TASKS,
    "binary-semantic": SEMANTIC_TASKS,
    "aenet-depth": ("depth_map",),
    "aenet-reflection": ("reflection_map",),
    "binary-geometric": GEOMETRIC_TASKS,
    "binary-semantic-without-face-attributes": ("spoof_type", "illumination"),
    "binary-semantic-without-spoof-type": ("face_attributes", "illumination"),
    "binary-semantic-without-illumination": ("face_attributes", "spoof_type"),
    "binary-semantic-depth": SEMANTIC_TASKS + ("depth_map",),
    "binary-semantic-reflection": SEMANTIC_TASKS + ("reflection_map",),
}


@dataclass(frozen=True)
class FullTrainingConfig:
    """Set the 15-epoch AENet recipe or explicit continuation target of epoch 25."""

    epochs: int = 15
    batch_size: int = 64
    workers: int = 4
    large_batch_size: int | None = None
    large_batch_start_epoch: int = 1
    gpu_augmentation: bool = False
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0001
    seed: int = 42
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.05

    def validate(self, *, preflight=False, screening=False):
        """Reject undeclared epoch budgets and invalid batch/optimizer/augmentation settings."""
        if self.epochs < 1 or (not preflight and self.epochs not in (15, 25)):
            raise ValueError(
                "The submitted protocol uses 15 or 25 epochs; shorter checks require explicit --preflight"
            )
        if self.large_batch_size is not None and (
            self.large_batch_size < self.batch_size
            or not 1 <= self.large_batch_start_epoch <= self.epochs
        ):
            raise ValueError("Invalid larger-batch epoch schedule")
        if self.batch_size < 2 or self.workers < 0:
            raise ValueError("Batch size must be >=2 and workers nonnegative")
        if (
            self.learning_rate != 0.005
            or not 0 <= self.momentum < 1
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError(
                "Use the published SGD learning rate .005 and valid explicit optimizer settings"
            )
        GeometricColorAugmentation(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


def run_full_training(
    root,
    inventory,
    train_annotations,
    validation_annotations,
    cache_directory,
    imagenet_weights,
    output,
    config,
    *,
    arm,
    preflight=False,
    resume=None,
    checkpoint_seconds=600,
    decoded_cache=None,
    reviewed_cache_view=None,
    screening=False,
    retain_epoch_checkpoints=False,
    benchmark_receipt=None,
):
    """Assemble source images, frozen selection, teacher maps and enabled model heads.

    Cache/data/initialization checks finish before optimization. The training engine
    owns epoch updates, validation selection and resumable checkpoint state.
    """
    config.validate(preflight=preflight, screening=screening)
    if checkpoint_seconds <= 0:
        raise ValueError("Checkpoint interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; select a GPU runtime. No CPU fallback.")
    if file_sha256(imagenet_weights) != RESNET_INITIALIZATION_SHA256:
        raise ValueError("ImageNet initialization checksum mismatch")
    if arm not in SUPERVISION_VARIANTS:
        raise ValueError("Unsupported supervision variant: " + arm)

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training-config.json").exists():
        raise FileExistsError(output / "training-config.json")
    if shutil.disk_usage(output).free < 700 * 1024**2:
        raise ValueError(
            "Need at least 700 MiB for selected/resumable training checkpoints"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    train = ReproductionImages(root, inventory, train_annotations)
    validation = ReproductionImages(root, inventory, validation_annotations)
    check_reproduction_partitions(train, validation)
    if decoded_cache is not None:
        train = DecodedFaces(train, Path(decoded_cache) / "train")
        validation = DecodedFaces(validation, Path(decoded_cache) / "validation")

    maps, cache_receipt = load_geometric_cache(cache_directory, train)
    if reviewed_cache_view is not None:
        # Verify geometric maps against their original cache before applying
        # identical row indices to images, labels and teacher targets.
        train = ReviewedCacheView(train, reviewed_cache_view, "train")
        validation = ReviewedCacheView(validation, reviewed_cache_view, "validation")
        maps = train.select_maps(maps)

    augmentation = (
        DeviceColorParameters if config.gpu_augmentation else GeometricColorAugmentation
    )
    training = GeometricTrainingImages(
        train,
        maps,
        augmentation(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        ),
    )
    training = SeededTrainingImages(training, config.seed)
    validation_images = BinaryReproductionImages(
        validation, face_transform(architecture=AENET_ARCHITECTURE)
    )

    # Construct every supervision arm from the same initialization RNG state.
    seed_training(config.seed)
    binary = build_model(AENET_ARCHITECTURE, imagenet_weights=imagenet_weights).cuda()
    model = MultitaskLivenessModel(
        binary, source_types=ALL_SOURCE_TYPES, geometry=True
    ).cuda()

    if benchmark_receipt is not None:
        if resume:
            raise ValueError("Loader selection is only allowed before a new comparison")
        measured = choose_loader_workers(
            training, validation_images, model, epoch_training_config(config, 1)
        )
        benchmark_receipt.parent.mkdir(parents=True, exist_ok=True)
        write_json(benchmark_receipt, measured)
        config = replace(config, workers=measured["selected_workers"])
        print(
            json.dumps(
                {"phase": "loader-selection", "selected_workers": config.workers}
            ),
            flush=True,
        )

    run = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "phase": "screening"
        if screening
        else "preflight"
        if preflight
        else "full-training",
        "arm": arm,
        "auxiliary_tasks": list(SUPERVISION_VARIANTS[arm]),
        "loss_weights": {
            "binary": 1.0,
            **{
                name: asdict(AuxiliaryLossWeights())[name]
                for name in SUPERVISION_VARIANTS[arm]
            },
        },
        "training_config": asdict(config),
        "initialization_sha256": file_sha256(imagenet_weights),
        "data": {
            "root": str(root.resolve()),
            "inventory": str(inventory.resolve()),
            "inventory_sha256": file_sha256(inventory),
            "train": train.bindings(),
            "validation": validation.bindings(),
            "test_annotations": str(
                (root / "metas/intra_test/test_label.json").resolve()
            ),
            "test_annotations_sha256": file_sha256(
                root / "metas/intra_test/test_label.json"
            ),
            "train_samples_per_epoch": len(train),
            "validation_samples": len(validation),
        },
        "geometry_cache_manifest_sha256": file_sha256(
            Path(cache_directory) / "geometric-cache.json"
        ),
        "geometry_source_bindings": cache_receipt["bindings"],
        "protocol": {
            "paper": "https://arxiv.org/abs/2007.12342",
            "epochs": config.epochs,
            "training": "Every declared train image once per epoch; FP32 SGD/validation/inference, constant .005 learning rate; no AMP skipped steps; per-image/epoch seeded color augmentation for exact mid-epoch replay",
            "comparison": "Shared initialization, split assignment, augmentation and epochs across the compared supervision configurations",
            "selection": "Minimum validation ACER at the validation EER operating point; tie by binary loss then earlier epoch; frozen validation threshold",
            "stopping": f"Exactly {config.epochs} epochs; validation-only screening"
            if screening
            else f"Exactly {config.epochs} epochs; no patience or artificial wall-clock stop",
            "validation_membership": "Existing project-derived subject split, not an available official validation split assignment",
            "unreported_recipe_choices": "Explicit batch size, momentum, weight decay, color strengths and seed are project choices; released inference ToTensor range adopted",
            "test_exposure": "Existing public test has prior experimental exposure; this training entry point never opens test or physical clips",
            "pseudo_label_limits": "Teacher TensorFlow parity is recorded separately in teacher-verification; sampled parity does not establish the unpublished AENet projection recipe",
        },
        "runtime": {
            "python": os.sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "checkpoint_seconds": checkpoint_seconds,
        "source_sha256": source_hashes(),
    }

    write_json(output / "training-config.json", run)
    source_output = output / "source"
    # Snapshot the package tree so later workspace edits cannot alter resume inputs.
    snapshot_sources(source_output, run["source_sha256"])

    return train_aenet(
        model,
        training,
        validation_images,
        config,
        run,
        output,
        arm=arm,
        tasks=SUPERVISION_VARIANTS[arm],
        resume=resume,
        checkpoint_seconds=checkpoint_seconds,
        preflight=preflight,
        screening=screening,
        retain_epoch_checkpoints=retain_epoch_checkpoints,
    )
