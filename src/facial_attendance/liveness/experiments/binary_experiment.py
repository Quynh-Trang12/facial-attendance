"""Assemble the bounded binary baseline from datasets, model and training steps."""

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    snapshot_sources,
    source_hashes,
    write_json,
)
from facial_attendance.liveness.datasets.selected_dataset import LivenessImages
from facial_attendance.liveness.evaluation.evaluation_metrics import (
    binary_metrics,
    family_metrics,
)
from facial_attendance.liveness.models.binary_model import (
    build_model,
    face_transform,
    load_checkpoint,
    save_checkpoint,
    state_version,
)
from facial_attendance.liveness.training.binary_training import train_binary
from facial_attendance.liveness.training.validation import (
    predict_dataset,
    save_predictions,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Set binary-baseline budget, optimizer settings and optional matched class exposure."""

    epochs: int = 10
    batch_size: int = 128
    workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 3
    seed: int = 42
    epoch_samples: int | None = None
    pos_weight: float | None = None
    epoch_live_samples: int | None = None


def run_training(
    root, inventory, readiness, output, config, *, smoke=False, resume=None
):
    """Assemble audited datasets and the binary baseline, then train and assess selection.

    Smoke stops before test. A full run opens test only after validation has fixed
    both checkpoint weights and threshold; existing output files are preserved.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; select a GPU runtime. No CPU fallback.")
    if min(config.epochs, config.batch_size, config.patience) < 1 or config.workers < 0:
        raise ValueError("Invalid training budget")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(output / "run.json")
    if shutil.disk_usage(output).free < 300 * 1024**2:
        raise ValueError("Need at least 300 MiB free for checkpoints and evaluation")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False

    train = LivenessImages(
        root, inventory, "train", transform=face_transform(True), audit_file=readiness
    )
    validation = LivenessImages(
        root, inventory, "validation", transform=face_transform(), audit_file=readiness
    )
    train_counts = {
        label: sum(r.binary_label == label for r in train.samples) for label in (0, 1)
    }
    if not all(train_counts.values()):
        raise ValueError("Training needs both live and spoof labels")
    if config.epoch_samples is not None and not 1 <= config.epoch_samples <= len(train):
        raise ValueError("Epoch sample budget must fit the training pool")
    positive_weight = (
        config.pos_weight
        if config.pos_weight is not None
        else train_counts[0] / train_counts[1]
    )
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("Positive loss weight must be finite and positive")
    if smoke:
        generator = torch.Generator().manual_seed(config.seed)
        train = Subset(
            train, torch.randperm(len(train), generator=generator)[:2048].tolist()
        )
        validation = Subset(
            validation,
            torch.randperm(len(validation), generator=generator)[:512].tolist(),
        )

    run = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "phase": "smoke"
        if smoke
        else ("quarantine_ablation" if config.epoch_samples else "baseline"),
        "training_config": asdict(config),
        "data": {
            "root": str(root.resolve()),
            "inventory": str(inventory.resolve()),
            "inventory_sha256": file_sha256(inventory),
            "readiness": str(readiness.resolve()),
            "readiness_sha256": file_sha256(readiness),
            "training_class_counts": train_counts,
            "training_pool": len(train),
            "train_used": min(config.epoch_samples or len(train), len(train)),
            "validation_used": len(validation),
        },
        "protocol": {
            "initialization": "random; weights=None; all parameters trained",
            "loss": "BCEWithLogitsLoss",
            "pos_weight": positive_weight,
            "sampling": "seeded subset without replacement per epoch, stratified when epoch_live_samples is set"
            if config.epoch_samples
            else "full training pool per epoch",
            "selection": "minimum validation ACER at its validation EER operating point; tie by validation BCE, then earlier epoch",
            "stopping": f"maximum {config.epochs} epochs; patience {config.patience}",
            "deployment_threshold": "validation EER threshold frozen with the selected checkpoint",
            "test": "once after weights and threshold are frozen; no test in smoke",
            "augmentation": "training-only random horizontal flip; no colour or JPEG augmentation",
            "test_exposure": "LIV-03 visually inspected and filtered source test images; this is a cleaned experimental subset, not an untouched benchmark",
        },
        "runtime": {
            "python": os.sys.version,
            "torch": str(torch.__version__),
            "torchvision": version("torchvision"),
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "gpu": torch.cuda.get_device_name(),
            "AMP": "float16",
            "shared_memory_bytes": os.statvfs("/dev/shm").f_bavail
            * os.statvfs("/dev/shm").f_frsize,
        },
        "source_sha256": source_hashes(),
    }
    write_json(output / "run.json", run)
    source_dir = output / "source"
    snapshot_sources(source_dir, run["source_sha256"])

    model = build_model().cuda()

    initial_version = state_version(model.state_dict())
    selection = train_binary(
        model,
        train,
        validation,
        config,
        run,
        output,
        positive_weight,
        initial_version,
        smoke=smoke,
        resume=resume,
    )
    if smoke:
        return selection

    best_epoch, updates = selection
    # Reload the selected checkpoint; final results never use the last epoch
    # accidentally, and the test split is opened only after selection finishes.
    selected, meta = load_checkpoint(output / "best.pt", "cuda")
    labels, scores, val_loss, _ = predict_dataset(selected, validation, config)
    threshold = meta["threshold"]
    save_checkpoint(output / "model.pt", selected.state_dict(), meta)
    save_predictions(output / "validation-predictions.csv", validation, labels, scores)
    results = {
        "validation": binary_metrics(labels, scores, threshold),
        "validation_BCE": val_loss,
    }
    deployed, deployed_meta = load_checkpoint(output / "model.pt", "cuda")

    test = LivenessImages(
        root, inventory, "test", transform=face_transform(), audit_file=readiness
    )
    labels, scores, test_loss, seconds = predict_dataset(deployed, test, config)
    save_predictions(output / "test-predictions.csv", test, labels, scores)
    results.update(
        test=binary_metrics(labels, scores, threshold),
        test_BCE=test_loss,
        test_by_attack=family_metrics(
            labels, scores, [r.attack_family for r in test.samples], threshold
        ),
        test_seconds=seconds,
        selected_epoch=best_epoch,
        threshold=threshold,
        model_version=deployed_meta["model_version"],
        checkpoint_sha256=file_sha256(output / "model.pt"),
        training_optimizer_updates=updates,
        selected_checkpoint_optimizer_updates=deployed_meta["optimizer_updates"],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        physical_presentation_tests="Not yet executed; requires recorded real/live/print/screen trials",
    )
    write_json(output / "evaluation.json", results)
    print(json.dumps(results), flush=True)
    return results
