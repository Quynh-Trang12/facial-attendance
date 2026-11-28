"""Create checkpoint metadata and preserve complete resumable epoch archives."""

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from facial_attendance.liveness.dataset_files import file_sha256, write_json
from facial_attendance.liveness.models.binary_model import (
    AENET_ARCHITECTURE,
    ARCHITECTURE,
    INPUT_SIZE,
    MEAN,
    RESNET_ARCHITECTURE,
    STD,
    normalization,
    save_checkpoint,
    state_version,
)
from facial_attendance.liveness.models.multitask_model import (
    ALL_SOURCE_TYPES,
    AuxiliaryLossWeights,
)
from facial_attendance.liveness.training.validation import save_predictions


def model_meta(model, run, updates, epoch, threshold=None):
    """Describe the binary baseline weights, recipe and selected validation threshold."""
    return {
        "schema": "liveness-checkpoint-v1",
        "model_id": "liveness-mobilenet-v3-small",
        "architecture": ARCHITECTURE,
        "model_version": state_version(model.state_dict()),
        "input_size": INPUT_SIZE,
        "mean": list(MEAN),
        "std": list(STD),
        "label_mapping": {"live": 0, "spoof": 1},
        "initialization": "random; weights=None",
        "optimizer_updates": updates,
        "epoch": epoch,
        "threshold": threshold,
        "threshold_source": "validation EER operating point"
        if threshold is not None
        else None,
        "data": run["data"],
        "training_config": run["training_config"],
    }


def checkpoint_metadata(model, run, *, epoch, updates, auxiliary, threshold):
    """Describe selected multitask weights without embedding optimizer resume state."""
    mean, std = normalization(RESNET_ARCHITECTURE)
    return {
        "schema": "liveness-checkpoint-v1",
        "model_id": "liveness-resnet18-binary-semantic"
        if auxiliary
        else "liveness-resnet18-binary-classification",
        "architecture": RESNET_ARCHITECTURE,
        "model_version": state_version(model.binary_model.state_dict()),
        "input_size": 224,
        "mean": list(mean),
        "std": list(std),
        "label_mapping": {"live": 0, "spoof": 1},
        "initialization": "ImageNet ResNet-18; reinitialized classification/auxiliary heads; all parameters fine-tuned",
        "initialization_sha256": run["initialization_sha256"],
        "epoch": epoch,
        "optimizer_updates": updates,
        "threshold": threshold,
        "threshold_source": "validation EER operating point",
        "threshold_validation": "Fixed clean+stress views of the same development images",
        "data": run["data"],
        "training_config": run["training_config"],
        "auxiliary_loss_weights": asdict(AuxiliaryLossWeights())
        if auxiliary
        else dict.fromkeys(asdict(AuxiliaryLossWeights()), 0.0),
        "source_sha256": run["source_sha256"],
        "contains_optimizer_state": False,
    }


def save_selected(directory, model, meta):
    """Save both deployment-only weights and the complete auxiliary-head state."""
    save_checkpoint(
        directory / "liveness-inference.pt", model.binary_model.state_dict(), meta
    )
    training_meta = {
        **meta,
        "architecture": meta.get(
            "training_architecture", "resnet18-binary-semantic-training-v1"
        ),
        "exported_binary_model_version": meta["model_version"],
        "model_version": state_version(model.state_dict()),
    }
    save_checkpoint(
        directory / "multitask-training.pt", model.state_dict(), training_meta
    )


def metadata(model, run, epoch, updates, threshold, auxiliary):
    """Describe the selected AENet binary export using the experiment-recorded tasks."""
    mean, std = normalization(AENET_ARCHITECTURE)
    return {
        "schema": "liveness-checkpoint-v1",
        "model_id": "liveness-resnet18-" + run["arm"],
        "architecture": AENET_ARCHITECTURE,
        "training_architecture": "resnet18-aenet-full-training-v1",
        "model_version": state_version(model.binary_model.state_dict()),
        "input_size": 224,
        "mean": list(mean),
        "std": list(std),
        "label_mapping": {"live": 0, "spoof": 1},
        "epoch": epoch,
        "optimizer_updates": updates,
        "threshold": threshold,
        "threshold_source": "validation EER operating point",
        "threshold_validation": "All declared validation samples",
        "phase": run["phase"],
        "inference_precision": "float32",
        "initialization": "ImageNet ResNet18; all parameters fine-tuned; task heads reinitialized",
        "initialization_sha256": run["initialization_sha256"],
        "source_types": list(ALL_SOURCE_TYPES),
        "auxiliary_enabled": auxiliary,
        "auxiliary_tasks": list(run["auxiliary_tasks"]),
        "loss_weights": run["loss_weights"],
        "training_config": run["training_config"],
        "data": run["data"],
        "geometry_cache_manifest_sha256": run["geometry_cache_manifest_sha256"],
        "source_sha256": run["source_sha256"],
    }


def archive_epoch_checkpoint(
    output, arm, epoch, metrics, inference_metadata, *, validation_predictions=None
):
    """Save one completed epoch for resume and later inference export.

    The .pt archive retains all model heads, optimizer state, random-generator states
    and the next training position. Its JSON receipt records SHA-256, validation
    selection information and, when supplied, the validation-prediction CSV checksum.
    Repeated attempts for this configuration reuse the same verified epoch archive.
    """
    directory = output.parent / (arm + "-epoch-checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"epoch-{epoch:03d}-training.pt"
    temporary = target.with_suffix(".pt.copying")
    shutil.copyfile(output / "last-training.pt", temporary)
    digest = file_sha256(temporary)
    if target.exists() and file_sha256(target) != digest:
        raise ValueError("Refusing to replace a different archived epoch checkpoint")
    temporary.replace(target)
    receipt = {
        "epoch": epoch,
        "checkpoint": target.name,
        "sha256": digest,
        "bytes": target.stat().st_size,
        "validation_and_training": metrics,
        "inference_metadata": inference_metadata,
        "weights_scope": "Complete multitask state_dict, including binary_model; optimizer and RNG included",
    }
    if validation_predictions is not None:
        dataset, labels, scores = validation_predictions
        predictions = directory / f"epoch-{epoch:03d}-validation.csv"
        # Commit scores only after their source/label alignment was checked.
        # Keep the receipt last so an interrupted write cannot certify a partial CSV.
        with tempfile.TemporaryDirectory(
            dir=directory, prefix="validation-"
        ) as temporary:
            pending = Path(temporary) / predictions.name
            save_predictions(pending, dataset, labels, scores)
            prediction_hash = file_sha256(pending)
            if predictions.exists() and file_sha256(predictions) != prediction_hash:
                raise ValueError(
                    "Refusing to replace different archived validation predictions"
                )
            pending.replace(predictions)
        receipt["validation_predictions"] = {
            "file": predictions.name,
            "sha256": prediction_hash,
            "bytes": predictions.stat().st_size,
            "samples": len(labels),
        }
    write_json(directory / f"epoch-{epoch:03d}.json", receipt)
