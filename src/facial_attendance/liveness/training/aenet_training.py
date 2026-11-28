"""Run AENet epochs and restore/save exact optimization state."""

import json
import shutil
import time
from pathlib import Path

import torch

from facial_attendance.liveness.dataset_files import (
    compatible_sources,
    file_sha256,
    write_json,
)
from facial_attendance.liveness.models.binary_model import (
    load_checkpoint,
    save_checkpoint,
)
from facial_attendance.liveness.training.batching import (
    epoch_loader,
    epoch_training_config,
    seed_training,
)
from facial_attendance.liveness.training.checkpoints import (
    archive_epoch_checkpoint,
    metadata,
    save_selected,
)
from facial_attendance.liveness.training.multitask_training import train_multitask_epoch
from facial_attendance.liveness.training.validation import (
    predict_dataset,
    validation_checkpoint,
)


def train_aenet(
    model,
    training,
    validation_images,
    config,
    run,
    output,
    *,
    arm,
    tasks,
    resume=None,
    checkpoint_seconds=600,
    preflight=False,
    screening=False,
    retain_epoch_checkpoints=False,
):
    """Return run results after validation-based selection and resumable epochs.

    Dataset membership and model construction are already fixed by the experiment.
    Resume rejects changed inputs before restoring weights, optimizer and RNG.
    """
    seed_training(config.seed)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    updates, start_epoch, best_key, best_epoch = 0, 1, (float("inf"), float("inf")), 0
    resume_progress = None
    resume_rng = None

    if resume:
        resumed = torch.load(
            Path(resume) / "last-training.pt", map_location="cpu", weights_only=True
        )
        for key in (
            "phase",
            "arm",
            "training_config",
            "data",
            "geometry_cache_manifest_sha256",
            "initialization_sha256",
            "source_sha256",
        ):
            same = (
                compatible_sources(resumed["run"][key], run[key])
                if key == "source_sha256"
                else resumed["run"][key] == run[key]
            )
            if not same:
                raise ValueError(f"Resume input/protocol changed: {key}")
        # Restore the full optimization state; a weights-only reload is not a continuation.
        model.load_state_dict(resumed["state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        updates = resumed["updates"]
        start_epoch = resumed.get("resume_epoch", resumed["epoch"] + 1)
        resume_progress = resumed.get("progress")
        resume_rng = (resumed.get("torch_rng"), resumed.get("cuda_rng"))
        best_key, best_epoch = tuple(resumed["best_key"]), resumed["best_epoch"]
        for name in (
            "liveness-inference.pt",
            "multitask-training.pt",
            "training-history.jsonl",
            "last-training.pt",
        ):
            if (Path(resume) / name).exists():
                shutil.copyfile(Path(resume) / name, output / name)

    if not (output / "training-history.jsonl").exists():
        (output / "training-history.jsonl").touch()
    started = time.monotonic()
    auxiliary = bool(tasks)
    last_checkpoint = time.monotonic()

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_config = epoch_training_config(config, epoch)
        seed_training(config.seed + epoch)

        # Restore RNG after epoch seeding, before the first remaining augmentation.
        if resume_rng is not None and resume_rng[0] is not None:
            torch.set_rng_state(resume_rng[0])
            torch.cuda.set_rng_state_all(resume_rng[1])

        partial = resume_progress if epoch == start_epoch else None
        base_updates = updates - (partial["optimizer_updates"] if partial else 0)

        def persist(progress, *, next_epoch=False):
            nonlocal last_checkpoint
            current_updates = (
                base_updates + progress["optimizer_updates"] if progress else updates
            )
            save_checkpoint(
                output / "last-training.pt",
                model.state_dict(),
                {"architecture": "resnet18-aenet-full-training-v1", "epoch": epoch},
                optimizer=optimizer.state_dict(),
                updates=current_updates,
                epoch=epoch,
                resume_epoch=epoch + 1 if next_epoch else epoch,
                progress=None if next_epoch else progress,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                best_key=list(best_key),
                best_epoch=best_epoch,
                run=run,
            )
            last_checkpoint = time.monotonic()

        # The callback runs after a completed update, so its cursor is safe to replay.
        def batch_boundary(progress):
            if time.monotonic() - last_checkpoint >= checkpoint_seconds:
                persist(progress)

        row = train_multitask_epoch(
            model,
            training,
            optimizer,
            epoch_config,
            epoch=epoch,
            auxiliary=auxiliary,
            loader=epoch_loader(
                training, epoch_config, epoch, partial["samples"] if partial else 0
            ),
            progress=partial,
            on_batch=batch_boundary,
            auxiliary_tasks=tasks,
        )
        updates = base_updates + row["optimizer_updates"]
        complete_progress = {
            "samples": row["samples"],
            "optimizer_updates": row["optimizer_updates"],
            "totals": {
                name: value * row["samples"] for name, value in row["losses"].items()
            },
            "gradient_proof": row["gradient_proof"],
        }
        persist(complete_progress)

        labels, scores, loss, seconds = predict_dataset(
            model.binary_model,
            validation_images,
            epoch_config,
            autocast=False,
        )
        key, metrics = validation_checkpoint(labels, scores, loss)
        row.update(
            epoch=epoch,
            batch_size=epoch_config.batch_size,
            validation=metrics,
            validation_BCE=loss,
            validation_seconds=seconds,
            elapsed_seconds=time.monotonic() - started,
        )

        if key < best_key:
            best_key, best_epoch = key, epoch
            save_selected(
                output,
                model,
                metadata(model, run, epoch, updates, metrics["threshold"], auxiliary),
            )

        with (output / "training-history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        persist(None, next_epoch=True)
        if retain_epoch_checkpoints:
            archive_epoch_checkpoint(
                output,
                arm,
                epoch,
                row,
                metadata(model, run, epoch, updates, metrics["threshold"], auxiliary),
                validation_predictions=(validation_images, labels, scores),
            )

        resume_progress, resume_rng = None, None
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "total_updates": updates,
                    "train_samples": row["samples"],
                    "validation_ACER": metrics["ACER"],
                }
            ),
            flush=True,
        )

    deployed, meta = load_checkpoint(output / "liveness-inference.pt", "cuda")
    result = {
        "phase": run["phase"],
        "arm": arm,
        "completed_epochs": config.epochs,
        "selected_epoch": best_epoch,
        "optimizer_updates": updates,
        "selected_optimizer_updates": meta["optimizer_updates"],
        "selection_key": list(best_key),
        "checkpoint_sha256": file_sha256(output / "liveness-inference.pt"),
        "threshold": meta["threshold"],
        "elapsed_seconds_this_execution": time.monotonic() - started,
        "full_training_completed": not preflight and not screening,
        "screening_completed": screening,
        "epoch_checkpoint_directory": str(output.parent / (arm + "-epoch-checkpoints"))
        if retain_epoch_checkpoints
        else None,
        "test_or_physical_used_for_selection": False,
    }

    write_json(output / "training-results.json", result)
    return result
