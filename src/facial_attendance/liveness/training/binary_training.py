"""Update binary weights and preserve optimizer, scaler and selection state."""

import csv
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn

from facial_attendance.liveness.dataset_files import (
    compatible_sources,
    file_sha256,
    write_json,
)
from facial_attendance.liveness.evaluation.evaluation_metrics import binary_metrics
from facial_attendance.liveness.models.binary_model import (
    save_checkpoint,
    state_version,
)
from facial_attendance.liveness.training.batching import make_loader, training_subset
from facial_attendance.liveness.training.checkpoints import model_meta
from facial_attendance.liveness.training.validation import (
    predict_dataset,
    validation_checkpoint,
)


def train_epoch(model, dataset, optimizer, scaler, criterion, config, epoch):
    """Run one AMP binary epoch and report coverage and actual optimizer updates.

    Nonfinite losses stop the run. Scaler-skipped updates are counted separately.
    """
    model.train()
    total_loss, samples, updates, skipped = 0.0, 0, 0, 0
    started = time.monotonic()
    for step, (images, targets) in enumerate(
        make_loader(dataset, config, training=True, epoch=epoch), 1
    ):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images).reshape(-1)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in epoch {epoch}, step {step}")

        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        did_update = scaler.get_scale() >= scale
        updates += did_update
        skipped += not did_update
        total_loss += loss.item() * len(images)
        samples += len(images)

        if step % 200 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "samples": samples,
                        "weighted_BCE": total_loss / samples,
                    }
                ),
                flush=True,
            )

    torch.cuda.synchronize()
    if updates < 1 or samples != len(dataset):
        raise RuntimeError(
            "Training did not update weights or cover the requested samples"
        )
    return {
        "weighted_training_BCE": total_loss / samples,
        "train_samples": samples,
        "optimizer_updates": int(updates),
        "amp_skipped_steps": int(skipped),
        "training_seconds": time.monotonic() - started,
    }


def train_binary(
    model,
    train,
    validation,
    config,
    run,
    output,
    positive_weight,
    initial_version,
    *,
    smoke=False,
    resume=None,
):
    """Train to the declared budget and return selection state or a smoke report.

    Test data is deliberately absent: the experiment opens it only after the
    selected validation checkpoint and threshold have been frozen.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device="cuda")
    )

    best_key, best_epoch, stale, updates, start_epoch = (
        (float("inf"), float("inf")),
        0,
        0,
        0,
        1,
    )

    if resume:
        old = torch.load(
            Path(resume) / "last.pt", map_location="cpu", weights_only=True
        )
        for key in ("training_config", "data", "source_sha256"):
            same = (
                compatible_sources(old["run"][key], run[key])
                if key == "source_sha256"
                else old["run"][key] == run[key]
            )
            if not same:
                raise ValueError(f"Resume {key} differs from original run")
        model.load_state_dict(old["state_dict"])
        optimizer.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
        scaler.load_state_dict(old["scaler"])
        best_key, best_epoch, stale, updates = (
            tuple(old["best_key"]),
            old["best_epoch"],
            old["stale"],
            old["updates"],
        )
        start_epoch = old["epoch"] + 1
        torch.set_rng_state(old["torch_rng"])
        torch.cuda.set_rng_state_all(old["cuda_rng"])
        shutil.copyfile(Path(resume) / "best.pt", output / "best.pt")
        shutil.copyfile(Path(resume) / "history.csv", output / "history.csv")
        write_json(
            output / "resume.json",
            {
                "last_checkpoint": str((Path(resume) / "last.pt").resolve()),
                "last_sha256": file_sha256(Path(resume) / "last.pt"),
                "history_sha256": file_sha256(Path(resume) / "history.csv"),
                "first_resumed_epoch": start_epoch,
            },
        )

    history = []
    for epoch in range(start_epoch, (1 if smoke else config.epochs) + 1):
        lr = optimizer.param_groups[0]["lr"]
        budget = (
            min(config.epoch_samples, len(train))
            if smoke and config.epoch_samples
            else config.epoch_samples
        )
        epoch_train = training_subset(
            train,
            budget,
            config.seed,
            epoch,
            None if smoke else config.epoch_live_samples,
        )
        row = train_epoch(
            model, epoch_train, optimizer, scaler, criterion, config, epoch
        )
        updates += row["optimizer_updates"]

        labels, scores, val_loss, seconds = predict_dataset(model, validation, config)
        val = binary_metrics(labels, scores, 0.5)
        key, operating = validation_checkpoint(labels, scores, val_loss)
        row.update(
            epoch=epoch,
            learning_rate=lr,
            validation_BCE=val_loss,
            validation_ACER=val["ACER"],
            validation_ROC_AUC=val["ROC_AUC"],
            validation_seconds=seconds,
            validation_EER_threshold=operating["threshold"],
            validation_ACER_at_EER=operating["ACER"],
        )

        if val["ACER"] is None:
            raise ValueError("Validation needs both classes")
        if key < best_key:
            best_key, best_epoch, stale = key, epoch, 0
            save_checkpoint(
                output / "best.pt",
                model.state_dict(),
                model_meta(model, run, updates, epoch, operating["threshold"]),
            )
        else:
            stale += 1

        history.append(row)
        with (output / "history.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        scheduler.step()
        save_checkpoint(
            output / "last.pt",
            model.state_dict(),
            model_meta(model, run, updates, epoch, operating["threshold"]),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            best_key=list(best_key),
            best_epoch=best_epoch,
            stale=stale,
            updates=updates,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            run=run,
        )

        print(json.dumps(row), flush=True)
        if not smoke and stale >= config.patience:
            break

    if state_version(model.state_dict()) == initial_version:
        raise RuntimeError("No model weights changed")

    if smoke:
        row = history[-1]
        summary = {
            "phase": "smoke",
            "result": "PASS",
            "weights_changed": True,
            "training_samples": len(train),
            "validation_samples": len(validation),
            "optimizer_updates": updates,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "train_images_per_second": row["train_samples"] / row["training_seconds"],
            "validation_images_per_second": len(validation) / row["validation_seconds"],
            "epoch": row,
            "scope": "Real CUDA optimization/data-loading preflight; not final model effectiveness",
        }
        write_json(output / "smoke.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    return best_epoch, updates
