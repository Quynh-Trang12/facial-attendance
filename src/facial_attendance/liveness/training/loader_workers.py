"""Measure DataLoader worker counts on disposable model state before training."""

import time
from copy import deepcopy
from dataclasses import replace

import torch
from torch.utils.data import Subset

from facial_attendance.liveness.models.multitask_model import (
    GEOMETRIC_TASKS,
    SEMANTIC_TASKS,
)
from facial_attendance.liveness.training.batching import epoch_loader
from facial_attendance.liveness.training.multitask_training import train_multitask_epoch
from facial_attendance.liveness.training.validation import predict_dataset


class TrainingWindow(Subset):
    """Expose the sample metadata required by the existing seeded loader."""

    @property
    def samples(self):
        return self.dataset.samples[: len(self)]


def choose_loader_workers(training, validation, model, config, *, budget_seconds=120):
    """Measure two counts on disposable weights; never update the training model.

    The deadline is checked at batch boundaries. An in-flight CUDA operation or
    worker startup cannot be safely interrupted, so it may overrun the deadline.
    """
    counts = sorted({max(1, config.workers // 2), config.workers}, reverse=True)
    window = TrainingWindow(training, range(min(len(training), 5 * config.batch_size)))
    validation_window = Subset(
        validation, range(min(len(validation), 2 * config.batch_size))
    )
    started = time.monotonic()
    rows = []

    class BudgetReached(Exception):
        pass

    # The real model and optimizer are untouched; fork_rng restores GPU/CPU RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for workers in counts:
            probe = deepcopy(model)
            optimizer = torch.optim.SGD(
                probe.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
            candidate = replace(config, workers=workers)
            measured_start = None
            updates = 0

            def boundary(progress):
                nonlocal measured_start, updates
                torch.cuda.synchronize()
                updates = progress["optimizer_updates"]
                if updates == 1:
                    measured_start = time.monotonic()  # Exclude the warm-up update.
                if time.monotonic() - started >= budget_seconds:
                    raise BudgetReached

            try:
                train_multitask_epoch(
                    probe,
                    window,
                    optimizer,
                    candidate,
                    epoch=1,
                    auxiliary=True,
                    auxiliary_tasks=SEMANTIC_TASKS + GEOMETRIC_TASKS,
                    loader=epoch_loader(window, candidate, 1),
                    on_batch=boundary,
                )
                train_seconds = time.monotonic() - measured_start
                if time.monotonic() - started >= budget_seconds:
                    raise BudgetReached
                _, _, _, validation_seconds = predict_dataset(
                    probe.binary_model,
                    validation_window,
                    candidate,
                    autocast=False,
                    on_batch=lambda: boundary({"optimizer_updates": updates}),
                )
                estimate = train_seconds / ((updates - 1) * config.batch_size) * len(
                    training
                ) + validation_seconds / len(validation_window) * len(validation)
                rows.append(
                    {
                        "workers": workers,
                        "training_seconds": train_seconds,
                        "validation_seconds": validation_seconds,
                        "estimated_epoch_seconds": estimate,
                    }
                )
            except BudgetReached:
                break
            finally:
                del optimizer, probe
                torch.cuda.empty_cache()
    # A short deadline must not turn an incomplete measurement into a winner.
    selected = (
        min(rows, key=lambda row: row["estimated_epoch_seconds"])["workers"]
        if rows
        else config.workers
    )
    return {
        "schema": "liveness-loader-benchmark-v1",
        "gpu": torch.cuda.get_device_name(),
        "batch_size": config.batch_size,
        "candidate_workers": counts,
        "measurements": rows,
        "selected_workers": selected,
        "elapsed_seconds": time.monotonic() - started,
        "budget_seconds": budget_seconds,
        "deadline_policy": "Checked at batch boundaries; in-flight work may overrun",
        "selection": "Lowest measured train-plus-validation epoch estimate; CPU ceiling if no complete measurement",
        "precision": "FP32",
        "supervision": "All semantic and geometric tasks",
        "training_weights_unchanged": True,
    }
