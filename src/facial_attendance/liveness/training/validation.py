"""Score aligned datasets and select an operating point using validation only."""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from facial_attendance.liveness.evaluation.evaluation_metrics import (
    binary_metrics,
    validation_threshold,
)
from facial_attendance.liveness.training.batching import make_loader, sample_rows


@torch.inference_mode()
def predict_dataset(model, dataset, config, *, autocast=True, on_batch=None):
    """Return ordered labels, spoof scores, mean binary loss and elapsed seconds.

    Evaluation has no gradients and must cover every declared dataset row.
    """
    model.eval()
    labels, scores, loss_sum = [], [], 0.0
    started = time.monotonic()
    for images, targets in make_loader(dataset, config):
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            logits = model(images).reshape(-1)
        targets_gpu = targets.cuda(non_blocking=True).float()
        loss_sum += nn.functional.binary_cross_entropy_with_logits(
            logits.float(), targets_gpu, reduction="sum"
        ).item()
        labels.extend(targets.tolist())
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        if on_batch is not None:
            on_batch()
    torch.cuda.synchronize()
    if len(labels) != len(dataset):
        raise RuntimeError("Evaluation coverage differs from dataset")
    return (
        np.asarray(labels),
        np.asarray(scores),
        loss_sum / len(dataset),
        time.monotonic() - started,
    )


def save_predictions(path, dataset, labels, scores):
    """Write scores with inventory identities after checking coverage and label order."""
    rows = sample_rows(dataset)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_path",
                "split",
                "source_type",
                "attack_family",
                "binary_label",
                "spoof_score",
            ]
        )
        for row, label, score in zip(rows, labels, scores, strict=True):
            if int(label) != row.binary_label:
                raise ValueError("Prediction/source target alignment failed")
            writer.writerow(
                [
                    row.relative_path,
                    row.split,
                    row.source_type,
                    row.attack_family,
                    int(label),
                    float(score),
                ]
            )


def validation_checkpoint(labels, scores, loss):
    """Return the validation selection key and its frozen EER operating point.

    Lower ACER wins, with binary loss as the tie-breaker; test data is not an input.
    """
    if not np.isfinite(loss):
        raise ValueError("Validation loss must be finite")
    threshold = validation_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    return (metrics["ACER"], float(loss)), metrics


@torch.inference_mode()
def evaluate_validation(model, clean, stress, config):
    """Compare clean/stress views of the same semantic validation cohort.

    The pooled validation scores choose the threshold; view labels must match.
    """
    model.eval()
    labels, views, losses, accuracies = [], [], [], {}
    for name, dataset in (("clean", clean), ("stress", stress)):
        targets_all, scores, loss_sum = [], [], 0.0
        correct = {"face_attributes": 0, "spoof_type": 0, "illumination": 0}
        for images, targets in make_loader(dataset, config):
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model.forward_tasks(images.cuda(non_blocking=True))
            binary = targets["binary"].cuda(non_blocking=True)
            loss_sum += F.cross_entropy(
                outputs["binary_classes"].float(), binary, reduction="sum"
            ).item()
            targets_all.extend(targets["binary"].tolist())
            scores.extend(outputs["binary"].float().sigmoid().cpu().tolist())
            for task in ("spoof_type", "illumination"):
                correct[task] += int(
                    (outputs[task].argmax(1).cpu() == targets[task]).sum()
                )
            correct["face_attributes"] += int(
                (
                    (outputs["face_attributes"].cpu() >= 0)
                    == targets["face_attributes"].bool()
                ).sum()
            )
        if len(targets_all) != len(dataset):
            raise RuntimeError("Validation coverage differs from the selected cohort")
        labels.append(np.asarray(targets_all))
        views.append(np.asarray(scores))
        losses.append(loss_sum / len(dataset))
        accuracies[name] = {
            task: value / (len(dataset) * (40 if task == "face_attributes" else 1))
            for task, value in correct.items()
        }
    if not np.array_equal(labels[0], labels[1]):
        raise ValueError("Validation views differ in split assignment/label order")
    key, pooled = validation_checkpoint(
        np.tile(labels[0], 2), np.concatenate(views), sum(losses) / 2
    )
    return key, {
        "pooled_views": pooled,
        "clean": binary_metrics(labels[0], views[0], pooled["threshold"]),
        "stress": binary_metrics(labels[0], views[1], pooled["threshold"]),
        "binary_CE": sum(losses) / 2,
        "auxiliary_accuracy": accuracies,
    }
