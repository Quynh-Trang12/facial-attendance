"""Recalculate validation/test reports from frozen scores and reviewed row indices.

Validation compares four historical checkpoints; test assesses only the saved
validation winner. Neither mode fits a threshold, runs inference or trains weights.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import csv
import json
import time

import numpy as np

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.datasets.reviewed_cache import (
    ReviewedCacheView,
    reviewed_prediction_indices,
)
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages
from facial_attendance.liveness.evaluation.evaluation_metrics import (
    binary_metrics,
    family_metrics,
    published_benchmark_metrics,
)


def saved_scores(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([int(row["binary_label"]) for row in rows])
    scores = np.array([float(row["spoof_score"]) for row in rows])
    return rows, labels, scores


# Historical best available checkpoints, frozen before this content review.
CHECKPOINTS = (
    ("binary-classification", "liveness-comparison", 13),
    ("binary-geometric", "liveness-comparison", 11),
    ("binary-semantic", "liveness-continuation", 21),
    ("aenet-full", "liveness-continuation", 18),
)


def compare_validation(project, directory):
    root = project / "data/raw/liveness/celeba-spoof-full"
    source = ReproductionImages(
        root,
        root / "audits/current-schema-2026-11-12/inventory.csv",
        root / "project_validation.json",
    )
    # Explicit diagnostic access keeps unresolved examples without calling them valid.
    view = ReviewedCacheView(source, directory, "validation", allow_pending_review=True)
    inventory = json.loads((directory / "cache-view.json").read_text())
    pending = inventory["splits"]["validation"]["pending_review"]
    results = {}
    for arm, run, epoch in CHECKPOINTS:
        receipt_path = project / (
            f"evidence/liveness/{run}/{arm}-epoch-checkpoints/epoch-{epoch:03d}.json"
        )
        receipt = json.loads(receipt_path.read_text())
        scores_path = receipt_path.parent / receipt["validation_predictions"]["file"]
        if file_sha256(scores_path) != receipt["validation_predictions"]["sha256"]:
            raise ValueError("Saved validation predictions checksum mismatch")
        rows, labels, scores = saved_scores(scores_path)
        if [r["relative_path"] for r in rows] != [
            s.relative_path for s in source.samples
        ]:
            raise ValueError("Saved scores differ from the original validation order")
        if not np.array_equal(labels, [s.binary_label for s in source.samples]):
            raise ValueError("Saved labels differ from source annotation labels")
        threshold = receipt["inference_metadata"]["threshold"]
        results[arm] = {
            "epoch": epoch,
            "receipt": str(receipt_path.relative_to(project)),
            "receipt_sha256": file_sha256(receipt_path),
            "validation_predictions_sha256": file_sha256(scores_path),
            "original": binary_metrics(labels, scores, threshold),
            "filtered": binary_metrics(
                labels[view.indices], scores[view.indices], threshold
            ),
        }
    return {
        "method": "SHA-verified saved validation scores; frozen historical best checkpoints and thresholds; no new inference, threshold fitting or training",
        "limitations": (
            "Only confirmed exclusions are removed. "
            + (
                "Unresolved reviewed cases remain in this diagnostic. "
                if pending
                else "The declared review queue is closed, without full dataset semantic clearance. "
            )
            + "Training defects remain in historical checkpoint provenance; their effect on learned weights is unmeasured. Validation has prior model-selection exposure; this is not an independent test result."
        ),
        "cache_view_sha256": file_sha256(directory / "cache-view.json"),
        "pending_validation_review": pending,
        "source_sha256": file_sha256(Path(__file__)),
        "models": results,
    }


def evaluate_test(project, directory):
    started = time.monotonic()
    artifact = json.loads(
        (project / "evidence/liveness/submission-artifacts.json").read_text()
    )
    choice = json.loads(
        (
            project / "evidence/liveness/liveness-continuation/model-choice.json"
        ).read_text()
    )
    selected = choice["arms"][choice["chosen"]]
    original_path = project / artifact["evidence"].get(
        "original_public_test", artifact["evidence"]["public_test"]
    )
    scores_path = project / artifact["evidence"].get(
        "original_public_predictions", artifact["evidence"]["public_predictions"]
    )
    original = json.loads(original_path.read_text())
    if original["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Test scores belong to a different selected checkpoint")
    if original["metrics"]["threshold"] != selected["threshold"]:
        raise ValueError("Test scores use a different operating threshold")
    root = project / "data/raw/liveness/celeba-spoof-full"
    source = ReproductionImages(
        root,
        root / "audits/current-schema-2026-11-12/inventory.csv",
        root / "metas/intra_test/test_label.json",
    )
    rows, labels, scores = saved_scores(scores_path)
    if [
        (r["relative_path"], int(r["binary_label"]), int(r["source_type"]))
        for r in rows
    ] != [(s.relative_path, s.binary_label, s.source_type) for s in source.samples]:
        raise ValueError("Test prediction paths/labels/types differ from source rows")
    if {r["split"] for r in rows} != {"test"}:
        raise ValueError("Prediction file contains another split")
    threshold = selected["threshold"]
    if binary_metrics(labels, scores, threshold) != original["metrics"]:
        raise ValueError("Saved predictions do not reproduce the original report")
    binding = source.bindings()
    data = {
        "inventory_sha256": binding["inventory_sha256"],
        "test": {"annotations_sha256": binding["annotations_sha256"]},
    }
    indices = reviewed_prediction_indices(rows, data, directory, "test")
    labels, scores = labels[indices], scores[indices]
    kept = [rows[int(i)] for i in indices]
    types = np.array([int(r["source_type"]) for r in kept])

    result = {
        "split": "test",
        "checkpoint_sha256": original["checkpoint_sha256"],
        "model_version": original["model_version"],
        "inference_precision": original["inference_precision"],
        "threshold_source": original["threshold_source"],
        "metrics": binary_metrics(labels, scores, threshold),
        "by_attack": family_metrics(
            labels, scores, [r["attack_family"] for r in kept], threshold
        ),
        "by_source_type": {
            str(t): binary_metrics(labels[types == t], scores[types == t], threshold)
            for t in sorted(set(types.tolist()))
        },
        "published_benchmark": published_benchmark_metrics(labels, scores),
        "original_metrics": original["metrics"],
        "excluded_samples": len(rows) - len(kept),
        "cache_view_sha256": file_sha256(directory / "cache-view.json"),
        "data": data,
        "original_report": str(original_path.relative_to(project)),
        "original_report_sha256": file_sha256(original_path),
        "predictions": str(scores_path.relative_to(project)),
        "predictions_sha256": file_sha256(scores_path),
        "source_sha256": file_sha256(Path(__file__)),
        "recalculation_seconds": time.monotonic() - started,
        "scope": "Current retained test subset; existing per-image scores, unchanged selected checkpoint and validation threshold. No inference, retraining or deployment-threshold fitting. Test has prior experimental exposure; this is not the intact public benchmark.",
    }
    return result, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cache-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    project, directory = args.project.resolve(), args.cache_view.resolve()
    predictions = args.output.with_name(args.output.stem + "-predictions.csv")
    if args.output.exists() or (args.split == "test" and predictions.exists()):
        raise FileExistsError("Preserve prior evidence; choose a new output path")
    if args.split == "validation":
        result = compare_validation(project, directory)
        summary = {arm: row["filtered"] for arm, row in result["models"].items()}
    else:
        result, kept = evaluate_test(project, directory)
        with predictions.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(kept[0]))
            writer.writeheader()
            writer.writerows(kept)
        result["filtered_predictions"] = str(predictions.resolve().relative_to(project))
        result["filtered_predictions_sha256"] = file_sha256(predictions)
        summary = result["metrics"]
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
