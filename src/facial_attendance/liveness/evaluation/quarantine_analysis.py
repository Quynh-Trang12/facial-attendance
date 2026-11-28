"""Frozen-model diagnostics on source-training quarantine, never an extra test set."""

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    local_path,
    read_json,
    source_path,
    write_json,
)
from facial_attendance.liveness.datasets.face_crop import bbox_path, face_box, read_face
from facial_attendance.liveness.datasets.image_inventory import (
    read_experimental_report,
    read_inventory,
    read_selection,
)
from facial_attendance.liveness.datasets.selected_dataset import LivenessImages
from facial_attendance.liveness.evaluation.evaluation_metrics import (
    binary_metrics,
    family_metrics,
)
from facial_attendance.liveness.experiments.binary_experiment import TrainingConfig
from facial_attendance.liveness.inference.liveness_adapter import LivenessAdapter
from facial_attendance.liveness.models.binary_model import (
    face_transform,
    load_checkpoint,
)
from facial_attendance.liveness.training.validation import (
    predict_dataset,
    save_predictions,
)


def training_quarantine(receipt, readiness, records):
    """Resolve the bound original queue, then keep current exclusions only."""
    queue_file = readiness.parent / "review-priority.csv"
    source_receipt = receipt
    if not queue_file.exists() and receipt.get("base_readiness_sha256"):
        parents = [
            Path(p)
            for p, digest in receipt["evidence_bindings"].items()
            if digest == receipt["base_readiness_sha256"]
        ]
        if len(parents) != 1:
            raise ValueError("Cannot resolve the bound base readiness")
        if file_sha256(parents[0]) != receipt["base_readiness_sha256"]:
            raise ValueError("Base readiness changed")
        source_receipt = read_json(parents[0])
        queue_file = parents[0].parent / "review-priority.csv"
    expected = source_receipt["evidence_bindings"].get(str(queue_file.resolve()))
    if not expected or file_sha256(queue_file) != expected:
        raise ValueError("Training quarantine queue is not bound to the release")
    excluded = read_selection(Path(receipt["selection_file"]), records)
    with queue_file.open(newline="", encoding="utf-8") as handle:
        return {
            r["relative_path"]: r
            for r in csv.DictReader(handle)
            if r["reason"] == "near_candidate_quarantine"
            and r["split"] == "train"
            and r["relative_path"] in excluded
        }


def inspect_run(run_directory, output):
    run_directory, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    run, evaluation = (
        read_json(run_directory / "run.json"),
        read_json(run_directory / "evaluation.json"),
    )
    checkpoint = run_directory / "model.pt"
    if file_sha256(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("Analysis checkpoint differs from evaluated model")
    root, inventory, readiness = (
        Path(run["data"][key]) for key in ("root", "inventory", "readiness")
    )
    if file_sha256(readiness) != run["data"]["readiness_sha256"]:
        raise ValueError("Readiness changed since training")
    receipt = read_experimental_report(root, readiness, file_sha256(inventory))
    records = read_inventory(inventory)
    by_path = {r.relative_path: r for r in records}
    queue = training_quarantine(receipt, readiness, records)
    selected = [by_path[path] for path in sorted(queue)]
    if not selected:
        raise ValueError("No training-origin candidate quarantine to diagnose")
    output.mkdir(parents=True, exist_ok=False)
    config = TrainingConfig(**run["training_config"])
    model, meta = load_checkpoint(checkpoint, "cuda")
    dataset = LivenessImages._for_audit(
        root,
        selected,
        "train",
        transform=face_transform(architecture=meta["architecture"]),
    )
    labels, scores, loss, seconds = predict_dataset(model, dataset, config)
    save_predictions(
        output / "training-quarantine-predictions.csv", dataset, labels, scores
    )
    metrics = binary_metrics(labels, scores, meta["threshold"])
    errors = [
        dict(
            relative_path=row.relative_path,
            source_label=row.binary_label,
            family=row.attack_family,
            spoof_score=float(score),
            label_disagreement=abs(float(score) - row.binary_label),
            reference_path=queue[row.relative_path]["reference_path"],
            reference_split=by_path[queue[row.relative_path]["reference_path"]].split,
        )
        for row, score in zip(dataset.samples, scores, strict=True)
    ]
    cases = []
    for family in ("live", "print", "screen"):
        cases.extend(
            sorted(
                (r for r in errors if r["family"] == family),
                key=lambda r: (-r["label_disagreement"], r["relative_path"]),
            )[:8]
        )
    write_json(output / "review-cases.json", cases)
    _review_previews(root, cases, by_path, output)
    # Integration examples use already evaluated test images only after model
    # selection; these outputs are not used to change weights or the threshold.
    adapter = LivenessAdapter(checkpoint, "cuda")
    test = LivenessImages(root, inventory, "test", audit_file=readiness)
    examples, seen = [], set()
    for i, row in enumerate(test.samples):
        if row.source_type in seen:
            continue
        seen.add(row.source_type)
        face, target = test[i]
        examples.append(
            {
                "relative_path": row.relative_path,
                "source_label": target,
                "source_type": row.source_type,
                **adapter.predict_face(face),
            }
        )
        if len(seen) == 7:
            break
    write_json(output / "integration-outputs.json", examples)
    summary = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_version": meta["model_version"],
        "training_quarantine_samples": len(dataset),
        "source_label_agreement_diagnostic": metrics,
        "unweighted_BCE": loss,
        "evaluation_seconds": seconds,
        "by_attack": family_metrics(
            labels, scores, [r.attack_family for r in selected], meta["threshold"]
        ),
        "reference_split_counts": dict(Counter(r["reference_split"] for r in errors)),
        "reviewed_case_budget": len(cases),
        "policy": receipt["candidate_policy"]["name"],
        "source_sha256": {
            str(p.resolve()): file_sha256(p)
            for p in (Path(__file__), source_path("inference/liveness_adapter.py"))
        },
        "interpretation_limits": [
            "These are excluded source-training images, not independent validation or physical attacks.",
            "Model agreement/disagreement with a public label does not adjudicate duplicate status or label truth.",
            "An evaluation-split reference prohibits blind reinstatement. A train reference alone does not prove absence of other evaluation neighbors.",
            "Restoration changes the data policy and needs its own bound release; this diagnostic does not restore images.",
            "This label-agreement diagnostic alone cannot measure the effect of quarantine; use the separately declared controlled comparison.",
        ],
    }
    write_json(output / "quarantine-diagnostic.json", summary)
    _training_curves(run_directory / "history.csv", output / "training-curves.png")
    return summary


def _review_previews(root, cases, by_path, output):
    for start in range(0, len(cases), 4):
        rows = cases[start : start + 4]
        sheet = Image.new("RGB", (896, len(rows) * 288), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            for side, path in enumerate((row["relative_path"], row["reference_path"])):
                sample = by_path[path]
                filename = local_path(root, path)
                boxfile = local_path(root, str(bbox_path(Path(path))))
                if (
                    file_sha256(filename) != sample.sha256
                    or file_sha256(boxfile) != sample.bbox_sha256
                ):
                    raise ValueError("Review image/box changed since inventory")
                with Image.open(filename) as source:
                    original = source.convert("RGB")
                box, _ = face_box(boxfile, original.size)
                marked = original.copy()
                ImageDraw.Draw(marked).rectangle(
                    box, outline="red", width=max(2, original.width // 100)
                )
                for j, image in enumerate((marked, read_face(filename))):
                    sheet.paste(
                        ImageOps.contain(image, (224, 224)),
                        ((side * 2 + j) * 224, i * 288),
                    )
                draw.text((side * 448 + 2, i * 288 + 225), path, fill="black")
                draw.text(
                    (side * 448 + 2, i * 288 + 241),
                    f"{sample.split} type={sample.source_type} label={sample.binary_label}",
                    fill="black",
                )
            draw.text(
                (2, i * 288 + 259),
                f"quarantine spoof score={row['spoof_score']:.4f}; model disagreement is not duplicate proof",
                fill="black",
            )
        sheet.save(output / f"quarantine-cases-{start:02}.png")


def _training_curves(history, output):
    # Plotting is optional at import; inference and metric helpers do not need
    # a display backend, and batch execution writes standalone artifacts.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(history).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(r["epoch"]) for r in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(
        epochs,
        [float(r["weighted_training_BCE"]) for r in rows],
        label="Train, weighted",
    )
    axes[0].plot(
        epochs,
        [float(r["validation_BCE"]) for r in rows],
        label="Validation, unweighted",
    )
    axes[0].set_ylabel("BCE (different weighting)")
    axes[0].legend()
    metric = (
        "validation_ACER_at_EER"
        if all("validation_ACER_at_EER" in r for r in rows)
        else "validation_ACER"
    )
    axes[1].plot(epochs, [100 * float(r[metric]) for r in rows])
    axes[1].set_ylabel(
        "Validation ACER at EER (%)"
        if metric.endswith("_at_EER")
        else "Validation ACER at 0.5 (%)"
    )
    axes[2].plot(epochs, [float(r["validation_ROC_AUC"]) for r in rows])
    axes[2].set_ylabel("Validation ROC-AUC")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
