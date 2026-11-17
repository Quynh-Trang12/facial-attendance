"""Compare the two face approaches on the shared evaluator's outputs.

The rules are in docs/face-comparison.md: both final models, cosine as the deciding
measure, selection and threshold on the validation pairs only, and the official trials
scored once afterwards. This module checks the exported embeddings before scoring and
computes what the evaluator does not: the paired bootstrap of the AUC difference, the
selection and the operating point of the fixed threshold on the official trials.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

APPROACHES = ("classification", "metric")
OFFICIAL_TRIALS_SHA256 = "55ecf1495188024f414e3b0faac2b7f4cc374000a809e3b98579ecd8f4f25fdd"
VAL_PAIRS_SHA256 = "772e998360268c04744d715b25c0feb9a38482a1b7dc1660d2b3bc90a74e8e8d"
EXPECTED_MODELS = {
    "classification": {
        "model_id": "20261113-160440-classification-78122af-s42",
        "model_version": "3fd31562478d",
        "training_val_pairs_cosine_auc": 0.9884,
    },
    "metric": {
        "model_id": "20261114-125242-metric-74a8657-s42",
        "model_version": "cb38ea47ceff",
        "training_val_pairs_cosine_auc": 0.9911,
    },
}
SCORES = ("cosine", "negative_euclidean")


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_path_strings(trials_file) -> set[str]:
    """Every distinct image string in the left and right columns of a trial file."""
    paths = set()
    for line in Path(trials_file).read_text(encoding="utf-8-sig").splitlines():
        columns = line.split()
        if columns and not line.lstrip().startswith("#"):
            paths.update(columns[:2])
    return paths


def check_export(export_dir, approach: str, *, official_trials, val_pairs, expected=EXPECTED_MODELS) -> dict:
    """Refuse an export that names the wrong model or does not cover exactly a trial file."""
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach {approach!r}")
    export_dir = Path(export_dir)
    summary = json.loads((export_dir / "export-summary.json").read_text())
    want = expected[approach]
    for key in ("model_id", "model_version"):
        if summary.get(key) != want[key]:
            raise ValueError(f"{approach} export has {key} {summary.get(key)!r}, expected {want[key]!r}")
    files = {}
    for name, trials in (("official-embeddings.npz", official_trials), ("val-pairs-embeddings.npz", val_pairs)):
        archive = export_dir / name
        with np.load(archive, allow_pickle=False) as data:
            paths, vectors = [str(p) for p in data["paths"]], data["raw_embeddings"]
        expected_paths = trial_path_strings(trials)
        if len(paths) != len(set(paths)) or set(paths) != expected_paths:
            missing, extra = len(expected_paths - set(paths)), len(set(paths) - expected_paths)
            raise ValueError(f"{approach} {name} does not list exactly the paths of {Path(trials).name} "
                             f"({missing} missing, {extra} extra, {len(paths) - len(set(paths))} repeated)")
        if vectors.shape != (len(paths), 512) or not np.isfinite(vectors).all():
            raise ValueError(f"{approach} {name} must hold finite float[{len(paths)}, 512] raw embeddings")
        files[name] = {"sha256": file_sha256(archive), "images": len(paths)}
    return {"approach": approach, "model_id": summary["model_id"],
            "model_version": summary["model_version"], "files": files}


def read_pair_scores(path) -> dict:
    """The evaluator's pair_scores.csv as aligned arrays."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no pair scores")
    return {"keys": [(r["left"], r["right"]) for r in rows],
            "labels": np.array([int(r["label"]) for r in rows]),
            **{name: np.array([float(r[name]) for r in rows]) for name in SCORES}}


def paired_bootstrap(labels, first, second, auc, *, replicates=2000, seed=42, level=0.95) -> dict:
    """Percentile interval of AUC(second) - AUC(first), resampling the same trials for both.

    `auc(labels, scores)` must be the evaluator's AUC. A replicate that draws only one
    class has no AUC; it is drawn again and counted.
    """
    labels, first, second = np.asarray(labels), np.asarray(first), np.asarray(second)
    if not len(labels) == len(first) == len(second) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Bootstrap needs aligned scores and both classes")
    rng = np.random.default_rng(seed)
    deltas, redrawn = [], 0
    while len(deltas) < replicates:
        index = rng.integers(0, len(labels), len(labels))
        sample = labels[index]
        if sample.min() == sample.max():
            redrawn += 1
            continue
        deltas.append(auc(sample, second[index]) - auc(sample, first[index]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(deltas, [tail, 100 - tail])
    observed = auc(labels, second) - auc(labels, first)
    return {"statistic": "AUC(second) - AUC(first)", "observed": float(observed),
            "low": float(low), "high": float(high), "level": level, "replicates": replicates,
            "seed": seed, "redrawn_single_class": redrawn, "excludes_zero": bool(low > 0 or high < 0)}


def select_approach(cosine_auc: dict) -> str:
    """The approach with the higher validation cosine AUC; an exact tie is left to the team."""
    (a, auc_a), (b, auc_b) = sorted(cosine_auc.items())
    if auc_a == auc_b:
        raise ValueError(f"Validation cosine AUC ties at {auc_a}; the selection rule does not decide")
    return a if auc_a > auc_b else b


def operating_point(labels, scores, threshold: float) -> dict:
    """False-accept and true-accept rates when a trial is accepted at score >= threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    accepted = scores >= threshold
    negatives, positives = labels == 0, labels == 1
    return {"threshold": float(threshold),
            "false_accepts": int(accepted[negatives].sum()), "negatives": int(negatives.sum()),
            "far": float(accepted[negatives].mean()),
            "true_accepts": int(accepted[positives].sum()), "positives": int(positives.sum()),
            "tar": float(accepted[positives].mean())}


def roc_point_at(roc: list[dict], threshold: float) -> dict:
    """The evaluator's ROC point that accepts exactly the scores >= threshold."""
    candidates = [p for p in roc if p["threshold"] is not None and p["threshold"] >= threshold]
    if not candidates:
        return {"fpr": 0.0, "tpr": 0.0, "threshold": None}
    return min(candidates, key=lambda p: p["threshold"])
