"""Measure live/spoof errors with live=0, spoof=1 and higher scores meaning spoof.

APCER counts spoof images accepted as live. BPCER counts live images rejected as
spoof. ACER averages those two rates; ROC-AUC measures ranking across thresholds.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def validate_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or len(labels) == 0
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Expected aligned binary labels and finite probabilities")
    return labels.astype(int), scores


def binary_metrics(labels, scores, threshold):
    """Count errors at one fixed threshold; score >= threshold predicts spoof."""
    labels, scores = validate_scores(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    predicted = scores >= threshold
    live, attack = labels == 0, labels == 1
    false_live = int((~predicted & attack).sum())
    false_spoof = int((predicted & live).sum())
    apcer = false_live / int(attack.sum()) if attack.any() else None
    bpcer = false_spoof / int(live.sum()) if live.any() else None
    return {
        "samples": len(labels),
        "live": int(live.sum()),
        "spoof": int(attack.sum()),
        "attack_accepted_as_live": false_live,
        "live_rejected_as_spoof": false_spoof,
        "threshold": float(threshold),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) / 2
        if apcer is not None and bpcer is not None
        else None,
        "accuracy": float((predicted == labels).mean()),
        "ROC_AUC": float(roc_auc_score(labels, scores))
        if live.any() and attack.any()
        else None,
    }


def validation_threshold(labels, scores):
    """Choose a validation threshold where APCER and BPCER are closest.

    An equal-distance tie prefers lower ACER, then the lower threshold. Both live
    and spoof labels are required; this function does not fit model weights.
    """
    labels, scores = validate_scores(labels, scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation threshold selection requires both classes")
    fpr, tpr, thresholds = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    finite = [
        i
        for i, threshold in enumerate(thresholds)
        if np.isfinite(threshold) and 0 <= threshold <= 1
    ]
    # Prefer the closest equal-error point, then lower mean error; a remaining
    # tie chooses the lower threshold (more conservative attack rejection).
    index = min(
        finite,
        key=lambda i: (
            abs(fpr[i] - (1 - tpr[i])),
            (fpr[i] + 1 - tpr[i]) / 2,
            thresholds[i],
        ),
    )
    return float(thresholds[index])


def family_metrics(labels, scores, families, threshold):
    labels, scores = validate_scores(labels, scores)
    families = np.asarray(families)
    if families.shape != labels.shape:
        raise ValueError("Attack-family alignment differs")
    result = {}
    for family in ("print", "screen"):
        selected = families == family
        if selected.any():
            result[family] = binary_metrics(
                labels[selected], scores[selected], threshold
            )
    return result


def published_benchmark_metrics(labels, scores):
    """Summarize test scores using the released CelebA-Spoof client's operating points.

    For each target false-positive rate, choose a cutoff from the sorted live scores
    and report the fraction of attack scores strictly above it. The separate EER
    summary also scans these supplied scores. These test-derived thresholds describe
    the benchmark; they do not replace the validation threshold stored in a model.
    """
    labels, scores = validate_scores(labels, scores)
    live, attack = scores[labels == 0], scores[labels == 1]
    if not len(live) or not len(attack):
        raise ValueError("Published benchmark metrics require both classes")
    ordered_live = np.sort(live)[::-1]
    operating = {}
    for fpr in (0.01, 0.005, 0.001):
        index = max(1, int(fpr * len(live))) - 1
        threshold = float(ordered_live[index])
        operating[str(fpr)] = {
            "threshold_descriptive_only": threshold,
            "observed_FPR": float((live > threshold).mean()),
            "TPR": float((attack > threshold).mean()),
        }
    threshold = validation_threshold(labels, scores)
    eer_point = binary_metrics(labels, scores, threshold)
    return {
        "TPR_at_FPR": operating,
        "nearest_discrete_EER": eer_point["ACER"],
        "descriptive_EER_operating_point": eer_point,
        "source": "https://github.com/ZhangYuanhan-AI/CelebA-Spoof/blob/master/intra_dataset_code/client.py",
        "scope": "Descriptive test-set threshold sweeps only; deployment threshold remains frozen from validation",
    }
