"""Offline evidence for the quality-gated multi-frame check-in.

Two questions, both answered at the deployed threshold tau:

- Error versus discard: if the pairs with the worst image quality are set aside,
  does the true-accept rate of the remaining pairs rise faster than when the same
  number of pairs is set aside at random? If so, the quality measure points at the
  frames that cause wrong decisions.
- k-shot fusion: does a probe fused from k images of a person, with the
  enrollment formula, reach that person's single gallery image more often than one
  image does, without accepting more impostors?

Only numpy is needed here, so the arithmetic is tested without images or torch.
"""

import numpy as np

from .probe_fusion import fuse

MEASURES = ("sharpness", "brightness", "contrast", "combined")
FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
# Fractions that decide which measure is used; 0 is identical for every measure.
SELECTION_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30)
MID_GREY = 128.0


def rank_auc(labels, scores) -> float:
    """P(score of a random positive > score of a random negative), ties counted as one half."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = scores[labels], np.sort(scores[~labels])
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUC needs at least one positive and one negative")
    below = np.searchsorted(negatives, positives, side="left")
    tied = np.searchsorted(negatives, positives, side="right") - below
    return float((below + 0.5 * tied).sum() / (len(positives) * len(negatives)))


def operating_point(labels, scores, tau) -> dict:
    """Accept when score >= tau, as the attendance rule does."""
    labels = np.asarray(labels).astype(bool)
    accepted = np.asarray(scores, dtype=np.float64) >= tau
    return {
        "tar": float(accepted[labels].mean()),
        "far": float(accepted[~labels].mean()),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
    }


def _percentile_rank(values):
    """Rank in [0, 1]; equal values share their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    sorted_values = values[order]
    # Average the ranks of tied values so the order of equal images does not matter.
    _, start, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    for first, count in zip(start, counts):
        if count > 1:
            ranks[order[first : first + count]] = first + (count - 1) / 2
    return ranks / max(len(values) - 1, 1)


def image_scores(rows) -> dict[str, dict[str, float]]:
    """Per-image quality scores where higher is better, for every measure.

    rows: dicts with path, sharpness, brightness and contrast. Brightness scores by
    closeness to mid-grey, since too dark and too bright both hurt. "combined" is the
    mean of the three percentile ranks within this set of images.
    """
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Each image must appear once in the quality table")
    raw = {
        "sharpness": np.array([float(row["sharpness"]) for row in rows]),
        "brightness": -np.abs(np.array([float(row["brightness"]) for row in rows]) - MID_GREY),
        "contrast": np.array([float(row["contrast"]) for row in rows]),
    }
    raw["combined"] = np.mean([_percentile_rank(raw[name]) for name in ("sharpness", "brightness", "contrast")], axis=0)
    return {name: dict(zip(paths, values.tolist())) for name, values in raw.items()}


def pair_quality(pairs, scores_by_path) -> np.ndarray:
    """A pair is as good as its worse image."""
    try:
        return np.array([min(scores_by_path[left], scores_by_path[right]) for left, right in pairs])
    except KeyError as error:
        raise ValueError(f"No quality score for image {error.args[0]}") from None


def _summary(labels, scores, tau):
    return operating_point(labels, scores, tau) | {"auc": rank_auc(labels, scores)}


def discard_curve(labels, scores, quality, tau, fractions=FRACTIONS) -> list[dict]:
    """Drop the lowest-quality pairs (stable order on ties) and score what remains."""
    labels, scores, quality = map(np.asarray, (labels, scores, quality))
    order = np.argsort(quality, kind="stable")
    curve = []
    for fraction in fractions:
        dropped = int(np.floor(fraction * len(labels)))
        keep = np.sort(order[dropped:])
        curve.append({"fraction": fraction, "kept": int(len(keep))} | _summary(labels[keep], scores[keep], tau))
    return curve


def random_discard(labels, scores, tau, fractions=FRACTIONS, seeds=20) -> list[dict]:
    """The same drops at random, averaged over seeds 0..seeds-1: the baseline to beat."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    runs = [discard_curve(labels, scores, np.random.default_rng(seed).random(len(labels)), tau, fractions)
            for seed in range(seeds)]
    curve = []
    for index, fraction in enumerate(fractions):
        points = [run[index] for run in runs]
        entry = {"fraction": fraction, "kept": points[0]["kept"], "seeds": seeds}
        for key in ("tar", "far", "auc"):
            values = np.array([point[key] for point in points])
            entry[key] = float(values.mean())
            entry[f"{key}_std"] = float(values.std())
        curve.append(entry)
    return curve


def tar_gain(curve, baseline, fractions=SELECTION_FRACTIONS) -> float:
    """Mean TAR above the random baseline over the given fractions."""
    by_fraction = {point["fraction"]: point["tar"] for point in curve}
    base = {point["fraction"]: point["tar"] for point in baseline}
    return float(np.mean([by_fraction[f] - base[f] for f in fractions]))


def select_measure(curves, baseline) -> dict:
    """Pick the measure with the largest mean TAR gain over random discard."""
    gains = {name: tar_gain(curve, baseline) for name, curve in curves.items()}
    # On a tie the simpler measure wins: MEASURES lists single measures before "combined".
    order = sorted(gains, key=lambda name: (MEASURES.index(name) if name in MEASURES else len(MEASURES), name))
    chosen = max(order, key=lambda name: gains[name])
    # Random drops already move TAR a little; a gain inside that spread proves nothing.
    noise = float(np.mean([point["tar_std"] for point in baseline if point["fraction"] in SELECTION_FRACTIONS]))
    return {
        "rule": "largest mean TAR gain over random discard at fractions "
                + ", ".join(f"{f:.2f}" for f in SELECTION_FRACTIONS)
                + "; ties go to the earlier of " + ", ".join(MEASURES)
                + "; it beats random only if the gain exceeds twice the random spread",
        "gains": gains,
        "measure": chosen,
        "random_tar_spread": noise,
        "beats_random": gains[chosen] > 2 * noise,
    }


def limit_percentiles(rows, low=5, high=95) -> dict:
    """Starting points for the quality limits, from the validation images."""
    result = {}
    for name in ("sharpness", "brightness", "contrast", "size"):
        values = np.array([float(row[name]) for row in rows])
        result[name] = {f"p{low}": float(np.percentile(values, low)), f"p{high}": float(np.percentile(values, high))}
    return result


def kshot(embeddings_by_identity, tau, ks=(1, 2, 3)) -> dict:
    """Fused k-image probe against one gallery image, rotated over the identity's images.

    Every identity contributes exactly max(ks) + 1 images, the first ones in its given
    order; identities with fewer are left out and counted. For fold g, image g is the
    gallery and the probe fuses the next k images in rotation. Genuine scores compare a
    probe with its own gallery image; impostor scores compare it with every other
    identity's gallery image.
    """
    size = max(ks) + 1
    usable = {name: np.asarray(vectors, dtype=np.float64)[:size]
              for name, vectors in sorted(embeddings_by_identity.items()) if len(vectors) >= size}
    skipped = len(embeddings_by_identity) - len(usable)
    if len(usable) < 2:
        raise ValueError("k-shot needs at least two identities with enough images")
    stack = np.stack(list(usable.values()))  # (identities, size, D)
    stack /= np.linalg.norm(stack, axis=2, keepdims=True)
    count = len(stack)
    off_diagonal = ~np.eye(count, dtype=bool)
    results = {}
    for k in ks:
        folds = []
        for gallery_index in range(size):
            gallery = stack[:, gallery_index]
            probe_indexes = [(gallery_index + step) % size for step in range(1, k + 1)]
            probes = np.stack([fuse(stack[i, probe_indexes]) for i in range(count)])
            matrix = probes @ gallery.T
            genuine, impostor = np.diag(matrix), matrix[off_diagonal]
            labels = np.concatenate([np.ones(count, bool), np.zeros(len(impostor), bool)])
            folds.append(_summary(labels, np.concatenate([genuine, impostor]), tau)
                         | {"gallery_index": gallery_index})
        results[str(k)] = {
            "folds": folds,
            "mean": {key: float(np.mean([fold[key] for fold in folds])) for key in ("tar", "far", "auc")},
        }
    return {"identities": count, "skipped_identities": skipped, "images_per_identity": size,
            "tau": float(tau), "k": results}
