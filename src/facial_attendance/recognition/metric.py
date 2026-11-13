"""Triplet loss with semi-hard online mining, and the identity check behind it.

The loss follows FaceNet (Schroff et al., 2015) on L2-normalized embeddings, with
squared Euclidean distance `d = ‖a − b‖² = 2 − 2·cos(a, b)`:

    L = mean over anchor–positive pairs of max(0, d(a, p) − d(a, n) + margin)

Every same-identity pair in a P×K batch is an anchor–positive pair. Its negative is
mined inside the batch with the semi-hard rule of `tfa.losses.TripletSemiHardLoss`:
the closest negative that is still farther than the positive. If no negative is
farther, the farthest one is used instead. Taking the hardest negative outright tends
to collapse every embedding to one point early in training, which is why the
semi-hard rule is used.

Mining is correct only if positives really share the anchor's identity and negatives
really do not. `triplet_loss` checks both on every batch, together with the batch
shape the loss needs, and raises if any check fails.
"""

import torch
from torch import nn

MARGIN = 0.2


def check_batch_shape(targets: torch.Tensor):
    """A batch needs 2+ identities (for negatives) and 2+ images each (for positives)."""
    identities, counts = torch.unique(targets, return_counts=True)
    if len(identities) < 2:
        raise ValueError("A triplet batch needs at least 2 identities")
    if int(counts.min()) < 2:
        short = identities[counts < 2].tolist()
        raise ValueError(f"Identities with fewer than 2 images in the batch: {short[:5]}")


def check_triplets(targets: torch.Tensor, anchors, positives, negatives):
    """Raise unless each positive shares its anchor's identity and each negative does not."""
    if bool((anchors == positives).any()):
        raise AssertionError("An anchor was paired with itself as its positive")
    if not bool((targets[anchors] == targets[positives]).all()):
        raise AssertionError("A positive has a different identity from its anchor")
    if bool((targets[anchors] == targets[negatives]).any()):
        raise AssertionError("A negative has the same identity as its anchor")


def triplet_loss(
    embeddings: torch.Tensor, targets: torch.Tensor, margin: float = MARGIN
) -> tuple[torch.Tensor, dict]:
    """Return the batch loss and counts that show what the mining did."""
    check_batch_shape(targets)
    # Distances are computed in float32 even under mixed precision.
    unit = nn.functional.normalize(embeddings.float(), dim=1)
    distance = (2 - 2 * unit @ unit.T).clamp_min(0)

    same = targets[:, None] == targets[None, :]
    self_pair = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    anchors, positives = torch.nonzero(same & ~self_pair, as_tuple=True)
    d_ap = distance[anchors, positives]

    # One row per anchor–positive pair, one column per batch image.
    d_an = distance[anchors]
    negative = ~same[anchors]
    farther = negative & (d_an > d_ap[:, None])
    has_farther = farther.any(dim=1)
    inf = torch.tensor(float("inf"), device=distance.device)
    # Semi-hard: the closest negative beyond the positive.
    closest_beyond = torch.where(farther, d_an, inf).argmin(dim=1)
    # Fallback: the farthest negative, when every negative is closer than the positive.
    farthest = torch.where(negative, d_an, -inf).argmax(dim=1)
    negatives = torch.where(has_farther, closest_beyond, farthest)
    check_triplets(targets, anchors, positives, negatives)

    d_neg = d_an.gather(1, negatives[:, None]).squeeze(1)
    per_pair = (d_ap - d_neg + margin).clamp_min(0)
    loss = per_pair.mean()
    # The counts are a record, not part of the graph.
    d_ap, d_neg, per_pair, unit = d_ap.detach(), d_neg.detach(), per_pair.detach(), unit.detach()
    stats = {
        "anchor_positive_pairs": len(anchors),
        "semi_hard": int((has_farther & (d_neg < d_ap + margin)).sum()),
        "fallback": int((~has_farther).sum()),
        "active": int((per_pair > 0).sum()),
        "mean_positive_distance": float(d_ap.mean()),
        "mean_negative_distance": float(d_neg.mean()),
        # Close to 0 when every image maps to the same point (collapse).
        "embedding_std": float(unit.std(dim=0).mean()),
    }
    return loss, stats
