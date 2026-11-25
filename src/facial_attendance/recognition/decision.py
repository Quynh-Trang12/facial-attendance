"""Name a probe face only when its best gallery match is both strong and unambiguous.

The threshold tau alone is not enough: unregistered people who look like an enrolled
person score above it (see configs/recognition.toml). The rule therefore also asks that
the best person beats every other enrolled person by a margin:

    accept the best label  iff  s1 >= tau  and  s1 - s2 >= margin

s1 is the best score and s2 the best score of a different person, so a person with
several templates never competes with themself. With one person enrolled there is no
s2; tau stands in for it, which means s1 >= tau + margin. Two people tied for the best
score are always unknown, whatever the margin.

tau comes from the checkpoint's meta["threshold"] and the margin from the config file;
neither has a default here.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import tomllib

ACCEPTED = "accepted"
EMPTY_GALLERY = "empty_gallery"
BELOW_THRESHOLD = "below_threshold"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Decision:
    """label is None (unknown) unless accepted; best_label is reported either way."""

    label: str | None
    accepted: bool
    reason: str
    best_label: str | None
    best_score: float | None
    # s2, or tau when only one person is enrolled; None for an empty gallery.
    runner_up_score: float | None


def _number(value, name) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def threshold_from_meta(meta) -> float:
    """The deployed tau; refuse a checkpoint whose threshold was never chosen."""
    if meta.get("threshold") is None:
        raise ValueError("The checkpoint has no meta['threshold']; choose one before deploying")
    return _number(meta["threshold"], "meta['threshold']")


def load_margin(config_file) -> float:
    """Read [decision] margin; it must be present and >= 0."""
    with open(config_file, "rb") as handle:
        section = tomllib.load(handle).get("decision", {})
    if set(section) != {"margin"}:
        raise ValueError("Expected exactly one decision key: margin")
    margin = _number(section["margin"], "margin")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    return margin


def decide(labels, scores, tau, margin) -> Decision:
    """Apply the rule to one score per gallery template (labels may repeat)."""
    labels = list(labels)
    scores = [_number(score, "score") for score in scores]
    tau = _number(tau, "tau")
    margin = _number(margin, "margin")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if len(labels) != len(scores):
        raise ValueError("Need one score per label")
    if not labels:
        return Decision(None, False, EMPTY_GALLERY, None, None, None)

    per_person = {}
    for label, score in zip(labels, scores):
        per_person[label] = max(score, per_person.get(label, score))
    # Sort by score; equal scores keep enrolment order so best_label is stable.
    ranked = sorted(per_person.items(), key=lambda item: -item[1])
    best_label, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else tau

    if s1 < tau:
        return Decision(None, False, BELOW_THRESHOLD, best_label, s1, s2)
    if (len(ranked) > 1 and s1 == s2) or s1 - s2 < margin:
        return Decision(None, False, AMBIGUOUS, best_label, s1, s2)
    return Decision(best_label, True, ACCEPTED, best_label, s1, s2)
