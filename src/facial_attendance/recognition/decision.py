"""Name a probe face only when its best gallery match is both strong and unambiguous.

The threshold tau alone is not enough: unregistered people who look like an enrolled
person score above it (see configs/recognition.toml). The rule therefore also asks that
the best person beats every other enrolled person by a margin:

    accept the best label  iff  s1 >= tau  and  s1 - s2 >= margin

s1 is the best score and s2 the best score of a different person, so a person with
several templates never competes with themself. With one person enrolled there is no
s2; tau stands in for it, which means s1 >= tau + margin. Two people tied for the best
score are always unknown, whatever the margin.

tau is chosen for comparing one pair of faces. Matching against a whole class gives an
unregistered person one chance per enrolled person, so the config may also set a higher
min_score for naming anyone: then s1 >= max(tau, min_score). With one person enrolled
tau, not min_score, still stands in for s2.

tau comes from the checkpoint's meta["threshold"]; the margin and min_score come from the
config file. Neither the margin nor tau has a default here.
"""

import tomllib
from dataclasses import dataclass
from math import isfinite
from numbers import Real

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
        raise ValueError(
            "The checkpoint has no meta['threshold']; choose one before deploying"
        )
    return _number(meta["threshold"], "meta['threshold']")


def _decision_section(config_file) -> dict:
    with open(config_file, "rb") as handle:
        section = tomllib.load(handle).get("decision", {})
    if "margin" not in section or not set(section) <= {"margin", "min_score"}:
        raise ValueError("Expected [decision] margin, and optionally min_score")
    return section


def load_margin(config_file) -> float:
    """Read [decision] margin; it must be present and >= 0."""
    margin = _number(_decision_section(config_file)["margin"], "margin")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    return margin


def load_min_score(config_file) -> float | None:
    """Read [decision] min_score, the lowest score that names anyone; None when absent."""
    section = _decision_section(config_file)
    if "min_score" not in section:
        return None
    min_score = _number(section["min_score"], "min_score")
    if not -1 <= min_score <= 1:
        raise ValueError("min_score must be a cosine in [-1, 1]")
    return min_score


def decide(labels, scores, tau, margin, min_score=None) -> Decision:
    """Apply the rule to one score per gallery template (labels may repeat)."""
    labels = list(labels)
    scores = [_number(score, "score") for score in scores]
    tau = _number(tau, "tau")
    margin = _number(margin, "margin")
    floor = tau if min_score is None else max(tau, _number(min_score, "min_score"))
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

    if s1 < floor:
        return Decision(None, False, BELOW_THRESHOLD, best_label, s1, s2)
    if (len(ranked) > 1 and s1 == s2) or s1 - s2 < margin:
        return Decision(None, False, AMBIGUOUS, best_label, s1, s2)
    return Decision(best_label, True, ACCEPTED, best_label, s1, s2)
