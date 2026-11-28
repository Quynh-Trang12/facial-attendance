"""Recheck the recorded face comparison from the evaluator's saved outputs alone.

The comparison (compare_face_approaches.py) records its numbers in selection.json and
official-summary.json. This script recomputes them without the evaluator or any model:
AUC by a rank formula, the threshold rule, the operating point, the paired bootstrap,
file hashes, run identities and the order of the two steps. Each check prints PASS or
FAIL; the exit code is 1 if any check fails.

    python scripts/check_face_comparison.py --selection evidence/FAC-05/selection.json \
        --validation-dir <dir with validation-classification/ and validation-metric/> \
        --official-summary evidence/FAC-05/official-summary.json \
        --official-dir <dir with official-classification/ and official-metric/> \
        --output <report.json>

Either step may be left out; it is then reported as not checked.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.comparison import (  # noqa: E402
    APPROACHES,
    EXPECTED_MODELS,
    OFFICIAL_TRIALS_SHA256,
    SCORES,
    VAL_PAIRS_SHA256,
    file_sha256,
    operating_point,
    paired_bootstrap,
    roc_point_at,
    select_approach,
)

EVALUATOR_COMMIT = "2d14220e912acfbdda8f378e94f81c9bafbbf3a6"
FAR_TARGET = 0.01
TOLERANCE = 1e-9


def rank_auc(labels, scores) -> float:
    """Mann-Whitney AUC from average ranks, so tied scores count one half."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    _, first, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def far_threshold(labels, scores, target=FAR_TARGET) -> dict:
    """Lowest observed score whose false-accept rate at score >= it is at most the target.

    Tied scores are accepted or rejected together, because the rate is computed per
    distinct score. `lower_far` is the rate at the next lower observed score.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    negatives = np.sort(scores[labels == 0])
    candidates = np.unique(scores)
    far = (len(negatives) - np.searchsorted(negatives, candidates, side="left")) / len(
        negatives
    )
    meets = np.flatnonzero(far <= target)
    if not len(meets):
        return {"threshold": None, "reject_all": True}
    index = meets[0]
    return {
        "threshold": float(candidates[index]),
        "reject_all": False,
        "lower_far": float(far[index - 1]) if index else None,
    }


def read_scores(path) -> dict:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {
        name: np.array([float(r[name]) for r in rows])
        for name in ("cosine", "euclidean_distance", "negative_euclidean")
    }
    return {
        "keys": [(r["left"], r["right"]) for r in rows],
        "labels": np.array([int(r["label"]) for r in rows]),
        **columns,
    }


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    def skip(self, name: str, reason: str):
        self.checks.append({"check": name, "ok": None, "detail": reason})
        print(f"----  {name}  not checked: {reason}")

    @property
    def failed(self) -> int:
        return sum(c["ok"] is False for c in self.checks)


def close(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def check_run(report, step: str, approach: str, folder: Path, record: dict) -> dict:
    """Hashes, evaluator identity, counts, score direction and AUC of one evaluator output."""
    label = f"{step} {approach}"
    hashes = {"pair_scores.csv": "pair_scores_sha256", "metrics.json": "metrics_sha256"}
    report.add(
        f"{label}: output files are the recorded ones",
        all(file_sha256(folder / name) == record[key] for name, key in hashes.items()),
    )
    metrics = json.loads((folder / "metrics.json").read_text())
    model = EXPECTED_MODELS[approach]
    trials = VAL_PAIRS_SHA256 if step == "validation" else OFFICIAL_TRIALS_SHA256
    report.add(
        f"{label}: evaluator commit, mode, trial file and model",
        metrics["git_commit"] == EVALUATOR_COMMIT
        and not metrics["git_tracked_dirty"]
        and metrics["mode"] == step
        and metrics["inputs"]["trials"]["sha256"] == trials
        and metrics["run_id"] == model["model_id"]
        and metrics["checkpoint_id"] == model["model_version"],
        f"{metrics['git_commit'][:7]}, {metrics['mode']}, {metrics['run_id']}",
    )
    scores = read_scores(folder / "pair_scores.csv")
    labels = scores["labels"]
    counts = (len(labels), int(labels.sum()), int((labels == 0).sum()))
    report.add(
        f"{label}: trial counts",
        counts == (record["count"], record["positive_count"], record["negative_count"]),
        f"{counts[0]} trials, {counts[1]} same, {counts[2]} different",
    )
    direction = (
        np.array_equal(scores["negative_euclidean"], -scores["euclidean_distance"])
        and (scores["euclidean_distance"] >= 0).all()
        and (np.abs(scores["cosine"]) <= 1 + 1e-6).all()
    )
    report.add(
        f"{label}: score direction (cosine in [-1, 1], AUC input is minus the distance)",
        direction,
    )
    for name in SCORES:
        auc = rank_auc(labels, scores[name])
        report.add(
            f"{label}: {name} AUC",
            close(auc, metrics["metrics"][name]["auc"])
            and close(auc, record["auc"][name])
            and auc > 0.5,
            f"rank formula {auc:.6f}, recorded {record['auc'][name]:.6f}",
        )
    return {"scores": scores, "metrics": metrics}


def check_pair(report, step: str, runs: dict, recorded: dict, names):
    """Both approaches scored the same trials, and the bootstrap reproduces."""
    first, second = (runs[a]["scores"] for a in APPROACHES)
    report.add(
        f"{step}: both approaches scored the same trials in the same order",
        first["keys"] == second["keys"]
        and np.array_equal(first["labels"], second["labels"]),
    )
    for name in names:
        again = paired_bootstrap(first["labels"], first[name], second[name], rank_auc)
        want = recorded[name]
        report.add(
            f"{step}: paired bootstrap of the {name} AUC difference",
            all(close(again[k], want[k]) for k in ("observed", "low", "high"))
            and again["excludes_zero"] == want["excludes_zero"],
            f"{again['observed']:+.4f} [{again['low']:+.4f}, {again['high']:+.4f}]",
        )


def check_validation(report, selection: dict, folder: Path):
    runs = {
        a: check_run(
            report,
            "validation",
            a,
            folder / f"validation-{a}",
            selection["validation"][a],
        )
        for a in APPROACHES
    }
    check_pair(report, "validation", runs, selection["bootstrap"], ["cosine"])
    decision = selection["decision"]
    chosen = select_approach(
        {
            a: rank_auc(runs[a]["scores"]["labels"], runs[a]["scores"]["cosine"])
            for a in APPROACHES
        }
    )
    model = EXPECTED_MODELS[chosen]
    report.add(
        "validation: selected approach has the higher cosine AUC",
        chosen == decision["selected_approach"]
        and model["model_id"] == decision["selected_model_id"]
        and model["model_version"] == decision["selected_model_version"],
        chosen,
    )
    for approach in APPROACHES:
        scores = runs[approach]["scores"]
        rule = far_threshold(scores["labels"], scores["cosine"])
        recorded = selection["validation"][approach]["threshold_selection"]
        point = operating_point(scores["labels"], scores["cosine"], rule["threshold"])
        ok = (
            rule["threshold"] == recorded["threshold"]
            and close(point["far"], recorded["far"])
            and close(point["tar"], recorded["tar"])
            and point["far"] <= FAR_TARGET
            and (rule["lower_far"] is None or rule["lower_far"] > FAR_TARGET)
        )
        if approach == chosen:
            ok = (
                ok
                and rule["threshold"] == decision["threshold"]
                and close(point["far"], decision["validation_far"])
            )
        report.add(
            f"validation {approach}: threshold is the lowest cosine with FAR <= 1%",
            ok,
            f"tau {rule['threshold']:.6f}, FAR {point['far']:.4%} ({point['false_accepts']} of "
            f"{point['negatives']}), TAR {point['tar']:.4%}, next lower FAR {rule['lower_far']:.4%}",
        )


def committed_blob_sha256(commit: str, path: str):
    """SHA-256 of a file as committed, or None outside a checkout that has the commit."""
    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob).hexdigest()


def check_official(
    report, selection_path: Path, selection: dict, summary: dict, folder: Path
):
    selection_sha = file_sha256(selection_path)
    report.add(
        "official: used the committed selection, decision unchanged",
        summary["selection_sha256"] == selection_sha
        and summary["decision_unchanged"] == selection["decision"],
        selection_sha[:12],
    )
    commit = summary["comparison_code"]["commit"]
    committed = committed_blob_sha256(commit, "evidence/FAC-05/selection.json")
    if committed is None:
        report.skip(
            "official: selection.json was committed before official scoring",
            f"commit {commit[:7]} not here",
        )
    else:
        report.add(
            "official: selection.json was committed before official scoring",
            committed == selection_sha
            and summary["created_utc"] > selection["created_utc"],
            f"in {commit[:7]}, validation {selection['created_utc']}, official {summary['created_utc']}",
        )
    runs = {
        a: check_run(
            report, "official", a, folder / f"official-{a}", summary["official"][a]
        )
        for a in APPROACHES
    }
    check_pair(report, "official", runs, summary["bootstrap"], SCORES)
    decision = selection["decision"]
    selected = runs[decision["selected_approach"]]
    point = operating_point(
        selected["scores"]["labels"],
        selected["scores"]["cosine"],
        decision["threshold"],
    )
    recorded = summary["operating_point"]
    roc = roc_point_at(
        selected["metrics"]["metrics"]["cosine"]["roc"], decision["threshold"]
    )
    report.add(
        "official: operating point of the fixed threshold",
        recorded["threshold"] == decision["threshold"]
        and all(
            point[k] == recorded[k]
            for k in ("false_accepts", "negatives", "true_accepts", "positives")
        )
        and close(roc["fpr"], point["far"])
        and close(roc["tpr"], point["tar"]),
        f"FAR {point['far']:.4%} ({point['false_accepts']} of {point['negatives']}), "
        f"TAR {point['tar']:.4%} ({point['true_accepts']} of {point['positives']})",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--selection", type=Path, required=True, help="Committed selection.json"
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        help="Folder with validation-classification/ and validation-metric/",
    )
    parser.add_argument(
        "--official-summary", type=Path, help="Committed official-summary.json"
    )
    parser.add_argument(
        "--official-dir",
        type=Path,
        help="Folder with official-classification/ and official-metric/",
    )
    parser.add_argument("--output", type=Path, help="Write the checks as JSON here")
    args = parser.parse_args(argv)
    if bool(args.official_summary) != bool(args.official_dir):
        parser.error("--official-summary and --official-dir go together")
    selection = json.loads(args.selection.read_text())
    report = Report()
    if args.validation_dir:
        check_validation(report, selection, args.validation_dir)
    else:
        report.skip("validation step", "no --validation-dir")
    if args.official_dir:
        summary = json.loads(args.official_summary.read_text())
        check_official(report, args.selection, selection, summary, args.official_dir)
    else:
        report.skip("official step", "no --official-dir")
    passed = sum(c["ok"] is True for c in report.checks)
    print(
        f"\n{passed} passed, {report.failed} failed, {sum(c['ok'] is None for c in report.checks)} not checked"
    )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "selection_sha256": file_sha256(args.selection),
                    "checks": report.checks,
                },
                indent=2,
            )
            + "\n"
        )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
