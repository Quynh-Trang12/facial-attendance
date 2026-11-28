"""Score both face approaches with the shared evaluator, in the order the comparison fixes.

Step 1, validation: check both exports, score the validation pairs, select the approach
and the cosine threshold, and write selection.json. Commit that file before step 2.

    python scripts/compare_face_approaches.py validation --evaluator-root <sys06 checkout> \
        --classification-export <dir> --metric-export <dir> --data-root <dataset> \
        --val-pairs <val-pairs.txt> --output <new dir>

Step 2, official: score the official trials once with the committed selection.json,
whose SHA-256 must be given, and report the fixed threshold's operating point.

    python scripts/compare_face_approaches.py official ... --selection selection.json \
        --selection-sha256 <hash of the committed file> --output <new dir>

The evaluator runs unchanged from its own checkout, so its metrics.json records its
commit and source hashes. Nothing here loads a model.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.comparison import (  # noqa: E402 — source checkout bootstrap precedes local imports.
    APPROACHES,
    EXPECTED_MODELS,
    OFFICIAL_TRIALS_SHA256,
    SCORES,
    VAL_PAIRS_SHA256,
    check_export,
    file_sha256,
    operating_point,
    paired_bootstrap,
    read_pair_scores,
    roc_point_at,
    select_approach,
)

EVALUATOR_COMMIT = "2d14220e912acfbdda8f378e94f81c9bafbbf3a6"
TRIALS_NAME = "verification_pairs_val.txt"


def load_evaluator_metrics(root: Path):
    """The evaluator's metrics module by file path, so its package does not shadow this one."""
    spec = importlib.util.spec_from_file_location(
        "sys06_metrics", root / "src/facial_attendance/evaluation/metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def code_state() -> dict:
    """This repository's commit, and whether tracked code differs from it."""
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    return {"commit": git_commit(ROOT), "tracked_dirty": bool(dirty.strip())}


def evaluate(
    root: Path,
    approach: str,
    export: Path,
    trials: Path,
    mode: str,
    output: Path,
    *,
    select=False,
) -> dict:
    model = EXPECTED_MODELS[approach]
    archive = export / (
        "val-pairs-embeddings.npz"
        if mode == "validation"
        else "official-embeddings.npz"
    )
    command = [
        sys.executable,
        str(root / "scripts/evaluate.py"),
        "verification",
        "--trials",
        str(trials),
        "--embeddings",
        str(archive),
        "--approach",
        approach,
        "--mode",
        mode,
        "--run-id",
        model["model_id"],
        "--checkpoint-id",
        model["model_version"],
        "--output",
        str(output),
    ]
    if select:
        command.append("--select-threshold")
    subprocess.run(command, check=True)
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics["git_commit"] != EVALUATOR_COMMIT or metrics["git_tracked_dirty"]:
        raise ValueError(
            f"Evaluator ran at {metrics['git_commit']} (dirty={metrics['git_tracked_dirty']}), "
            f"expected clean {EVALUATOR_COMMIT}"
        )
    want = VAL_PAIRS_SHA256 if mode == "validation" else OFFICIAL_TRIALS_SHA256
    if metrics["inputs"]["trials"]["sha256"] != want:
        raise ValueError(
            f"{mode} trial file hash {metrics['inputs']['trials']['sha256']} is not {want}"
        )
    return metrics


def run_record(metrics: dict, output: Path) -> dict:
    return {
        "auc": {name: metrics["metrics"][name]["auc"] for name in SCORES},
        "count": metrics["metrics"]["cosine"]["count"],
        "positive_count": metrics["metrics"]["cosine"]["positive_count"],
        "negative_count": metrics["metrics"]["cosine"]["negative_count"],
        "output": output.name,
        "metrics_sha256": file_sha256(output / "metrics.json"),
        "pair_scores_sha256": file_sha256(output / "pair_scores.csv"),
    }


def bootstraps(m, outputs: dict, names) -> dict:
    """Paired bootstrap of AUC(metric) - AUC(classification) for each named score."""
    first, second = (
        read_pair_scores(outputs[a] / "pair_scores.csv") for a in APPROACHES
    )
    if first["keys"] != second["keys"] or (first["labels"] != second["labels"]).any():
        raise ValueError(
            "The two approaches did not score the same trials in the same order"
        )

    def auc(labels, scores):
        return m.binary_metrics(labels, scores)["auc"]

    return {
        name: paired_bootstrap(first["labels"], first[name], second[name], auc)
        | {"statistic": "AUC(metric) - AUC(classification)"}
        for name in names
    }


def common(args):
    """Checks shared by both steps; nothing is written until they pass."""
    trials = args.data_root / TRIALS_NAME
    if (
        file_sha256(trials) != OFFICIAL_TRIALS_SHA256
        or file_sha256(args.val_pairs) != VAL_PAIRS_SHA256
    ):
        raise ValueError(
            "The trial file or the validation pairs are not the fixed versions"
        )
    if (commit := git_commit(args.evaluator_root)) != EVALUATOR_COMMIT:
        raise ValueError(
            f"Evaluator checkout is at {commit}, expected {EVALUATOR_COMMIT}"
        )
    exports = {
        "classification": args.classification_export,
        "metric": args.metric_export,
    }
    checks = {
        a: check_export(exports[a], a, official_trials=trials, val_pairs=args.val_pairs)
        for a in APPROACHES
    }
    args.output.mkdir(parents=True)
    return trials, exports, checks


def cmd_validation(args) -> dict:
    trials, exports, checks = common(args)
    m = load_evaluator_metrics(args.evaluator_root)
    outputs, runs = {}, {}
    for approach in APPROACHES:
        outputs[approach] = args.output / f"validation-{approach}"
        metrics = evaluate(
            args.evaluator_root,
            approach,
            exports[approach],
            args.val_pairs,
            "validation",
            outputs[approach],
            select=True,
        )
        runs[approach] = run_record(metrics, outputs[approach]) | {
            "threshold_selection": metrics["selection"]
        }
        runs[approach]["training_cosine_auc"] = EXPECTED_MODELS[approach][
            "training_val_pairs_cosine_auc"
        ]
    selected = select_approach({a: runs[a]["auc"]["cosine"] for a in APPROACHES})
    chosen = runs[selected]["threshold_selection"]
    record = {
        "step": "validation",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluator_commit": EVALUATOR_COMMIT,
        "comparison_code": code_state(),
        "val_pairs_sha256": VAL_PAIRS_SHA256,
        "exports": checks,
        "validation": runs,
        "bootstrap": bootstraps(m, outputs, ["cosine"]),
        "decision": {
            "rule": "higher validation-pair cosine AUC",
            "selected_approach": selected,
            "selected_model_id": EXPECTED_MODELS[selected]["model_id"],
            "selected_model_version": EXPECTED_MODELS[selected]["model_version"],
            "measure": "cosine similarity of raw embeddings, accept when score >= threshold",
            "threshold": chosen["threshold"],
            "threshold_rule": "lowest validation cosine with FAR <= 1%",
            "validation_far": chosen["far"],
            "validation_tar": chosen["tar"],
            "reject_all": chosen["reject_all"],
        },
    }
    (args.output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def cmd_official(args) -> dict:
    if file_sha256(args.selection) != args.selection_sha256:
        raise ValueError(
            "selection.json differs from the committed record; official scoring refused"
        )
    selection = json.loads(args.selection.read_text())
    trials, exports, checks = common(args)
    for approach in APPROACHES:
        if checks[approach]["files"] != selection["exports"][approach]["files"]:
            raise ValueError(f"The {approach} export changed after the validation step")
    m = load_evaluator_metrics(args.evaluator_root)
    outputs, runs = {}, {}
    for approach in APPROACHES:
        outputs[approach] = args.output / f"official-{approach}"
        metrics = evaluate(
            args.evaluator_root,
            approach,
            exports[approach],
            trials,
            "official",
            outputs[approach],
        )
        runs[approach] = run_record(metrics, outputs[approach])
        runs[approach]["roc_metrics"] = metrics["metrics"]
    decision = selection["decision"]
    selected = decision["selected_approach"]
    scores = read_pair_scores(outputs[selected] / "pair_scores.csv")
    point = operating_point(scores["labels"], scores["cosine"], decision["threshold"])
    roc = roc_point_at(
        runs[selected]["roc_metrics"]["cosine"]["roc"], decision["threshold"]
    )
    if abs(roc["fpr"] - point["far"]) > 1e-12 or abs(roc["tpr"] - point["tar"]) > 1e-12:
        raise ValueError(
            f"ROC point {roc} disagrees with the counted operating point {point}"
        )
    for approach in APPROACHES:
        del runs[approach]["roc_metrics"]
    record = {
        "step": "official",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluator_commit": EVALUATOR_COMMIT,
        "comparison_code": code_state(),
        "official_trials_sha256": OFFICIAL_TRIALS_SHA256,
        "selection_sha256": args.selection_sha256,
        "decision_unchanged": decision,
        "official": runs,
        "bootstrap": bootstraps(m, outputs, SCORES),
        "operating_point": {"approach": selected, **point, "roc_point": roc},
    }
    (args.output / "official-summary.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    for step, run in (("validation", cmd_validation), ("official", cmd_official)):
        p = sub.add_parser(step)
        p.add_argument(
            "--evaluator-root",
            type=Path,
            required=True,
            help=f"Checkout at {EVALUATOR_COMMIT[:7]}",
        )
        p.add_argument("--classification-export", type=Path, required=True)
        p.add_argument("--metric-export", type=Path, required=True)
        p.add_argument(
            "--data-root",
            type=Path,
            required=True,
            help=f"Folder holding {TRIALS_NAME}",
        )
        p.add_argument("--val-pairs", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True, help="New directory")
        if step == "official":
            p.add_argument("--selection", type=Path, required=True)
            p.add_argument("--selection-sha256", required=True)
        p.set_defaults(run=run)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory; earlier results are kept")
    record = args.run(args)
    runs = record.get("validation") or record["official"]
    shown = {
        "auc": {a: runs[a]["auc"] for a in APPROACHES},
        "bootstrap": record["bootstrap"],
        "decision": record.get("decision") or record["decision_unchanged"],
    }
    if "operating_point" in record:
        shown["operating_point"] = record["operating_point"]
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
