"""Run LIV-03 quality stages in order; preserve existing data and evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Quality scanning requires the full project environment, unlike basic CSV reads.
from facial_attendance.liveness.quality_audit import scan, exact_duplicates
from facial_attendance.liveness.near_search import complete_near_search
from facial_attendance.liveness.experimental_readiness import prepare_experiment
from facial_attendance.liveness.quality_review import (
    review_near, provenance, visual_queue, calibrate_search, preview_review, preview_quarantine, assess_reviews,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("exact", "near", "near-complete", "provenance", "visual", "preview", "assess", "experimental", "experimental-preview"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-description", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, help="Optional diagnostic cap; capped search cannot pass acceptance. Default: no cap")
    parser.add_argument("--search-output", type=Path, help="New directory for the complete compact candidate graph")
    parser.add_argument("--review-output", type=Path, help="New assessment JSON; defaults to quality.json in the run")
    parser.add_argument("--supporting-evidence", type=Path, action="append", default=[],
                        help="Bind a review attachment to the assessment; repeat for multiple files")
    parser.add_argument("--candidate-dir", type=Path, help="Complete graph directory for experimental readiness")
    parser.add_argument("--near-reviews", type=Path, help="Attributed natural-pair review JSON")
    parser.add_argument("--visual-review", type=Path, help="Visual review CSV; confirmed rejections are excluded")
    parser.add_argument("--experiment-output", type=Path, help="New experimental subset and readiness directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--preview-offset", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--preview-output", type=Path, help="New PNG for an experimental quarantine review")
    parser.add_argument("--anchor-index", type=int, default=0)
    args = parser.parse_args(argv)
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if args.stage == "experimental-preview":
        if args.preview_output is None:
            parser.error('experimental-preview requires --preview-output')
        print(preview_quarantine(root, args.inventory, output, args.preview_output,
                                 anchor_index=args.anchor_index, offset=args.preview_offset, count=args.preview_count))
        return 0
    elif args.stage == "experimental":
        if args.candidate_dir is None or args.experiment_output is None:
            parser.error('experimental requires --candidate-dir and --experiment-output')
        result = prepare_experiment(root, args.inventory, args.data_description, output,
                                    args.candidate_dir, args.experiment_output,
                                    near_reviews=args.near_reviews, visual_review=args.visual_review,
                                    batch_size=args.batch_size, image_size=args.image_size,
                                    supporting_evidence=args.supporting_evidence)
    elif args.stage == "exact":
        scan(root, args.inventory, args.data_description, output, workers=args.workers)
        result = exact_duplicates(output)
    else:
        required = {"near": "exact.json", "near-complete": "exact.json", "provenance": "near.json", "visual": "provenance.json",
                    "preview": "visual.json", "assess": "visual.json"}
        if not (output / required[args.stage]).is_file():
            parser.error(f"Run the preceding stage first: missing {required[args.stage]}")
        if args.stage == "near-complete":
            if args.search_output is None:
                parser.error('near-complete requires --search-output')
            result = complete_near_search(output, args.search_output)
        elif args.stage == "near":
            result = review_near(output, max_pairs=args.max_pairs)
            calibrate_search(output)
        elif args.stage == "provenance":
            result = provenance(root, args.inventory, args.data_description, output)
        elif args.stage == "visual":
            result = visual_queue(args.inventory, output)
        elif args.stage == "preview":
            print(preview_review(root, output, offset=args.preview_offset, count=args.preview_count))
            return 0
        else:
            result = assess_reviews(root, args.inventory, args.data_description, output,
                                    args.review_output or output / 'quality.json',
                                    supporting_evidence=args.supporting_evidence)
    print(json.dumps(result, indent=2))
    return 0 if result["result"] == "PASS" else 1 if result["result"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
