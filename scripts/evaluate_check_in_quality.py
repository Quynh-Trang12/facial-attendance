"""Offline evidence for the quality-gated multi-frame check-in.

Three steps, run in this order on Colab (the images are there):

  quality   Measure every image of a trial list with recognition.quality.measure.
  discard   validation: compare each quality measure's error-versus-discard curve with
            random discard on the validation pairs and choose one measure.
            official: apply that choice, and only that choice decides, on the
            official trials. It refuses to run without the validation record.
  kshot     Embed the val_data and test_data images with the deployed checkpoint and
            compare a fused k-image probe with a single image (k = 1, 2, 3).

Scores come from the FAC-05 pair_scores.csv files, so discard needs no network.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facial_attendance.recognition.check_in_evaluation import (  # noqa: E402
    FRACTIONS,
    MEASURES,
    discard_curve,
    image_scores,
    kshot,
    limit_percentiles,
    pair_quality,
    random_discard,
    select_measure,
)

QUALITY_COLUMNS = ("path", "sharpness", "brightness", "contrast", "size")
CAVEAT = ("val_data and test_data hold other images of the 4,000 training identities, so absolute "
          "rates are optimistic; the claim is the change between k values at the same tau.")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_directory(parser, path: Path) -> Path:
    if path.exists():
        parser.error(f"{path} exists; choose a new output directory, earlier results are kept")
    path.mkdir(parents=True)
    return path


def load_layout(args):
    from facial_attendance.config import load_paths
    from facial_attendance.recognition.data_loader import FaceLayout
    from facial_attendance.recognition.download import download_face_data
    from facial_attendance.recognition.exclusions import apply_exclusions

    paths = load_paths(args.paths_config, root=ROOT)
    layout = (download_face_data(cache_dir=paths.kaggle_cache) if args.download
              else FaceLayout.discover(args.data_root or paths.face_data))
    return apply_exclusions(layout, args.exclusions)


def read_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run_quality(args, parser) -> int:
    from facial_attendance.recognition.embeddings import trial_keys
    from facial_attendance.recognition.quality import measure

    if args.output.exists():
        parser.error(f"{args.output} exists; earlier results are kept")
    layout = load_layout(args)
    trials = layout.pairs_file if args.trials == "official" else Path(args.trials)
    keys = trial_keys(layout, trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(QUALITY_COLUMNS)
        for index, (text, path) in enumerate(keys, start=1):
            quality = measure(read_rgb(path))
            writer.writerow([text, quality.sharpness, quality.brightness, quality.contrast, quality.size])
            if index % 2000 == 0:
                print(f"{index} of {len(keys)} images", flush=True)
    summary = {"trials": str(args.trials), "trials_sha256": sha256(trials), "images": len(keys),
               "quality_sha256": sha256(args.output)}
    Path(f"{args.output}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def read_pairs(path: Path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(row["left"], row["right"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["cosine"]) for row in rows]
    return pairs, labels, scores


def read_quality(path: Path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != QUALITY_COLUMNS:
        raise ValueError(f"{path} must have the columns {', '.join(QUALITY_COLUMNS)}")
    return rows


def plot(curves, baseline, highlight, title, file: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    fractions = [100 * point["fraction"] for point in baseline]
    tar = [point["tar"] for point in baseline]
    spread = [point["tar_std"] for point in baseline]
    axis.plot(fractions, tar, color="grey", linestyle="--", label="random discard (mean of seeds)")
    axis.fill_between(fractions, [t - s for t, s in zip(tar, spread)], [t + s for t, s in zip(tar, spread)],
                      color="grey", alpha=0.2)
    for name, curve in curves.items():
        axis.plot([100 * p["fraction"] for p in curve], [p["tar"] for p in curve], marker="o",
                  linewidth=2.5 if name == highlight else 1, alpha=1 if name == highlight else 0.5, label=name)
    axis.set_xlabel("Lowest-quality pairs set aside (%)")
    axis.set_ylabel("True-accept rate at tau")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(file, dpi=150)
    plt.close(figure)


def run_discard(args, parser) -> int:
    if args.step == "official" and args.selection is None:
        parser.error("official needs --selection, the discard-validation.json chosen beforehand")
    selection = json.loads(args.selection.read_text()) if args.selection else None
    if selection is not None and selection.get("step") != "validation":
        parser.error("--selection must be a discard-validation.json")
    output = new_directory(parser, args.output)

    pairs, labels, scores = read_pairs(args.pair_scores)
    rows = read_quality(args.quality)
    by_measure = image_scores(rows)
    curves = {name: discard_curve(labels, scores, pair_quality(pairs, by_measure[name]), args.tau)
              for name in MEASURES}
    baseline = random_discard(labels, scores, args.tau, seeds=args.seeds)
    record = {
        "step": args.step,
        "tau": args.tau,
        "pairs": len(pairs),
        "fractions": list(FRACTIONS),
        "pair_scores_sha256": sha256(args.pair_scores),
        "quality_sha256": sha256(args.quality),
        "random_discard": baseline,
    }
    if args.step == "validation":
        choice = select_measure(curves, baseline)
        record |= {"selection": choice, "curves": curves, "limits_start": limit_percentiles(rows)}
        measure = choice["measure"]
    else:
        measure = selection["selection"]["measure"]
        record |= {
            "selection_sha256": sha256(args.selection),
            "measure": measure,
            "selected_curve": curves[measure],
            # Reported for transparency only; the choice was fixed on the validation pairs.
            "other_curves": {name: curve for name, curve in curves.items() if name != measure},
        }
    (output / f"discard-{args.step}.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.no_plot:
        shown = curves if args.step == "validation" else {measure: curves[measure]}
        plot(shown, baseline, measure, f"Error versus discard, {args.step} ({len(pairs):,} pairs)",
             output / f"discard-{args.step}.png")
    start = curves[measure][0]
    print(f"{args.step}: measure {measure}; at 0 % TAR {start['tar']:.4f}, FAR {start['far']:.4f}, "
          f"AUC {start['auc']:.4f}")
    for point, base in zip(curves[measure], baseline):
        print(f"  drop {100 * point['fraction']:4.0f} %  TAR {point['tar']:.4f} (random {base['tar']:.4f})  "
              f"FAR {point['far']:.4f}")
    return 0


def run_kshot(args, parser) -> int:
    import numpy as np
    import torch

    from facial_attendance.recognition.data_loader import IMAGE_SUFFIXES
    from facial_attendance.recognition.embeddings import embed_paths
    from facial_attendance.recognition.model import load_checkpoint

    output = new_directory(parser, args.output)
    layout = load_layout(args)
    images = {}
    for split in ("val_data", "test_data"):
        for folder in sorted((layout.root / "classification_data" / split).iterdir()):
            if folder.is_dir():
                images.setdefault(folder.name, []).extend(
                    sorted(p for p in folder.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES and p.resolve() not in layout.excluded))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(args.checkpoint, device)
    if meta.get("threshold") is None:
        parser.error("The checkpoint has no deployed threshold; use deploy.pt")
    paths = [path for name in sorted(images) for path in images[name]]
    embeddings = embed_paths(model, meta, paths, device=device)
    by_identity, start = {}, 0
    for name in sorted(images):
        by_identity[name] = embeddings[start : start + len(images[name])]
        start += len(images[name])
    result = kshot(by_identity, meta["threshold"])
    record = {
        "model_id": meta["model_id"],
        "model_version": meta["model_version"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "images_embedded": len(paths),
        "caveat": CAVEAT,
    } | result
    (output / "kshot.json").write_text(json.dumps(record, indent=2) + "\n")
    relative = [str(p.relative_to(layout.root)) for p in paths]
    np.savez(output / "kshot-embeddings.npz", paths=np.array(relative), raw_embeddings=embeddings)
    for k, entry in result["k"].items():
        mean = entry["mean"]
        print(f"k={k}: TAR {mean['tar']:.4f}  FAR {mean['far']:.4f}  AUC {mean['auc']:.4f}")
    print(f"{result['identities']} identities, {result['skipped_identities']} skipped (fewer than "
          f"{result['images_per_identity']} images)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="command", required=True)

    def data_options(sub):
        from facial_attendance.recognition.exclusions import DEFAULT_EXCLUSIONS

        source = sub.add_mutually_exclusive_group()
        source.add_argument("--download", action="store_true")
        source.add_argument("--data-root", type=Path)
        sub.add_argument("--paths-config", type=Path, default=ROOT / "configs/project.toml")
        sub.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)

    quality = steps.add_parser("quality", help="measure the images of a trial list")
    data_options(quality)
    quality.add_argument("--trials", required=True, help='"official" or a trial file such as val-pairs.txt')
    quality.add_argument("--output", type=Path, required=True, help="New CSV file")

    discard = steps.add_parser("discard", help="error versus discard")
    discard.add_argument("step", choices=("validation", "official"))
    discard.add_argument("--pair-scores", type=Path, required=True, help="FAC-05 pair_scores.csv of the metric model")
    discard.add_argument("--quality", type=Path, required=True, help="CSV from the quality step")
    discard.add_argument("--tau", type=float, required=True)
    discard.add_argument("--selection", type=Path, help="discard-validation.json (official step only)")
    discard.add_argument("--seeds", type=int, default=20)
    discard.add_argument("--output", type=Path, required=True, help="New directory")
    discard.add_argument("--no-plot", action="store_true")

    shot = steps.add_parser("kshot", help="single image against a fused k-image probe")
    data_options(shot)
    shot.add_argument("--checkpoint", type=Path, required=True, help="deploy.pt")
    shot.add_argument("--output", type=Path, required=True, help="New directory")
    shot.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    return {"quality": run_quality, "discard": run_discard, "kshot": run_kshot}[args.command](args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
