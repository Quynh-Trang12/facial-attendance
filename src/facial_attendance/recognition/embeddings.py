"""Use a saved face checkpoint: embed images, score two images, export evaluator inputs.

Exports follow the shared evaluator's input contract: raw embeddings before L2
normalization in an NPZ keyed by the exact path strings of a trial file, and
`id,true,predicted` CSVs with a JSON label map for classification accuracy.
"""

import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .data_loader import ClassificationFaces, FaceLayout, _read_image, _verification_path
from .model import eval_transform


def _transform(meta: dict):
    return eval_transform(meta["input_size"])


@torch.no_grad()
def embed_paths(model, meta, paths, *, device="cpu", batch_size=256) -> np.ndarray:
    """Raw float32[N, D] embeddings from the layer before the head, in input order."""
    transform = _transform(meta)
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack([_read_image(Path(p), transform) for p in paths[start : start + batch_size]])
        chunks.append(model.embed(images.to(device)).float().cpu())
    if not chunks:
        return np.zeros((0, meta["embedding_dim"]), dtype=np.float32)
    return torch.cat(chunks).numpy().astype(np.float32)


def score_pair(left: np.ndarray, right: np.ndarray) -> dict:
    """Cosine similarity (higher = same) and Euclidean distance (lower = same), raw vectors."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if norms == 0:
        raise ValueError("A zero embedding has no direction to compare")
    return {
        "cosine": float(left @ right / norms),
        "euclidean_distance": float(np.linalg.norm(left - right)),
        "euclidean_distance_normalized": float(
            np.linalg.norm(left / np.linalg.norm(left) - right / np.linalg.norm(right))
        ),
    }


def score_images(model, meta, left: Path, right: Path, *, device="cpu") -> dict:
    embeddings = embed_paths(model, meta, [left, right], device=device)
    return {"left": str(left), "right": str(right), "embedding_dim": embeddings.shape[1]} | score_pair(*embeddings)


def trial_keys(layout: FaceLayout, trials_file: Path) -> list[tuple[str, Path]]:
    """Unique path strings of a trial file, as written, with the image each one names.

    Official trials resolve like the loader does; other lists (the validation pairs) are
    relative to the dataset root.
    """
    official = Path(trials_file).resolve() == layout.pairs_file.resolve()
    keys = {}
    for line_number, line in enumerate(Path(trials_file).read_text().splitlines(), start=1):
        columns = line.split()
        if not columns or line.lstrip().startswith("#"):
            continue
        for text in columns[:2]:
            if text not in keys:
                path = (
                    _verification_path(layout, text, line_number)
                    if official
                    else (layout.root / PurePosixPath(text)).resolve()
                )
                if not path.is_file():
                    raise ValueError(f"{trials_file} line {line_number}: missing image {text}")
                keys[text] = path
    return list(keys.items())


def export_trial_embeddings(model, meta, layout, trials_file, output: Path, *, device="cpu", batch_size=256) -> dict:
    keys = trial_keys(layout, trials_file)
    raw = embed_paths(model, meta, [path for _, path in keys], device=device, batch_size=batch_size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:  # never overwrite an earlier export
        np.savez(handle, paths=np.array([text for text, _ in keys]), raw_embeddings=raw)
    return {"file": str(output), "images": len(keys), "embedding_dim": int(raw.shape[1])}


@torch.no_grad()
def export_predictions(model, meta, layout, split: str, output_dir: Path, *, device="cpu", batch_size=256) -> dict:
    """Top-1 predictions on val_data or test_data with the train-derived label map."""
    train = ClassificationFaces(layout, "train_data")
    dataset = ClassificationFaces(layout, split, class_to_idx=train.class_to_idx, transform=_transform(meta))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {index: identity for identity, index in train.class_to_idx.items()}
    with (output_dir / "face_labels.json").open("w") as handle:
        json.dump({str(k): v for k, v in sorted(labels.items())}, handle)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    predicted = []
    for images, _ in loader:
        predicted += model(images.to(device)).argmax(1).cpu().tolist()
    correct = 0
    with (output_dir / f"{split}-predictions.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "true", "predicted"])
        for sample, guess in zip(dataset.samples, predicted):
            writer.writerow([sample.path.relative_to(layout.root).as_posix(), sample.target, guess])
            correct += sample.target == guess
    return {"split": split, "images": len(dataset), "top1": correct / len(dataset)}
