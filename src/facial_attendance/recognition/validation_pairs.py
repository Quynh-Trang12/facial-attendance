"""The seeded val_data pair list used to select epochs and thresholds.

The experiment settings fix it: one same-identity pair for every identity that still has
two images, and as many different-identity pairs drawn uniformly. It is built once and
reused for every checkpoint of both approaches; the official trials never select anything.
The file uses the official trial format, so the shared evaluator reads it unchanged.
"""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import random

from .data_loader import ClassificationFaces, FaceLayout, FacePair


def build_validation_pairs(
    dataset: ClassificationFaces, *, seed: int = 42
) -> tuple[FacePair, ...]:
    if dataset.split != "val_data":
        raise ValueError("Validation pairs are built from val_data only")
    by_identity = defaultdict(list)
    for sample in dataset.samples:
        by_identity[sample.identity].append(sample.path)
    identities = sorted(by_identity)
    rng = random.Random(seed)
    positives = [
        FacePair(*rng.sample(sorted(by_identity[identity]), 2), 1)
        for identity in identities
        if len(by_identity[identity]) >= 2
    ]
    if not positives or len(identities) < 2:
        raise ValueError("val_data needs two identities and one with two images")
    # Distinct negatives, so no pair counts twice in the AUC; ordering the two paths
    # makes (a, b) and (b, a) the same pair.
    negatives, seen = [], set()
    attempts = 0
    while len(negatives) < len(positives):
        attempts += 1
        if attempts > 100 * len(positives):
            raise ValueError("Too few identities to draw distinct negative pairs")
        left_id, right_id = rng.sample(identities, 2)
        key = tuple(
            sorted((rng.choice(by_identity[left_id]), rng.choice(by_identity[right_id])))
        )
        if key not in seen:
            seen.add(key)
            negatives.append(FacePair(*key, 0))
    pairs = positives + negatives
    rng.shuffle(pairs)
    return tuple(pairs)


def _relative(layout: FaceLayout, path: Path) -> str:
    return PurePosixPath(Path(path).resolve().relative_to(layout.root)).as_posix()


def write_pairs(file: Path, layout: FaceLayout, pairs) -> str:
    """Write `left right label` rows relative to the dataset root; return the SHA-256."""
    text = "".join(
        f"{_relative(layout, p.left)} {_relative(layout, p.right)} {p.label}\n"
        for p in pairs
    )
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return sha256(text.encode()).hexdigest()


def read_pairs(file: Path, layout: FaceLayout) -> tuple[FacePair, ...]:
    pairs = []
    for line_number, line in enumerate(Path(file).read_text().splitlines(), start=1):
        columns = line.split()
        if len(columns) != 3 or columns[2] not in {"0", "1"}:
            raise ValueError(f"{Path(file).name} line {line_number}: expected `left right 0|1`")
        left, right = (layout.root / PurePosixPath(c) for c in columns[:2])
        pairs.append(FacePair(left.resolve(), right.resolve(), int(columns[2])))
    return tuple(pairs)
