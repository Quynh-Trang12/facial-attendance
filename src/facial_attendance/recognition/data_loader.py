"""Map-style face datasets and reproducible training-pair sampling."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import random


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train_data", "val_data", "test_data")
# Classification images byte-identical to another partition, generated from the full
# data by scripts/audit_face_data.py. Loading the competition as published would leak.
DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[3] / "configs/face-exclusions.tsv"


def _read_exclusions(root: Path, file: Path) -> frozenset[Path]:
    """Read listed paths; a listed file that is absent means this is different data."""
    excluded = set()
    # Official trials stay whole, so only classification images can be dropped.
    base = (root / "classification_data").resolve()
    with file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            relative = PurePosixPath(line.split("\t")[0].strip())
            path = (root / relative).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise ValueError(
                    f"{file.name} line {line_number} is not a classification image "
                    f"of this dataset: {relative}"
                )
            excluded.add(path)
    return frozenset(excluded)


@dataclass(frozen=True)
class FaceLayout:
    root: Path
    excluded: frozenset[Path] = frozenset()

    @classmethod
    def discover(
        cls, directory: Path, *, exclusions: Path | None = DEFAULT_EXCLUSIONS
    ) -> "FaceLayout":
        """Accept the dataset root or one enclosing archive directory, without guessing files.

        Every dataset built from the layout drops the listed images. Only `exclusions=None`
        gives the unfiltered data, for generating the list or for generated fixtures.
        """
        directory = Path(directory).expanduser().resolve()
        candidates = [directory]
        if directory.is_dir():
            candidates += sorted(p for p in directory.iterdir() if p.is_dir())
        matches = []
        for candidate in candidates:
            if (
                all(
                    (candidate / "classification_data" / split).is_dir()
                    for split in SPLITS
                )
                and (candidate / "verification_data").is_dir()
                and (candidate / "verification_pairs_val.txt").is_file()
            ):
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one course dataset root under {directory}; found {len(matches)}. "
                "Required: classification_data/{train_data,val_data,test_data}, "
                "verification_data, verification_pairs_val.txt. Extract archives first; "
                "do not rename a different dataset to make this check pass."
            )
        if exclusions is None:
            return cls(matches[0])
        return cls(matches[0], _read_exclusions(matches[0], Path(exclusions)))

    @property
    def pairs_file(self) -> Path:
        return self.root / "verification_pairs_val.txt"


@dataclass(frozen=True)
class FaceSample:
    path: Path
    identity: str
    target: int


@dataclass(frozen=True)
class FacePair:
    left: Path
    right: Path
    label: int  # 1 = same identity; 0 = different identities.


def _read_image(path: Path, transform: Callable | None):
    from PIL import Image

    try:
        with Image.open(path) as source:
            # Grayscale/RGBA sources must still yield the same three-channel CNN input.
            image = source.convert("RGB")
            return transform(image) if transform is not None else image.copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot decode face image {path}: {error}") from error


def image_transform(image_size: int):
    """Inspection transform only; final model normalization belongs to its training contract."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    from torchvision.transforms import Compose, Resize, ToTensor

    return Compose([Resize((image_size, image_size)), ToTensor()])


def _identity_mapping(
    identities: list[str], split: str, provided: dict | None
) -> dict[str, int]:
    """Build the encoding on train once; reject label drift in other splits."""
    if provided is None:
        if split != "train_data":
            raise ValueError("Validation/test must reuse the training identity mapping")
        # Folder IDs remain strings; sorted enumeration gives a stable target index
        # without assuming numeric IDs are already contiguous class labels.
        provided = {identity: index for index, identity in enumerate(identities)}
    if not all(
        isinstance(k, str) and type(v) is int for k, v in provided.items()
    ) or sorted(provided.values()) != list(range(len(provided))):
        raise ValueError(
            "Identity mapping must contain unique, contiguous integer targets"
        )
    unknown = set(identities) - set(provided)
    if unknown:
        raise ValueError(f"Unknown identities in {split}: {sorted(unknown)}")
    return dict(provided)


def _index_images(
    folder: Path, identities: list[str], mapping: dict[str, int]
) -> tuple[FaceSample, ...]:
    """Index paths/labels only; decode lazily instead of holding 380k images in RAM."""
    folder = folder.resolve()
    samples = []
    for identity in identities:
        files = sorted(
            p
            for p in (folder / identity).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise ValueError(
                f"Identity {identity!r} has no image files in {folder.name}"
            )
        for path in files:
            resolved = path.resolve()
            # A symlink must not silently bring images from another split/storage tree.
            if not resolved.is_relative_to(folder):
                raise ValueError(f"Image path escapes its classification split: {path}")
            samples.append(FaceSample(resolved, identity, mapping[identity]))
    return tuple(samples)


class ClassificationFaces:
    """Map-style dataset returning (RGB image/transformed tensor, integer target)."""

    def __init__(
        self,
        layout: FaceLayout,
        split: str,
        *,
        class_to_idx: dict[str, int] | None = None,
        transform: Callable | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown classification split: {split}")
        self.layout, self.split, self.transform = layout, split, transform
        folder = layout.root / "classification_data" / split
        identities = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if not identities:
            raise ValueError(f"No identity folders in {folder}")
        self.class_to_idx = _identity_mapping(identities, split, class_to_idx)
        indexed = _index_images(folder, identities, self.class_to_idx)
        self.samples = tuple(s for s in indexed if s.path not in layout.excluded)
        self.excluded_count = len(indexed) - len(self.samples)
        # The mapping keeps every folder, so a class left without training images would
        # be an untrainable target. A held-out identity may lose its images.
        emptied = set(identities) - {s.identity for s in self.samples}
        if split == "train_data" and emptied:
            raise ValueError(f"Exclusions empty training identities: {sorted(emptied)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return _read_image(sample.path, self.transform), sample.target


def _verification_path(layout: FaceLayout, text: str, line_number: int) -> Path:
    relative = PurePosixPath(text.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in text:
        raise ValueError(f"Unsafe verification path at line {line_number}: {text}")
    base = (layout.root / "verification_data").resolve()
    # Accept both official prefixed paths and bare names, but never images from
    # classification splits or arbitrary files outside verification_data.
    candidates = {(layout.root / relative).resolve(), (base / relative).resolve()}
    matches = [p for p in candidates if p.is_relative_to(base) and p.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Missing or ambiguous verification image at line {line_number}: {text}"
        )
    return matches[0]


def read_verification_pairs(layout: FaceLayout) -> tuple[FacePair, ...]:
    pairs = []
    # Keep official order/labels so future scores align with the published trials.
    with layout.pairs_file.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            columns = line.split()
            if len(columns) != 3 or columns[2] not in {"0", "1"}:
                raise ValueError(
                    f"Expected two image paths and label 0/1 at line {line_number}"
                )
            pairs.append(
                FacePair(
                    _verification_path(layout, columns[0], line_number),
                    _verification_path(layout, columns[1], line_number),
                    int(columns[2]),
                )
            )
    if not pairs:
        raise ValueError("Verification trials file is empty")
    return tuple(pairs)


class PairFaces:
    """Return (left RGB image/tensor, right RGB image/tensor, same/different label)."""

    def __init__(
        self, pairs: tuple[FacePair, ...], *, transform: Callable | None = None
    ):
        if not pairs:
            raise ValueError("Pair dataset is empty")
        self.pairs = tuple(pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        return (
            _read_image(pair.left, self.transform),
            _read_image(pair.right, self.transform),
            pair.label,
        )


class VerificationPairs(PairFaces):
    """Load official trials without resampling or reordering their labels."""

    def __init__(self, layout: FaceLayout, *, transform: Callable | None = None):
        super().__init__(read_verification_pairs(layout), transform=transform)


def sample_training_pairs(
    dataset: ClassificationFaces, *, count_per_label: int, seed: int = 42
) -> tuple[FacePair, ...]:
    """Draw a balanced reproducible training sample; official verification trials stay fixed.

    Identities are sampled uniformly, then images uniformly within the selected identity.
    Positive images are distinct. Sampling is with replacement across pairs, so repeated
    pairs are possible. This is an inspectable baseline, not hard-negative mining.
    """
    # Sampling from held-out splits would mix evaluation data into training.
    if dataset.split != "train_data":
        raise ValueError("Training pairs may only be sampled from train_data")
    if count_per_label < 1:
        raise ValueError("count_per_label must be positive")
    by_identity = defaultdict(list)
    for sample in dataset.samples:
        by_identity[sample.identity].append(sample.path)
    identities = sorted(by_identity)
    # A singleton can contribute a negative, but cannot form a distinct positive.
    positive_identities = [
        identity for identity in identities if len(by_identity[identity]) >= 2
    ]
    if len(identities) < 2 or not positive_identities:
        raise ValueError(
            "Pair sampling needs two identities and an identity with at least two images"
        )
    # A local RNG preserves reproducibility without changing the model-training RNG.
    rng = random.Random(seed)
    pairs = []
    for _ in range(count_per_label):
        # Draw identities uniformly so large identity folders do not dominate pairs.
        identity = rng.choice(positive_identities)
        left, right = rng.sample(by_identity[identity], 2)
        pairs.append(FacePair(left, right, 1))
        left_id, right_id = rng.sample(identities, 2)
        pairs.append(
            FacePair(
                rng.choice(by_identity[left_id]), rng.choice(by_identity[right_id]), 0
            )
        )
    rng.shuffle(pairs)
    return tuple(pairs)
