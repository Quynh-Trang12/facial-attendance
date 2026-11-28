"""Read, write and apply the list of classification images the loader must not use.

The list records cleaning decisions; the audit generates it from content hashes. Changing
how the data is cleaned means generating a different list, not changing the loader.
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath

from .data_loader import FaceLayout

DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[3] / "configs/face-exclusions.tsv"
COLUMNS = ("path", "reason", "sha256", "kept_path")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_exclusions(root: Path, file: Path) -> frozenset[Path]:
    """Return the listed paths; any entry that does not match this dataset is an error.

    A listed image that is missing, outside classification_data, changed since the list
    was generated, or whose kept copy is missing means the list belongs to other data.
    """
    root = Path(root).resolve()
    # Official trials stay whole, so only classification images can be dropped.
    base = (root / "classification_data").resolve()
    excluded = set()
    with Path(file).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            problem = None
            if len(columns) != len(COLUMNS):
                problem = f"expected {len(COLUMNS)} tab-separated columns {COLUMNS}"
            else:
                relative, _, digest, kept = columns
                path = (root / PurePosixPath(relative)).resolve()
                if not path.is_relative_to(base) or not path.is_file():
                    problem = f"{relative} is not a classification image of this dataset"
                elif _file_sha256(path) != digest:
                    problem = f"{relative} does not have the listed content hash"
                elif not (root / PurePosixPath(kept)).resolve().is_file():
                    problem = f"kept copy {kept} is missing"
            if problem:
                raise ValueError(f"{Path(file).name} line {line_number}: {problem}")
            excluded.add(path)
    return frozenset(excluded)


def apply_exclusions(layout: FaceLayout, file: Path = DEFAULT_EXCLUSIONS) -> FaceLayout:
    return replace(layout, excluded=read_exclusions(layout.root, file))


def load_face_layout(
    directory: Path, *, exclusions: Path | None = DEFAULT_EXCLUSIONS
) -> FaceLayout:
    """The one entry point for training and checks: find the data, then apply the list.

    Only `exclusions=None` gives the data as published, for audits that generate a list.
    """
    layout = FaceLayout.discover(directory)
    return layout if exclusions is None else apply_exclusions(layout, exclusions)


def write_exclusions(file: Path, entries: list[tuple[str, str, str, str]], header: list[str]):
    """Write rows of COLUMNS once; a new list changes what every model trains on."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("x", encoding="utf-8") as handle:
        handle.writelines(f"# {line}\n" for line in header)
        handle.write(f"# columns: {', '.join(COLUMNS)}\n")
        handle.writelines("\t".join(entry) + "\n" for entry in entries)
