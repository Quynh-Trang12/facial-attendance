"""Read dataset files with consistent path, JSON and checksum validation."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def local_path(root: Path, value: str) -> Path:
    """Resolve symbolic links and reject paths outside the dataset root."""
    relative = PurePosixPath(value.replace("\\", "/"))
    if not value or relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError(f"Unsafe dataset path: {value!r}")

    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside dataset: {value}")

    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value

    return result


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object
    )
