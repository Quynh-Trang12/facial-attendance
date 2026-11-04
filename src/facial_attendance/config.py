"""Load shared storage paths without creating data or output directories."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectPaths:
    face_data: Path
    liveness_data: Path
    emotion_data: Path
    checkpoints: Path
    attendance: Path


def load_paths(config_file: Path, *, root: Path) -> ProjectPaths:
    """Resolve relative paths from the repository root; allow external datasets."""
    with config_file.open("rb") as handle:
        section = tomllib.load(handle).get("paths", {})
    fields = ProjectPaths.__dataclass_fields__
    if set(section) != set(fields):
        raise ValueError(f"Expected path keys: {', '.join(fields)}")
    resolved = {}
    for key in fields:
        value = section[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path {key!r} must be a nonempty string")
        path = Path(value).expanduser()
        resolved[key] = (root / path).resolve()
    return ProjectPaths(**resolved)
