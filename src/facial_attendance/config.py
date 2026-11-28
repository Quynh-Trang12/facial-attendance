"""Load shared storage paths without creating data or output directories."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Shared storage locations, resolved independently of the working directory."""

    kaggle_cache: Path
    face_data: Path
    liveness_data: Path
    emotion_data: Path
    checkpoints: Path
    attendance: Path
    camera_trials: Path = Path("data/liveness/camera-trials")
    detector_model: Path = Path("artifacts/detectors/face_detection_yunet_2023mar.onnx")


def load_paths(config_file: Path, *, root: Path, overrides=None) -> ProjectPaths:
    """Resolve relative paths from the repository root; allow external datasets."""
    with config_file.open("rb") as handle:
        section = tomllib.load(handle).get("paths", {})
    section.update(
        {
            key: str(value)
            for key, value in (overrides or {}).items()
            if value is not None
        }
    )
    # Reject misspelled or omitted keys rather than silently using another dataset.
    fields = ProjectPaths.__dataclass_fields__
    optional = {"camera_trials", "detector_model"}
    if set(section) - set(fields) or (set(fields) - optional) - set(section):
        raise ValueError(f"Expected path keys: {', '.join(fields)}")
    resolved = {}
    for key in fields:
        value = section.get(key, str(fields[key].default))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path {key!r} must be a nonempty string")
        path = Path(value).expanduser()
        # pathlib leaves absolute paths intact, so Drive/external disks also work.
        resolved[key] = (root / path).resolve()
    return ProjectPaths(**resolved)
