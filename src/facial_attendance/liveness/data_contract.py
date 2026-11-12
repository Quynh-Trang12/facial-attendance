"""CelebA-Spoof label mapping and per-image inventory contract."""

from dataclasses import dataclass


SPLITS = ("train", "validation", "test")
FAMILY_NAMES = ("live", "print", "screen")
SOURCE = "https://github.com/ZhangYuanhan-AI/CelebA-Spoof"
FAMILIES = {
    0: "live",
    1: "print",
    2: "print",
    3: "print",
    7: "screen",
    8: "screen",
    9: "screen",
}
EXCLUDED = {4, 5, 6, 10}


@dataclass(frozen=True)
class LivenessSample:
    relative_path: str
    subject_id: str
    split: str
    source_type: int
    binary_label: int
    attack_family: str
    sha256: str
    bbox_sha256: str
    source_media_id: str = ""
    session_id: str = ""
    exclusion_reason: str = ""
    original_split: str = ""


def label_family(source_type: int, binary_label: int, relative_path: str) -> str:
    if source_type not in FAMILIES and source_type not in EXCLUDED:
        raise ValueError(f"Unknown source type {source_type}: {relative_path}")
    if binary_label != int(source_type != 0):
        raise ValueError(f"Conflicting source type/binary labels: {relative_path}")

    return FAMILIES.get(source_type, "outside_scope")
