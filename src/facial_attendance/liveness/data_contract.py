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
QUALITY_REQUIREMENTS = (
    "exact_deduplication", "near_duplicate_review", "source_relationship_coverage",
    "known_common_media", "visual_labels_and_boxes", "loader_provenance",
    "face_input_integrity", "selected_family_coverage",
)
TASK_REQUIREMENTS = ("labels_preprocessing_splits", "sample_batch_alignment_and_traceability", *QUALITY_REQUIREMENTS)
EXPERIMENT_REQUIREMENTS = (
    "source_annotation_mapping", "selected_input_integrity", "partition_integrity",
    "selected_family_coverage", "exact_deduplication", "near_candidate_policy",
    "sample_batch_alignment_and_traceability",
)
EXPERIMENT_IMPLEMENTATION_FILES = (
    'data_contract.py', 'data_loader.py', 'dataset_files.py', 'image_inventory.py',
    'face_crop.py', 'split_rules.py', 'dataset_audit.py', 'audit_report.py',
    'quality_audit.py', 'image_fingerprints.py', 'near_search.py',
    'candidate_policy.py', 'experimental_readiness.py', 'sample_batches.py',
)
SELECTION_REASONS = {
    "exact_duplicate", "duplicate_label_conflict", "confirmed_near_duplicate",
    "near_candidate_quarantine", "exact_crop_duplicate", "crop_label_conflict",
    "visual_review_rejection", "invalid_face_input",
}


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
