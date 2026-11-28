"""Inspect public CelebA-Spoof metadata against the published AENet protocol.

This reports differences rather than relabelling a project-derived partition as
an official validation split. Source annotations and existing audit evidence are
read-only. Recipe facts distinguish the paper from its released inference code.
"""

import json
from collections import Counter
from pathlib import Path, PurePosixPath

from facial_attendance.liveness.dataset_files import file_sha256, local_path

PAPER_URL = "https://arxiv.org/abs/2007.12342"
AUTHOR_INFERENCE_URL = (
    "https://github.com/ZhangYuanhan-AI/CelebA-Spoof/tree/master/intra_dataset_code"
)
PUBLISHED_RECIPE = {
    "paper": PAPER_URL,
    "backbone": "ImageNet-initialized ResNet-18",
    "input_size": [224, 224],
    "detector": "RetinaFace; public release supplies bounding boxes",
    "optimizer": "SGD",
    "learning_rate": 0.005,
    "epochs": 50,
    "augmentation": "color distortion; strength/distribution unspecified in the paper",
    "semantic_heads": {"face_attributes": 40, "spoof_type": 11, "illumination": 5},
    "loss_weights": {
        "face_attributes": 1.0,
        "spoof_type": 0.1,
        "illumination": 0.01,
        "depth_map": 0.1,
        "reflection_map": 0.1,
    },
    "geometry": {
        "depth_teacher": "PRNet",
        "reflection_teacher": "Zhang et al., CVPR 2018",
        "resolution": [14, 14],
        "spoof_depth": "zero",
        "live_reflection": "zero",
    },
    "released_inference": {
        "source": AUTHOR_INFERENCE_URL,
        "RGB": True,
        "transform": "Resize(224,224), ToTensor; no mean/std normalization",
    },
    "unspecified_training_details": [
        "batch size",
        "SGD momentum",
        "weight decay",
        "learning-rate schedule",
        "random seed",
        "exact validation split assignment",
        "training normalization",
        "pseudo-label rasterization/normalization details",
    ],
}


def annotation_summary(path, root):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Source annotations must be a nonempty path-to-label mapping")
    subjects, types, binary = set(), Counter(), Counter()
    path_label_conflicts = []
    for name, attributes in raw.items():
        parts = PurePosixPath(name).parts
        local_path(root, name)
        if len(parts) != 5 or parts[0] != "Data" or parts[1] not in ("train", "test"):
            raise ValueError(f"Unexpected source image path: {name}")
        if (
            not isinstance(attributes, list)
            or len(attributes) != 44
            or any(type(value) is not int for value in attributes)
            or any(value not in (0, 1) for value in attributes[:40])
            or attributes[40] not in range(11)
            or attributes[41] not in range(5)
            or attributes[43] != int(attributes[40] != 0)
            or (attributes[41] == 0) != (attributes[43] == 0)
        ):
            raise ValueError(f"Invalid source attribute schema: {name}")
        if attributes[43] and any(attributes[:40]):
            raise ValueError(f"Spoof face attributes must be zero: {name}")
        if parts[3] != ("live" if attributes[43] == 0 else "spoof"):
            # Original public annotations contain known path/label conflicts.
            # Report source facts; do not silently change labels from filenames.
            path_label_conflicts.append(name)
        subjects.add(parts[2])
        types[attributes[40]] += 1
        binary[attributes[43]] += 1
    return (
        raw,
        subjects,
        {
            "annotations": len(raw),
            "subjects": len(subjects),
            "source_type_counts": dict(sorted(types.items())),
            "binary_label_counts": dict(sorted(binary.items())),
            "annotation_sha256": file_sha256(Path(path)),
            "path_label_conflicts": path_label_conflicts,
        },
    )


def inspect_public_protocol(root):
    root = Path(root).resolve()
    annotations, subjects, summaries = {}, {}, {}
    for split in ("train", "test"):
        path = root / f"metas/intra_test/{split}_label.json"
        annotations[split], subjects[split], summaries[split] = annotation_summary(
            path, root
        )
    public_validation = sorted(
        str(path.relative_to(root))
        for path in (root / "metas").rglob("*")
        if path.is_file() and ("val" in path.stem.lower())
    )
    total = sum(len(rows) for rows in annotations.values())
    return {
        "data_root": str(root),
        "public_metadata": summaries,
        "public_train_test_subject_overlap": sorted(
            subjects["train"] & subjects["test"]
        ),
        "public_train_test_path_overlap": sorted(
            annotations["train"].keys() & annotations["test"].keys()
        ),
        "public_validation_metadata_files": public_validation,
        "public_annotated_images": total,
        "paper_reported_images": 625537,
        "difference_from_paper_reported_count": 625537 - total,
        "validation_attribution": "Project-derived; exact paper split assignment unavailable in inspected public metadata"
        if not public_validation
        else "Candidate files require split assignment verification",
        "recipe": PUBLISHED_RECIPE,
        "scope": "Metadata inspection, not pixel decoding, teacher verification or training readiness certification",
    }
