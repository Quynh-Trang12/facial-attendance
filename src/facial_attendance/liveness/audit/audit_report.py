"""Build audit reports and traceable sample-batch previews with readback checks."""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageDraw

from facial_attendance.liveness.dataset_files import (
    file_sha256,
    local_path,
    read_json,
    source_root,
)
from facial_attendance.liveness.datasets.data_contract import TASK_REQUIREMENTS
from facial_attendance.liveness.datasets.face_crop import bbox_path, read_face
from facial_attendance.liveness.datasets.image_inventory import read_quality_report


@dataclass(frozen=True)
class Preprocessing:
    crop: str = "supplied _BB.txt, 224-coordinate system, clipped to image"
    color: str = "RGB"
    size: int = 224
    tensor: str = "float32 CHW /255, [0,1]"
    augmentation: str = "none"
    normalization: str = "no learned normalization; loader baseline"


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    device: str
    packages: dict[str, str]
    decode_workers: int

    @classmethod
    def capture(cls, workers: int) -> "RuntimeInfo":
        packages = {}

        for name in ("torch", "torchvision", "Pillow", "scikit-learn"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "not installed"

        return cls(sys.version, "CPU", packages, workers)


@dataclass(frozen=True)
class BatchEvidence:
    saved_batch_readback: str
    shape: list[int]
    image_dtype: str
    target_dtype: str
    range: list[float]
    alignment: str
    rows: list[dict]
    tensor_sha256: str
    preview_sha256: str


@dataclass
class AuditReport:
    data_root: Path
    preprocessing: Preprocessing
    runtime: RuntimeInfo
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    audit: dict = field(default_factory=dict)
    extraction: dict | None = None
    archive_parts: int | None = None
    inventory_sha256: str | None = None
    batches: dict[str, BatchEvidence] | None = None
    known_media_check: str | None = None
    requirements: dict[str, str] | None = None
    accepted: bool = False
    pipeline_completed: bool = False
    result: str = "FAIL"
    error: str | None = None
    quality_report: dict | None = None

    @classmethod
    def start(cls, root: Path, image_size: int, workers: int) -> "AuditReport":
        return cls(root, Preprocessing(size=image_size), RuntimeInfo.capture(workers))

    def record_extraction(self) -> None:
        path = self.data_root / "extraction.json"
        if not path.is_file():
            return

        extraction = read_json(path)
        fields = (
            "recorded_at",
            "archive_entries",
            "extracted_files",
            "uncompressed_payload_bytes",
            "crc_all_files",
            "selection",
        )
        if (
            not isinstance(extraction, dict)
            or not {*fields, "archive_parts"} <= extraction.keys()
            or not isinstance(extraction["archive_parts"], list)
        ):
            raise ValueError("Incomplete extraction record")

        self.extraction = {key: extraction[key] for key in fields}
        self.extraction.update(
            evidence_sha256=file_sha256(path), performed_in_this_run=False
        )
        self.archive_parts = len(extraction["archive_parts"])
        if "script_sha256" in extraction:
            self.extraction["script_sha256"] = extraction["script_sha256"]

    def complete(self, quality_file: Path | None = None) -> None:
        """Separate successful execution from evidence for task acceptance."""
        self.known_media_check = (
            "PASS for supplied common-media groups"
            if self.audit["known_common_media_groups"]
            else "UNVERIFIED: no known common-media group supplied"
        )
        self.requirements = {
            "labels_preprocessing_splits": "PASS",
            "known_common_media": self.known_media_check,
            "sample_batch_alignment_and_traceability": "PASS",
            "exact_deduplication": "UNVERIFIED: quality audit not supplied",
            "near_duplicate_review": "UNVERIFIED: quality audit not supplied",
            "source_relationship_coverage": "UNVERIFIED: provenance review required",
            "visual_labels_and_boxes": "UNVERIFIED: per-image review required",
            "loader_provenance": "UNVERIFIED: quality evidence not supplied",
            "face_input_integrity": "PASS",
            "selected_family_coverage": "PASS",
        }
        if quality_file is not None:
            quality = read_quality_report(
                self.data_root, quality_file, self.inventory_sha256
            )
            self.requirements.update(quality["requirements"])
            self.known_media_check = quality["requirements"]["known_common_media"]
            self.quality_report = {
                "path": str(quality_file.resolve()),
                "sha256": file_sha256(quality_file),
            }
            self.audit["cleaned_subset_counts"] = quality["cleaned_subset_counts"]
        self.pipeline_completed = True
        self.accepted = all(value == "PASS" for value in self.requirements.values())
        failed = any(value.startswith("FAIL") for value in self.requirements.values())
        self.result = "PASS" if self.accepted else "FAIL" if failed else "INCOMPLETE"

    def fail(self, error: Exception) -> None:
        self.accepted = False
        self.result = "FAIL"
        self.pipeline_completed = False
        self.error = str(error)

    def to_dict(self) -> dict:
        # Audit counters are already computed by the loader. Serialization must
        # not recalculate them or reinterpret excluded image failures.
        report = {
            "task": "LIV-03",
            "recorded_at": self.recorded_at,
            "python": self.runtime.python,
            "data_root": str(self.data_root),
            "preprocessing": asdict(self.preprocessing),
            "runtime": asdict(self.runtime),
            **self.audit,
            "accepted": self.accepted,
            "result": self.result,
            "pipeline_completed": self.pipeline_completed,
        }

        for name in (
            "extraction",
            "archive_parts",
            "inventory_sha256",
            "known_media_check",
            "requirements",
            "error",
            "quality_report",
        ):
            value = getattr(self, name)
            if value is not None:
                report[name] = value

        if self.batches is not None:
            report["batches"] = {
                split: asdict(batch) for split, batch in self.batches.items()
            }

        return report

    def save(self, output: Path, summary_output: Path, entrypoint: Path) -> None:
        script = entrypoint.resolve()
        root = (
            script.parents[2]
            if script.parent.parent.name == "scripts"
            else script.parents[1]
        )
        report = self.to_dict()
        sources = (
            entrypoint.resolve(),
            *sorted(source_root().rglob("*.py")),
            root / "src/facial_attendance/recognition/data_loader.py",
        )
        report["implementation_sha256"] = {
            path.relative_to(root).as_posix(): file_sha256(path) for path in sources
        }

        audit_file = output / "audit.json"
        with audit_file.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

        write_summary(audit_file, summary_output)


def write_summary(audit_file: Path, output_file: Path) -> None:
    """Export recorded results, keeping per-image records in local audit data."""
    report = read_json(audit_file)
    summary = dict(report)
    disagreements = summary.pop("folder_label_disagreements", [])
    summary["folder_label_disagreement_count"] = len(disagreements)
    summary["batches"] = {
        split: {
            **{key: value for key, value in batch.items() if key != "rows"},
            "targets": [row["binary_label"] for row in batch["rows"]],
        }
        for split, batch in report.get("batches", {}).items()
    }

    summary["audit_sha256"] = file_sha256(audit_file)
    summary["local_output"] = str(audit_file.parent.resolve())
    summary["summary_generator_sha256"] = file_sha256(Path(__file__))

    requirements = report.get("requirements", {})
    summary["all_task_requirements_verified"] = (
        report.get("accepted") is True
        and report.get("result") == "PASS"
        and set(TASK_REQUIREMENTS) <= set(requirements)
        and all(value == "PASS" for value in requirements.values())
    )

    with output_file.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def sample_batch(
    dataset, output: Path, batch_size: int, image_size: int
) -> BatchEvidence:
    # This boundary keeps CLI help and annotation failures usable without Torch,
    # matching the project's Pillow-only source-check job.
    import torch
    from torch.utils.data import DataLoader, Subset

    # Include each selected source type that exists, then fill in inventory order.
    indices, types = [], set()
    for index, row in enumerate(dataset.samples):
        if row.source_type not in types:
            indices.append(index)
            types.add(row.source_type)
    if batch_size < len(indices):
        raise ValueError(f"batch-size must cover {len(indices)} source types")
    indices += [i for i in range(len(dataset)) if i not in indices][
        : batch_size - len(indices)
    ]

    images, targets = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
        )
    )

    if (
        images.shape != (len(indices), 3, image_size, image_size)
        or images.dtype != torch.float32
        or targets.dtype != torch.int64
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError(f"Invalid batch in {dataset.split}")

    records = []
    sheet = Image.new("RGB", (image_size * len(indices), image_size + 64), "white")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        row = dataset.samples[index]
        face = read_face(
            local_path(dataset.root, row.relative_path),
            box_file=local_path(dataset.root, str(bbox_path(Path(row.relative_path)))),
        )
        expected = dataset.transform(face)
        if int(targets[position]) != row.binary_label or not torch.equal(
            images[position], expected
        ):
            raise ValueError(f"Image/target alignment failed: {row.relative_path}")

        sheet.paste(
            face.resize((image_size, image_size), Image.Resampling.BILINEAR),
            (position * image_size, 0),
        )
        draw.text(
            (position * image_size + 3, image_size + 3),
            f"index={index} target={row.binary_label}\n{row.split} type={row.source_type}\n{row.attack_family}",
            fill="black",
        )
        records.append({"dataset_index": index, **asdict(row)})

    stem = output / dataset.split
    sheet.save(stem.with_suffix(".png"))
    torch.save(
        {
            "images": images,
            "targets": targets,
            "relative_paths": [r["relative_path"] for r in records],
        },
        stem.with_suffix(".pt"),
    )

    saved = torch.load(stem.with_suffix(".pt"), weights_only=True)
    if (
        not torch.equal(saved["images"], images)
        or not torch.equal(saved["targets"], targets)
        or saved["relative_paths"] != [r["relative_path"] for r in records]
    ):
        raise ValueError(f"Saved batch readback failed: {dataset.split}")

    return BatchEvidence(
        saved_batch_readback="PASS",
        shape=list(images.shape),
        image_dtype=str(images.dtype),
        target_dtype=str(targets.dtype),
        range=[images.min().item(), images.max().item()],
        alignment="PASS",
        rows=records,
        tensor_sha256=file_sha256(stem.with_suffix(".pt")),
        preview_sha256=file_sha256(stem.with_suffix(".png")),
    )
