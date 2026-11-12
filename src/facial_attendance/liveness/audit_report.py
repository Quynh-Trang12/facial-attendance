"""Build and serialize audit evidence without performing image or split checks."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

from facial_attendance.liveness.dataset_files import file_sha256, read_json


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
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    audit: dict = field(default_factory=dict)
    extraction: dict | None = None
    archive_parts: int | None = None
    inventory_sha256: str | None = None
    batches: dict[str, BatchEvidence] | None = None
    known_media_check: str | None = None
    requirements: dict[str, str] | None = None
    accepted: bool = False
    result: str = "FAIL"
    error: str | None = None

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

    def complete(self) -> None:
        """Record completed checks while retaining unknown media relationships."""
        self.known_media_check = (
            "PASS for supplied common-media groups"
            if self.audit["known_common_media_groups"]
            else "UNVERIFIED: no known common-media group supplied"
        )
        self.requirements = {
            "labels_preprocessing_splits": "PASS",
            "known_common_media": self.known_media_check,
            "sample_batch_alignment_and_traceability": "PASS",
        }

        self.accepted = True
        self.result = "PASS"

    def fail(self, error: Exception) -> None:
        self.accepted = False
        self.result = "FAIL"
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
        }

        for name in (
            "extraction",
            "archive_parts",
            "inventory_sha256",
            "known_media_check",
            "requirements",
            "error",
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
        root = entrypoint.resolve().parents[1]
        report = self.to_dict()
        sources = (
            entrypoint.resolve(),
            *sorted(Path(__file__).parent.glob("*.py")),
            root / "src/facial_attendance/recognition/data_loader.py",
        )
        report["implementation_sha256"] = {
            str(path.relative_to(root)): file_sha256(path) for path in sources
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
        and all(
            requirements.get(key, "").startswith("PASS")
            for key in (
                "labels_preprocessing_splits",
                "known_common_media",
                "sample_batch_alignment_and_traceability",
            )
        )
    )

    with output_file.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
