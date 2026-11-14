"""Load selected RGB face images and targets from an audited inventory."""

from collections.abc import Callable
from pathlib import Path, PurePosixPath

from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import bbox_path, read_face
from facial_attendance.liveness.image_inventory import read_audited_inventory
from facial_attendance.liveness.data_contract import SPLITS


class LivenessImages:
    """Return (cropped RGB image/transformed tensor, 0=live/1=spoof).

    ``samples[index]`` traces each return to its inventory row. Tensor conversion
    is supplied by the existing recognition.image_transform(size) utility; the
    dataset itself stays usable in the Pillow-only source-check environment.
    ``audit_file`` accepts an experimental-readiness receipt or a fully accepted
    audit. Experimental readiness does not assert complete visual/provenance
    verification or model effectiveness.
    """

    def __init__(
        self,
        root: Path,
        inventory: Path,
        split: str,
        *,
        transform: Callable | None = None,
        audit_file: Path | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown liveness split: {split}")

        records, excluded = read_audited_inventory(
            root, inventory, audit_file or inventory.parent / "audit.json"
        )
        self.root, self.split, self.transform = root.resolve(), split, transform
        self.samples = tuple(
            row
            for row in records
            if row.split == split and not row.exclusion_reason and row.relative_path not in excluded
        )

    @classmethod
    def _for_audit(cls, root, records, split, *, transform=None, excluded=()):
        """Diagnostic batches inside the audit, before acceptance is possible."""
        if split not in SPLITS:
            raise ValueError(f"Unknown liveness split: {split}")
        dataset = cls.__new__(cls)
        dataset.root, dataset.split, dataset.transform = root.resolve(), split, transform
        dataset.samples = tuple(row for row in records if row.split == split
                                and not row.exclusion_reason and row.relative_path not in excluded)
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = local_path(self.root, sample.relative_path)
        box = local_path(self.root, str(bbox_path(PurePosixPath(sample.relative_path))))

        if file_sha256(path) != sample.sha256 or file_sha256(box) != sample.bbox_sha256:
            raise ValueError(f"Image/box changed since audit: {sample.relative_path}")

        image = read_face(path, box_file=box)
        return (
            self.transform(image) if self.transform is not None else image
        ), sample.binary_label
