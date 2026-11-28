"""Frozen exclusions govern the audit, including retained rows outside previews."""

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.datasets.data_contract import (
    LivenessSample,
    label_family,
)
from facial_attendance.liveness.datasets.image_inventory import write_inventory
from facial_attendance.liveness.datasets.reviewed_cache import row_order_sha256

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/audit/audit_liveness_selection.py"
)
spec = importlib.util.spec_from_file_location("reviewed_selection_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(project):
    root = project / "data/raw/liveness/celeba-spoof-full"
    directory = project / "evidence/view"
    directory.mkdir(parents=True)
    (project / "configs").mkdir()
    records, split_records, annotations, exclusions = [], {}, {}, []
    for split, filename in audit.SPLIT_ANNOTATIONS.items():
        subject = str(len(split_records) + 1)
        original = "test" if split == "test" else "train"
        rows, mapping = [], {}
        for index in range(5):
            source_type = (0, 1, 7, 7, 7)[index]
            binary = int(source_type > 0)
            name = f"Data/{original}/{subject}/{'spoof' if source_type else 'live'}/{index}.png"
            image = root / name
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224), (index * 50, int(subject) * 30, 100)).save(
                image
            )
            box = image.with_name(image.stem + "_BB.txt")
            box.write_text("0 0 224 224 0.99")
            row = LivenessSample(
                name,
                subject,
                split,
                source_type,
                binary,
                label_family(source_type, binary, name),
                file_sha256(image),
                file_sha256(box),
                original_split=original,
            )
            rows.append(row)
            mapping[name] = [0] * 40 + [source_type, binary, 0, binary]
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping))
        annotations[split] = path
        records.extend(rows)
        split_records[split] = rows
        row = rows[-1]
        exclusions.append(
            dict(
                relative_path=row.relative_path,
                split=split,
                binary_label=row.binary_label,
                source_type=row.source_type,
                decision="quarantine",
                source_sha256=row.sha256,
                bbox_sha256=row.bbox_sha256,
            )
        )
    inventory = root / "audits/current-schema-2026-11-12/inventory.csv"
    inventory.parent.mkdir(parents=True)
    write_inventory(inventory, records)
    quarantine = directory / "quarantine.csv"
    with quarantine.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exclusions[0]))
        writer.writeheader()
        writer.writerows(exclusions)
    (directory / "pending-review.csv").write_text("relative_path\n")
    view = dict(
        schema="liveness-reviewed-cache-view-v1",
        inventory_sha256=file_sha256(inventory),
        quarantine_sha256=file_sha256(quarantine),
        pending_review_sha256=file_sha256(directory / "pending-review.csv"),
        splits={},
    )
    for split, rows in split_records.items():
        path = directory / f"{split}-keep-indices.npy"
        np.save(path, np.array([0, 1, 2, 3], dtype=np.int64))
        view["splits"][split] = dict(
            annotations_sha256=file_sha256(annotations[split]),
            base_samples=5,
            base_row_order_sha256=row_order_sha256(rows),
            retained_samples=4,
            quarantined_samples=1,
            pending_review=0,
            keep_indices=path.name,
            keep_indices_sha256=file_sha256(path),
        )
    (directory / "cache-view.json").write_text(json.dumps(view))
    release = dict(
        files={}, crop_review=dict(cache_view="evidence/view/cache-view.json")
    )
    for path in (inventory, *annotations.values(), directory / "cache-view.json"):
        name = path.relative_to(project).as_posix()
        release["files"][name] = dict(delivery_path=name, sha256=file_sha256(path))
    (project / "configs/liveness-release.json").write_text(json.dumps(release))
    return root, directory, release, view, split_records


class ReviewedSelectionAuditTests(unittest.TestCase):
    def test_full_audit_excludes_unusable_inputs_and_preserves_frozen_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root, directory, _, _, records = fixture(project)
            for rows in records.values():
                (root / rows[-1].relative_path).write_bytes(b"unusable excluded image")
            before = {p: p.read_bytes() for p in directory.iterdir()}
            result = audit.run(project, project / "output", workers=2)
            self.assertTrue(result["completed"])
            for split, row in result["splits"].items():
                self.assertEqual(row["integrity_checked_rows"], 4)
                self.assertEqual(row["retained_samples"], 4)
                self.assertNotIn(
                    records[split][-1].relative_path,
                    {r["relative_path"] for r in row["examples"]},
                )
                self.assertTrue((project / "output" / f"{split}.png").is_file())
            self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir()})
            with self.assertRaises(FileExistsError):
                audit.run(project, project / "output")

    def test_equal_size_wrong_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            _, directory, release, view, _ = fixture(project)
            path = directory / "train-keep-indices.npy"
            np.save(path, np.array([0, 1, 2, 4], dtype=np.int64))
            view["splits"]["train"]["keep_indices_sha256"] = file_sha256(path)
            view_path = directory / "cache-view.json"
            view_path.write_text(json.dumps(view))
            release["files"]["evidence/view/cache-view.json"]["sha256"] = file_sha256(
                view_path
            )
            (project / "configs/liveness-release.json").write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "quarantine complement"):
                audit.run(project, project / "output", inspect=True)
            self.assertFalse(
                json.loads((project / "output/summary.json").read_text())["completed"]
            )

    def test_audit_checks_retained_images_outside_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root, _, _, _, records = fixture(project)
            (root / records["train"][3].relative_path).write_bytes(
                b"changed retained image"
            )
            result = audit.run(project, project / "inspect", inspect=True)
            self.assertEqual(result["splits"]["train"]["integrity_checked_rows"], 0)
            with self.assertRaisesRegex(ValueError, "changed since inventory"):
                audit.run(project, project / "full")

    def test_colab_stages_frozen_metadata_without_reselecting_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "shared"
            _, directory, release, _, _ = fixture(project)
            for path in directory.iterdir():
                name = path.relative_to(project).as_posix()
                release["files"][name] = dict(
                    delivery_path=name, sha256=file_sha256(path)
                )
            notebook = (
                SCRIPT.parents[2]
                / "notebooks/liveness-data-preparation-and-audit.ipynb"
            )
            cells = {
                c["id"]: "".join(c["source"])
                for c in json.loads(notebook.read_text())["cells"]
            }
            scope = dict(
                IS_COLAB=True,
                ACTION="audit",
                release=release,
                Path=Path,
                repo=Path(temporary) / "vm",
                DRIVE_PROJECT=project,
                hashlib=hashlib,
            )
            exec(
                compile(cells["drive-source-verification"], "verification", "exec"),
                scope,
            )
            exec(compile(cells["frozen-input-preflight"], "preflight", "exec"), scope)
            self.assertFalse(scope["repo"].exists())
            exec(compile(cells["frozen-selection-stage"], "staging", "exec"), scope)
            for name, record in release["files"].items():
                self.assertEqual(
                    (scope["repo"] / name).read_bytes(),
                    (project / record["delivery_path"]).read_bytes(),
                )
            scope["repo"] = Path(temporary) / "must-not-extract"
            annotation = (
                project / "data/raw/liveness/celeba-spoof-full/project_validation.json"
            )
            annotation.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                exec(
                    compile(cells["frozen-input-preflight"], "preflight", "exec"), scope
                )
            self.assertFalse(scope["repo"].exists())

    def test_changed_frozen_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root, _, _, _, _ = fixture(project)
            (root / audit.SPLIT_ANNOTATIONS["train"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "has changed"):
                audit.run(project, project / "output")


if __name__ == "__main__":
    unittest.main()
