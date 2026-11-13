"""Controlled fixtures prove loader contracts; they are not CelebA-Spoof data."""

import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from facial_attendance.liveness.dataset_audit import audit_dataset, audit_splits
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import read_face
from facial_attendance.liveness.image_inventory import (
    build_inventory,
    read_inventory,
    write_inventory,
)
from facial_attendance.liveness.data_contract import EXCLUDED, FAMILIES, SOURCE, SPLITS
from facial_attendance.liveness.audit_report import AuditReport
from facial_attendance.liveness.split_rules import read_subject_splits


class LivenessDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="liveness-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.annotations = {}
        for split_index, split in enumerate(SPLITS):
            labels = {}
            for code in sorted(set(FAMILIES) | EXCLUDED):
                kind = "live" if code == 0 else "spoof"
                name = f"Data/{split}/{split_index + 100}/{kind}/{code}.png"
                image = self.root / name
                image.parent.mkdir(parents=True, exist_ok=True)
                # Distinct pixels per split/type; RGB channel order is observable.
                Image.new(
                    "RGB", (16, 12), (30 + split_index * 40 + code, 20, 200)
                ).save(image)
                image.with_name(image.stem + "_BB.txt").write_text("0 0 224 224 0.99\n")
                attributes = [0] * 44
                attributes[40], attributes[43] = code, int(code != 0)
                labels[name] = attributes
            self.annotations[split] = labels
            (self.root / f"{split}.json").write_text(json.dumps(labels))
        self.data_description = self.root / "data-description.json"
        self.description = {
            "source_url": SOURCE,
            "dataset_archive": "SYNTHETIC FIXTURE, not acquired data",
            "received_on": "2026-11-11",
            "split_evidence": "Generated disjoint fixture subjects",
            "selection": "All 11 synthetic source codes in each split",
            "split_annotations": {split: f"{split}.json" for split in SPLITS},
        }
        self.save_data_description()

    def save_data_description(self):
        self.data_description.write_text(json.dumps(self.description), encoding="utf-8")

    def save_labels(self, split):
        (self.root / f"{split}.json").write_text(json.dumps(self.annotations[split]))

    def inventory(self):
        return audit_dataset(self.root, self.data_description)

    def test_mapping_exclusions_and_fixed_splits(self):
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(len(records), 33)
        for split in SPLITS:
            self.assertEqual(report["counts"][split]["selected"], 7)
            self.assertEqual(report["counts"][split]["excluded"], 4)
            self.assertEqual(
                report["counts"][split]["families"],
                {"live": 1, "print": 3, "screen": 3},
            )
        self.assertEqual(report["source_media_unknown"], 33)
        path = self.root / "inventory.csv"
        write_inventory(path, records)
        self.assertEqual(read_inventory(path), records)
        for split in SPLITS:
            dataset = LivenessImages(self.root, path, split)
            for index, row in enumerate(dataset.samples):
                image, target = dataset[index]
                self.assertEqual(target, int(row.source_type != 0))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(row.split, split)
                self.assertEqual(image.getpixel((0, 0))[1:], (20, 200))

    def test_missing_project_validation_is_not_invented(self):
        del self.description["split_annotations"]["validation"]
        self.save_data_description()
        with self.assertRaisesRegex(ValueError, "Project train, validation and test"):
            self.inventory()

    def test_inventory_construction_does_not_decode_images(self):
        name = next(iter(self.annotations["train"]))
        (self.root / name).write_bytes(b"not an image")
        inventory = build_inventory(self.root, self.data_description)
        self.assertEqual(len(inventory.records), 33)
        self.assertTrue(all(not row.sha256 for row in inventory.records))
        _, report = self.inventory()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["blocking_image_errors"][0]["relative_path"], name)

    def test_loader_import_does_not_load_audit_or_reporting(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from facial_attendance.liveness.data_loader import LivenessImages; "
                "assert 'facial_attendance.liveness.dataset_audit' not in sys.modules; "
                "assert 'facial_attendance.liveness.audit_report' not in sys.modules",
            ],
            check=True,
        )

    def test_cpu_workers_preserve_inventory_order_and_hashes(self):
        sequential, _ = self.inventory()
        parallel, report = audit_dataset(self.root, self.data_description, workers=4)
        self.assertTrue(report["accepted"])
        self.assertEqual(parallel, sequential)

    def test_validation_subject_file_cannot_take_test_subjects(self):
        for split, ids in (
            ("train", ("100", "101", "shared")),
            ("test", ("102", "shared")),
        ):
            labels = {f"Data/{split}/{subject}/live/a.png": [0] * 44 for subject in ids}
            (self.root / f"original-{split}.json").write_text(json.dumps(labels))
        policy = {
            "source_annotations": {s: f"original-{s}.json" for s in ("train", "test")},
            "quarantined_train_subjects": ["shared"],
            "validation_subject_file": "subjects.json",
        }
        assignment = self.root / "subjects.json"
        assignment.write_text(json.dumps({"validation_subjects": ["101"]}))
        assignments = read_subject_splits(self.root, policy)
        self.assertEqual(assignments.quarantined_train_subjects, {"shared"})
        self.assertEqual(assignments.validation_subjects, {"101"})
        for invalid in (["102"], ["shared"], ["100", "101"], ["101", "101"]):
            assignment.write_text(json.dumps({"validation_subjects": invalid}))
            with self.subTest(subjects=invalid), self.assertRaises(ValueError):
                read_subject_splits(self.root, policy)

    def test_derived_validation_keeps_original_annotations_and_enforces_assignment(
        self,
    ):
        validation = self.annotations["validation"]
        rewritten = {
            name.replace("Data/validation/", "Data/train/"): attrs
            for name, attrs in validation.items()
        }
        shutil.copytree(self.root / "Data/validation/101", self.root / "Data/train/101")
        self.annotations["validation"] = rewritten
        self.save_labels("validation")
        (self.root / "original-train.json").write_text(
            json.dumps({**self.annotations["train"], **rewritten})
        )
        (self.root / "original-test.json").write_text(
            json.dumps(self.annotations["test"])
        )
        subjects = self.root / "subjects.json"
        subjects.write_text(json.dumps({"validation_subjects": ["101"]}))
        self.description["validation_from_train"] = {
            "method": "Explicit fixture validation subject selection",
            "validation_subject_file": "subjects.json",
            "source_annotations": {s: f"original-{s}.json" for s in ("train", "test")},
            "quarantined_train_subjects": [],
        }
        self.save_data_description()
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(
            {row.original_split for row in records if row.split == "validation"},
            {"train"},
        )
        # Full-dataset delivery must not silently omit an inconvenient row.
        self.description["selection_scope"] = "full_dataset"
        self.save_data_description()
        missing_name, missing_attrs = self.annotations["train"].popitem()
        self.save_labels("train")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.inventory()
        self.annotations["train"][missing_name] = missing_attrs
        self.save_labels("train")
        subjects.write_text(json.dumps({"validation_subjects": ["100"]}))
        with self.assertRaisesRegex(ValueError, "subject split policy violated"):
            self.inventory()

    def test_full_inventory_retains_subject_quarantine_but_loader_excludes_it(self):
        validation = self.annotations["validation"]
        rewritten = {
            n.replace("Data/validation/", "Data/train/"): a
            for n, a in validation.items()
        }
        shutil.copytree(self.root / "Data/validation/101", self.root / "Data/train/101")
        self.annotations["validation"] = rewritten
        self.save_labels("validation")
        name = "Data/train/102/live/quarantined.png"
        image = self.root / name
        image.parent.mkdir(parents=True)
        Image.new("RGB", (16, 12), (173, 70, 10)).save(image)
        image.with_name("quarantined_BB.txt").write_text("0 0 224 224 1")
        quarantined = {name: [0] * 44}
        duplicate_name = "Data/train/103/live/duplicate.png"
        duplicate = self.root / duplicate_name
        duplicate.parent.mkdir(parents=True)
        val_live = next(n for n, a in rewritten.items() if a[40] == 0)
        shutil.copyfile(self.root / val_live, duplicate)
        duplicate.with_name("duplicate_BB.txt").write_text("0 0 224 224 1")
        quarantined[duplicate_name] = [0] * 44
        (self.root / "quarantine.json").write_text(json.dumps(quarantined))
        (self.root / "original-train.json").write_text(
            json.dumps({**self.annotations["train"], **rewritten, **quarantined})
        )
        (self.root / "original-test.json").write_text(
            json.dumps(self.annotations["test"])
        )
        (self.root / "subjects.json").write_text(
            json.dumps({"validation_subjects": ["101"]})
        )
        self.description.update(
            {
                "selection_scope": "full_dataset",
                "quarantine_annotations": "quarantine.json",
                "validation_from_train": {
                    "method": "Fixture validation subjects",
                    "validation_subject_file": "subjects.json",
                    "source_annotations": {
                        s: f"original-{s}.json" for s in ("train", "test")
                    },
                    "quarantined_train_subjects": ["102"],
                    "duplicate_train_subjects": ["103"],
                },
            }
        )
        self.save_data_description()
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["inventory_rows"], 35)
        self.assertEqual(report["quarantined_rows"], 2)
        self.assertEqual(report["overlaps"], [])
        inventory = self.root / "inventory.csv"
        write_inventory(inventory, records)
        self.assertEqual(read_inventory(inventory), records)
        self.assertNotIn(
            name,
            {
                s.relative_path
                for s in LivenessImages(self.root, inventory, "train").samples
            },
        )
        self.assertEqual(
            next(
                r.exclusion_reason for r in records if r.relative_path == duplicate_name
            ),
            "training_subject_has_duplicate_in_evaluation",
        )
        Image.new("RGB", (16, 12), (173, 80, 11)).save(duplicate)
        with self.assertRaisesRegex(
            ValueError, "No identical-image evidence for quarantined training subjects"
        ):
            self.inventory()

    def test_rich_json_labels_override_misleading_folder_name(self):
        name = next(n for n, a in self.annotations["train"].items() if a[40] == 0)
        self.annotations["train"][name][40] = 7
        self.annotations["train"][name][43] = 1
        # Keep another legitimate live example so each family remains represented.
        new_name = name.replace("0.png", "new-live.png")
        original = self.root / name
        Image.new("RGB", (12, 12), (100, 50, 10)).save(self.root / new_name)
        (self.root / new_name).with_name("new-live_BB.txt").write_text("0 0 224 224 1")
        self.annotations["train"][new_name] = [0] * 44
        self.save_labels("train")
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["folder_label_disagreements"], [name])
        self.assertEqual(
            next(row for row in records if row.relative_path == name).binary_label, 1
        )

    def test_multipart_reader_crosses_boundaries_and_seeks(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/prepare_liveness_data.py"
        )
        spec = importlib.util.spec_from_file_location("prepare_liveness", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = [self.root / f"part{i}" for i in range(3)]
        for path, data in zip(paths, (b"abc", b"defg", b"hi")):
            path.write_bytes(data)
        with module.ArchiveParts(paths) as reader:
            self.assertEqual(reader.read(5), b"abcde")
            self.assertEqual(reader.seek(-3, 2), 6)
            self.assertEqual(reader.read(), b"ghi")
            self.assertEqual(reader.read(1), b"")
            reader.seek(2)
            self.assertEqual(reader.read(5), b"cdefg")
            with self.assertRaises(ValueError):
                reader.seek(-1)

    def test_full_dataset_preparation_reuses_fixed_subjects_with_empty_quarantine(
        self,
    ):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/prepare_liveness_data.py"
        )
        spec = importlib.util.spec_from_file_location("prepare_full_liveness", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = self.root / "metas/intra_test"
        metadata.mkdir(parents=True)
        validation = {
            n.replace("Data/validation/", "Data/train/"): a
            for n, a in self.annotations["validation"].items()
        }
        shutil.copytree(self.root / "Data/validation/101", self.root / "Data/train/101")
        (metadata / "train_label.json").write_text(
            json.dumps({**self.annotations["train"], **validation})
        )
        (metadata / "test_label.json").write_text(json.dumps(self.annotations["test"]))
        validation_subject_file = self.root / "validation-subjects.json"
        validation_subject_file.write_text(
            json.dumps(
                {
                    "method": "Fixed fixture split assignment",
                    "validation_subjects": ["101"],
                    "quarantined_train_subjects": [],
                }
            )
        )
        before = validation_subject_file.read_bytes()
        prepared = self.root / "prepared"
        prepared.mkdir()
        # Move only this fixture's earlier data description out of the target
        # so the preparation entry point can enforce no-overwrite behavior.
        self.data_description.rename(prepared / "earlier-data-description.json")
        module.configure_dataset(self.root, validation_subject_file)
        self.assertEqual(validation_subject_file.read_bytes(), before)
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["inventory_rows"], 33)
        self.assertEqual(report["quarantined_rows"], 0)
        assignment = json.loads(before)
        assignment["validation_subjects"] = ["101", "101"]
        validation_subject_file.write_text(json.dumps(assignment))
        with self.assertRaisesRegex(ValueError, "unique validation subject IDs"):
            module.configure_dataset(self.root, validation_subject_file)

    def test_source_type_and_binary_disagreement_or_unknown_rejected(self):
        name = next(iter(self.annotations["train"]))
        for code, binary in [(0, 1), (99, 1), (True, 1)]:
            with self.subTest(code=code):
                self.annotations["train"][name][40] = code
                self.annotations["train"][name][43] = binary
                self.save_labels("train")
                with self.assertRaises(ValueError):
                    self.inventory()

    def test_duplicate_json_paths_are_rejected(self):
        (self.root / "train.json").write_text('{"x": [], "x": []}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.inventory()

    def test_known_video_frames_leak_even_when_subjects_and_hashes_differ(self):
        names = [next(iter(self.annotations[s])) for s in ("train", "test")]
        relationships = self.root / "relationships.csv"
        with relationships.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["relative_path", "source_media_id", "session_id"])
            writer.writerows((name, "documented-video-17", "") for name in names)
        self.description["relationships"] = relationships.name
        self.save_data_description()
        _, report = self.inventory()
        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["overlaps"],
            [
                {
                    "field": "source_media_id",
                    "value": "documented-video-17",
                    "splits": ["test", "train"],
                }
            ],
        )
        # Valid counterpart: separate frames from that video stay in training.
        train_names = list(self.annotations["train"])[:2]
        with relationships.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["relative_path", "source_media_id", "session_id"])
            writer.writerows(
                (name, "documented-video-17", "session-17") for name in train_names
            )
        _, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["source_media_known"], 2)

    def test_each_overlap_key_is_checked_before_filtering(self):
        records, _ = self.inventory()
        train = next(
            row for row in records if row.split == "train" and row.source_type == 10
        )
        test = next(
            row for row in records if row.split == "test" and row.source_type == 10
        )
        for field in ("subject_id", "sha256", "source_media_id", "session_id"):
            with self.subTest(field=field):
                left, right = (
                    replace(train, **{field: "shared"}),
                    replace(test, **{field: "shared"}),
                )
                self.assertEqual(
                    audit_splits([left, right])["overlaps"][0]["field"], field
                )

    def test_renamed_identical_files_across_splits_rejected(self):
        source = self.root / next(iter(self.annotations["train"]))
        target = self.root / next(iter(self.annotations["test"]))
        shutil.copyfile(source, target)
        _, report = self.inventory()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["overlaps"][0]["field"], "sha256")

    def test_corrupt_image_and_box_are_reported_without_silent_removal(self):
        name = next(iter(self.annotations["train"]))
        (self.root / name).write_bytes(b"not an image")
        _, report = self.inventory()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["image_errors"][0]["relative_path"], name)
        self.assertEqual(report["counts"]["train"]["selected"], 7)

    def test_changed_image_or_box_after_audit_rejected(self):
        records, _ = self.inventory()
        inventory = self.root / "inventory.csv"
        write_inventory(inventory, records)
        dataset = LivenessImages(self.root, inventory, "train")
        image = self.root / dataset.samples[0].relative_path
        image.with_name(image.stem + "_BB.txt").write_text("0 0 20 20 1")
        with self.assertRaisesRegex(ValueError, "changed since audit"):
            dataset[0]

    def test_empty_box_for_preexcluded_mask_preserves_pixel_integrity_audit(self):
        name = next(n for n, a in self.annotations["train"].items() if a[40] == 4)
        image = self.root / name
        image.with_name(image.stem + "_BB.txt").write_text("")
        records, report = self.inventory()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["blocking_image_errors"], [])
        self.assertEqual(report["excluded_image_errors"][0]["relative_path"], name)
        self.assertEqual(report["excluded_image_errors"][0]["raw_image_decode"], "PASS")
        self.assertEqual(
            report["image_integrity"],
            {
                "raw_images_decoded": 33,
                "valid_face_inputs": 32,
                "invalid_face_inputs": 1,
            },
        )
        inventory = self.root / "inventory.csv"
        write_inventory(inventory, records)
        self.assertEqual(len(LivenessImages(self.root, inventory, "train")), 7)

    def test_box_geometry_rgb_and_grayscale(self):
        image = self.root / "crop.png"
        Image.new("L", (224, 224), 80).save(image)
        box = image.with_name("crop_BB.txt")
        box.write_text("-10 20 100 80 1")
        face = read_face(image)
        self.assertEqual(face.size, (90, 80))
        self.assertEqual(face.getpixel((0, 0)), (80, 80, 80))
        for value in (
            "0 0 0 10 1",
            "nan 0 100 100 1",
            "300 0 10 10 1",
            "0 0 10 10 0",
            "0 0 10",
        ):
            box.write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_face(image)

    def test_paths_reject_traversal_and_external_symlink_accept_internal_alias(self):
        path = self.root / "inside.txt"
        path.write_text("okay")
        self.assertEqual(local_path(self.root, "inside.txt"), path)
        for value in (
            "../outside",
            "folder\\..\\inside.txt",
            "/etc/passwd",
            "C:\\data\\face.png",
        ):
            with self.assertRaises(ValueError):
                local_path(self.root, value)
        try:
            (self.root / "alias").symlink_to(path)
            (self.root / "escape").symlink_to(Path(__file__).resolve())
        except OSError:
            self.skipTest("Symlink creation unavailable")
        self.assertEqual(local_path(self.root, "alias"), path)
        with self.assertRaises(ValueError):
            local_path(self.root, "escape")

    def test_report_failure_cannot_inherit_accepted_inventory(self):
        report = AuditReport.start(self.root, image_size=16, workers=1)
        report.audit = {"accepted": True, "counts": {"train": {"selected": 7}}}
        report.fail(ValueError("Batch alignment failed"))
        saved = report.to_dict()
        self.assertFalse(saved["accepted"])
        self.assertEqual(saved["result"], "FAIL")
        self.assertEqual(saved["error"], "Batch alignment failed")
        self.assertEqual(saved["counts"], report.audit["counts"])
        self.assertTrue(report.audit["accepted"])

    def test_audit_output_directories_separate_routes_and_preserve_overrides(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_liveness_data.py"
        spec = importlib.util.spec_from_file_location("liveness_output_cli", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli.ROOT = self.root
        for route in ("local", "colab"):
            args = cli.parse_arguments(
                ["--data-description", "source.json", "--execution-route", route]
            )
            output = cli.audit_output_directory(args)
            self.assertEqual(output.parent, self.root / "evidence/runs/LIV-03" / route)
            self.assertRegex(output.name, r"^\d{8}T\d{12}Z$")
            self.assertFalse(output.exists())
        args = cli.parse_arguments(["--data-description", "source.json"])
        self.assertEqual(args.execution_route, "local")
        custom = self.root / "external output" / "initial"
        args = cli.parse_arguments(
            [
                "--data-description",
                "source.json",
                "--execution-route",
                "colab",
                "--output-dir",
                str(custom),
            ]
        )
        self.assertEqual(cli.audit_output_directory(args), custom.resolve())
        self.assertFalse(custom.exists())
        with self.assertRaises(SystemExit):
            cli.parse_arguments(
                ["--data-description", "source.json", "--execution-route", "other"]
            )

    def test_real_cli_tensor_alignment_and_saved_inventory(self):
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("torchvision") is None
        ):
            self.skipTest("Tensor integration requires the project ML environment")
        script = Path(__file__).resolve().parents[1] / "scripts/check_liveness_data.py"
        spec = importlib.util.spec_from_file_location("liveness_cli", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        extraction = {
            "recorded_at": "2026-01-01T00:00:00Z",
            "archive_entries": 1,
            "extracted_files": 1,
            "uncompressed_payload_bytes": 1,
            "crc_all_files": "FIXTURE",
            "selection": "Synthetic extraction metadata",
            "archive_parts": [{"name": "fixture.zip.001"}],
        }
        (self.root / "extraction.json").write_text(json.dumps(extraction))
        output = self.root / "audit"
        self.assertEqual(
            cli.main(
                [
                    "--data-root",
                    str(self.root),
                    "--data-description",
                    str(self.data_description),
                    "--output-dir",
                    str(output),
                    "--image-size",
                    "16",
                ]
            ),
            0,
        )
        report = json.loads((output / "audit.json").read_text())
        self.assertTrue(report["accepted"])
        self.assertIn("UNVERIFIED", report["known_media_check"])
        summary_file = output / "summary.json"
        summary = json.loads(summary_file.read_text())
        self.assertFalse(summary["all_task_requirements_verified"])
        self.assertEqual(summary["requirements"], report["requirements"])
        reporting_source = "src/facial_attendance/liveness/audit_report.py"
        self.assertEqual(
            report["implementation_sha256"][reporting_source],
            file_sha256(script.parents[1] / reporting_source),
        )
        self.assertEqual(
            summary["summary_generator_sha256"],
            report["implementation_sha256"][reporting_source],
        )
        self.assertEqual(summary["runtime"], report["runtime"])
        self.assertEqual(summary["runtime"]["device"], "CPU")
        self.assertEqual(summary["extraction"], report["extraction"])
        for key, value in extraction.items():
            if key != "archive_parts":
                self.assertEqual(summary["extraction"][key], value)
        self.assertEqual(summary["archive_parts"], 1)
        self.assertFalse(summary["extraction"]["performed_in_this_run"])
        for key in (
            "counts",
            "overlaps",
            "annotations",
            "excluded_image_errors",
            "missing_families",
            "source_media_unknown",
            "limitations",
        ):
            self.assertEqual(summary[key], report[key])
        self.assertEqual(
            summary["folder_label_disagreement_count"],
            len(report["folder_label_disagreements"]),
        )
        exported = self.root / "exported-summary.json"
        self.assertEqual(
            cli.main(
                [
                    "--summarize-audit",
                    str(output / "audit.json"),
                    "--summary-output",
                    str(exported),
                ]
            ),
            0,
        )
        self.assertEqual(exported.read_bytes(), summary_file.read_bytes())
        with self.assertRaises(FileExistsError):
            cli.main(
                [
                    "--summarize-audit",
                    str(output / "audit.json"),
                    "--summary-output",
                    str(exported),
                ]
            )
        import torch

        for split in SPLITS:
            saved = torch.load(output / f"{split}.pt", weights_only=True)
            self.assertEqual(list(saved["images"].shape), [7, 3, 16, 16])
            self.assertEqual(summary["batches"][split]["saved_batch_readback"], "PASS")
            rows = report["batches"][split]["rows"]
            self.assertEqual(
                saved["targets"].tolist(), [row["binary_label"] for row in rows]
            )
            self.assertEqual(
                saved["relative_paths"], [row["relative_path"] for row in rows]
            )
            self.assertAlmostEqual(
                saved["images"][0, 0, 0, 0].item(),
                (30 + SPLITS.index(split) * 40) / 255,
                places=6,
            )

        # A failed audit must also export its failure, never a completion claim.
        del self.description["split_annotations"]["validation"]
        self.save_data_description()
        failed_output = self.root / "failed-audit"
        self.assertEqual(
            cli.main(
                [
                    "--data-root",
                    str(self.root),
                    "--data-description",
                    str(self.data_description),
                    "--output-dir",
                    str(failed_output),
                ]
            ),
            1,
        )
        failed_summary = json.loads((failed_output / "summary.json").read_text())
        self.assertEqual(failed_summary["result"], "FAIL")
        self.assertFalse(failed_summary["accepted"])
        self.assertFalse(failed_summary["all_task_requirements_verified"])


if __name__ == "__main__":
    unittest.main()
