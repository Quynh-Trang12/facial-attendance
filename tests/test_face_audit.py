"""Generated fixtures with planted defects; each defect must surface in the audit."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from facial_attendance.recognition.audit import audit_dataset
from facial_attendance.recognition.data_loader import FaceLayout


class AuditFixture(unittest.TestCase):
    """A clean dataset: distinct images, valid identities and two unique trials."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="face-audit-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.shade = 0
        # Identity "c" has a single training image, so singleton counting is exercised.
        for split, counts in {
            "train_data": {"a": 3, "b": 2, "c": 1},
            "val_data": {"a": 1, "b": 1},
            "test_data": {"a": 1, "c": 1},
        }.items():
            for identity, count in counts.items():
                for index in range(count):
                    self.image(f"classification_data/{split}/{identity}/{index}.png")
        for index in range(4):
            self.image(f"verification_data/v{index}.png", size=(6, 5), mode="L")
        self.pairs_file = self.root / "verification_pairs_val.txt"
        self.pairs_file.write_text(
            "verification_data/v0.png verification_data/v1.png 1\n"
            "verification_data/v2.png verification_data/v3.png 0\n"
        )
        self.layout = FaceLayout.discover(self.root)

    def image(self, relative: str, *, size=(8, 8), mode="RGB"):
        # A distinct colour per file keeps the clean fixture free of accidental duplicates.
        self.shade += 1
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        color = self.shade if mode == "L" else (self.shade, 255 - self.shade, 7)
        Image.new(mode, size, color=color).save(path)

    def copy(self, source: str, target: str):
        (self.root / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / source, self.root / target)

    def add_trials(self, text: str):
        with self.pairs_file.open("a") as handle:
            handle.write(text)


class FaceAuditTests(AuditFixture):
    def test_clean_fixture_passes_with_identity_and_image_statistics(self):
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual((report["failures"], report["warnings"]), ([], []))
        self.assertEqual(report["files_audited"], 14)
        train = report["partitions"]["train_data"]
        self.assertEqual(train["identities"], 3)
        self.assertEqual(
            train["images_per_identity"],
            {"min": 1, "median": 2, "max": 3, "singletons": 1},
        )
        self.assertTrue(report["partitions"]["val_data"]["subset_of_train"])
        self.assertEqual(report["partitions"]["val_data"]["train_identities_absent"], 1)
        self.assertEqual(report["images"]["sizes"]["verification_data"], {"6x5": 4})
        self.assertEqual(report["images"]["modes"]["train_data"], {"RGB": 6})
        self.assertEqual(report["trials"]["label_counts"], {"0": 1, "1": 1})
        self.assertEqual(report["trials"]["distinct_images_used"], 4)

    def test_train_copy_in_validation_is_leakage(self):
        self.copy(
            "classification_data/train_data/a/0.png",
            "classification_data/val_data/a/9.png",
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "FAIL")
        leak = report["duplicates"]["across_partitions"]["train_data/val_data"]
        self.assertEqual(leak["shared_images"], 1)
        self.assertIn(
            "classification_data/val_data/a/9.png", leak["examples"][0]["paths"]
        )

    def test_classification_copy_in_verification_is_leakage(self):
        self.copy(
            "classification_data/test_data/c/0.png", "verification_data/extra.png"
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "FAIL")
        parts = report["duplicates"]["across_partitions"]
        self.assertEqual(parts["test_data/verification_data"]["shared_images"], 1)
        self.assertEqual(parts["train_data/verification_data"]["shared_images"], 0)

    def test_image_under_two_identities_is_a_label_warning(self):
        self.copy(
            "classification_data/train_data/a/0.png",
            "classification_data/train_data/b/9.png",
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["duplicates"]["across_identities"]["groups"], 1)
        self.assertTrue(any("more than one identity" in w for w in report["warnings"]))

    def test_duplicate_inside_one_identity_is_a_warning(self):
        self.copy(
            "classification_data/train_data/a/0.png",
            "classification_data/train_data/a/9.png",
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "PASS")
        within = report["duplicates"]["within_identity"]
        self.assertEqual((within["groups"], within["redundant_files"]), (1, 1))
        self.assertEqual(report["duplicates"]["across_identities"]["groups"], 0)

    def test_conflicting_trial_labels_fail_even_in_reverse_order(self):
        self.add_trials("verification_data/v1.png verification_data/v0.png 0\n")
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "FAIL")
        conflict = report["trials"]["conflicting_labels"]
        self.assertEqual(conflict["pairs"], 1)
        self.assertEqual(
            [t["trial"] for t in conflict["examples"][0]["trials"]], [1, 3]
        )

    def test_repeated_self_and_identical_content_trials_are_warnings(self):
        self.copy("verification_data/v2.png", "verification_data/v2-copy.png")
        self.add_trials(
            "verification_data/v0.png verification_data/v1.png 1\n"
            "v3.png v3.png 1\n"
            "v2.png v2-copy.png 0\n"
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "PASS")
        trials = report["trials"]
        self.assertEqual(trials["repeated_pairs"]["extra_trials"], 1)
        self.assertEqual(trials["self_pairs"]["count"], 1)
        self.assertEqual(trials["identical_content_pairs"]["by_label"], {"0": 1})
        self.assertEqual(len(report["warnings"]), 3)

    def test_trial_image_with_unlisted_suffix_is_still_audited(self):
        self.copy("verification_data/v0.png", "verification_data/v9.dat")
        self.add_trials("v9.dat v3.png 0\n")
        report = audit_dataset(self.layout)
        self.assertEqual(report["partitions"]["verification_data"]["images"], 5)
        self.assertEqual(report["trials"]["identical_content_pairs"]["count"], 0)

    def test_corrupt_image_fails_with_its_path(self):
        (self.root / "classification_data/train_data/b/1.png").write_text(
            "not an image"
        )
        report = audit_dataset(self.layout)
        self.assertEqual(report["status"], "FAIL")
        failures = report["images"]["decode_failures"]
        self.assertEqual(failures["count"], 1)
        self.assertEqual(
            failures["examples"][0]["path"], "classification_data/train_data/b/1.png"
        )

    def test_spot_check_is_seeded_and_uses_relative_paths(self):
        first = audit_dataset(self.layout, seed=7)["spot_check"]
        self.assertEqual(first, audit_dataset(self.layout, seed=7)["spot_check"])
        self.assertEqual(len(first["train_samples"]), 6)
        self.assertEqual(len(first["trials"]), 2)
        self.assertNotIn(str(self.root), json.dumps(first))
        self.assertTrue(
            first["train_samples"][0]["path"].startswith(
                "classification_data/train_data/"
            )
        )

    def test_manifest_digest_changes_with_content(self):
        before = audit_dataset(self.layout)["dataset_manifest_sha256"]
        self.image("classification_data/train_data/a/0.png")
        self.assertNotEqual(
            before, audit_dataset(self.layout)["dataset_manifest_sha256"]
        )


class AuditCliTests(AuditFixture):
    def setUp(self):
        super().setUp()
        script = Path(__file__).resolve().parents[1] / "scripts/audit_face_data.py"
        spec = importlib.util.spec_from_file_location("audit_cli_for_test", script)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.output = self.root / "reports/audit.json"

    def run_cli(self, *extra) -> tuple[int, dict]:
        args = ["--data-root", str(self.root), "--output", str(self.output), *extra]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = self.cli.main(args)
        return code, json.loads(self.output.read_text())

    def test_clean_run_writes_pass_report_with_source_hashes(self):
        code, report = self.run_cli()
        self.assertEqual(
            (code, report["status"], report["task"]), (0, "PASS", "FAC-01.3")
        )
        self.assertIn(
            "src/facial_attendance/recognition/audit.py", report["source_sha256"]
        )

    def test_failed_audit_exits_nonzero(self):
        (self.root / "classification_data/train_data/a/0.png").write_text("broken")
        code, report = self.run_cli()
        self.assertEqual((code, report["status"]), (1, "FAIL"))

    def test_unknown_held_out_identity_is_recorded_as_failure(self):
        (self.root / "classification_data/val_data/b").rename(
            self.root / "classification_data/val_data/stranger"
        )
        code, report = self.run_cli()
        self.assertEqual((code, report["status"]), (1, "FAIL"))
        self.assertIn("Unknown identities in val_data", report["error"])

    def test_existing_report_is_never_overwritten(self):
        self.run_cli()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli.parse_arguments(
                ["--data-root", str(self.root), "--output", str(self.output)]
            )


if __name__ == "__main__":
    unittest.main()
