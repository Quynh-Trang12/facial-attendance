"""Device/version policy tests require no GPU or ML package installation in CI."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facial_attendance.environment import (
    check_versions,
    main,
    select_device,
    validate_python,
    validate_platform,
    version_matches,
    read_pins,
)


class RuntimeTests(unittest.TestCase):
    def test_supported_platform_targets(self):
        for system, machine in (("Linux", "x86_64"), ("Windows", "AMD64")):
            with self.subTest(system=system):
                validate_platform(system, machine, device="cpu")
                validate_platform(system, machine, device="cuda")
        validate_platform("Darwin", "arm64", macos_version="14.0", device="cpu")

    def test_mac_intel_and_rosetta_are_rejected_before_imports(self):
        with self.assertRaisesRegex(RuntimeError, "Intel/Rosetta"):
            validate_platform("Darwin", "x86_64", macos_version="15.0")

    def test_mac_requires_supported_os_release(self):
        for release in ("13.6", ""):
            with (
                self.subTest(release=release),
                self.assertRaisesRegex(RuntimeError, "macOS 14"),
            ):
                validate_platform("Darwin", "arm64", macos_version=release)

    def test_mac_cuda_is_rejected_without_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "Select --device cpu explicitly"):
            validate_platform("Darwin", "arm64", macos_version="14.0", device="cuda")

    def test_removed_snapshot_flag_is_rejected_before_execution(self):
        with (
            patch("facial_attendance.environment.check_environment") as check,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--strict-owner", "--operator", "test", "--output", "unused.json"])
        self.assertEqual(raised.exception.code, 2)
        check.assert_not_called()

    def test_unsupported_architecture_or_os_has_clear_error(self):
        for system, machine in (("Windows", "ARM64"), ("FreeBSD", "x86_64")):
            with self.subTest(system=system), self.assertRaises(RuntimeError):
                validate_platform(system, machine)

    def test_missing_ml_stack_produces_failed_report(self):
        # CI has no ML stack. Import errors must become inspectable evidence,
        # rather than preventing CLI parsing or crashing before the report exists.
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "missing-runtime.json"
            with (
                patch("facial_attendance.environment.check_versions", return_value={}),
                patch.dict("sys.modules", {"facial_attendance.runtime_checks": None}),
                redirect_stdout(StringIO()),
            ):
                code = main(
                    ["--device", "cpu", "--operator", "test", "--output", str(output)]
                )
            report = json.loads(output.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("ModuleNotFoundError", report["error"])
        self.assertNotIn("training", report)

    def test_cpu_is_explicitly_supported_without_cuda(self):
        self.assertEqual(select_device("cpu", cuda_available=False), "cpu")

    def test_unavailable_requested_gpu_does_not_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is inaccessible"):
            select_device("cuda", cuda_available=False)

    def test_available_gpu_is_selected(self):
        self.assertEqual(select_device("cuda", cuda_available=True), "cuda:0")

    def test_unknown_device_is_rejected(self):
        with self.assertRaises(ValueError):
            select_device("automatic-fallback", cuda_available=False)

    def test_team_python_accepts_313_patches_only(self):
        validate_python((3, 13, 16))
        validate_python((3, 13, 18))
        for actual in ((3, 12, 13), (3, 14, 0)):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                validate_python(actual)

    def test_versions_only_require_shared_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "requirements.txt").write_text("torch==2.13.0\n")
            # No reference Python file or archived lock is needed for this check.
            with (
                patch("facial_attendance.environment.version", return_value="2.13.0+cpu"),
                patch("facial_attendance.environment.sys.version_info", (3, 13, 17)),
            ):
                report = check_versions(root)
        self.assertEqual(report["versions"], {"torch": "2.13.0+cpu"})
        self.assertEqual(report["version_policy"], "portable_pins")

    def test_public_version_accepts_cpu_and_cuda_builds(self):
        self.assertTrue(version_matches("2.13.0", "2.13.0+cpu"))
        self.assertTrue(version_matches("2.13.0", "2.13.0+cu126"))
        self.assertFalse(version_matches("2.13.0", "2.11.0"))
        self.assertFalse(version_matches("2.13.0+cu126", "2.13.0+cpu"))

    def test_install_profiles_match_shared_public_versions(self):
        repo = Path(__file__).resolve().parents[1]
        common = read_pins(repo / "requirements.txt")
        for profile in ["requirements-cpu.txt", "requirements-cuda.txt"]:
            pins = [
                line
                for line in (repo / profile).read_text().splitlines()
                if line and not line.startswith(("#", "--"))
            ]
            for pin in pins:
                name, build = pin.split("==")
                self.assertTrue(version_matches(common[name], build), (profile, pin))


if __name__ == "__main__":
    unittest.main()
