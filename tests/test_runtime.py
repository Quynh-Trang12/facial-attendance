"""Device/version policy tests require no GPU or ML package installation in CI."""

from pathlib import Path
import unittest

from facial_attendance.environment import (
    select_device,
    validate_python,
    version_matches,
    read_pins,
)


class RuntimeTests(unittest.TestCase):
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

    def test_portable_python_does_not_require_owner_patch(self):
        validate_python((3, 12, 13))
        validate_python((3, 13, 18))
        with self.assertRaises(RuntimeError):
            validate_python((3, 11, 9))

    def test_owner_replay_still_enforces_owner_patch(self):
        with self.assertRaises(RuntimeError):
            validate_python((3, 13, 18), owner_patch="3.13.16")

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
