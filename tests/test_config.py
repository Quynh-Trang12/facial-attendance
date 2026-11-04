"""Portable storage configuration must not depend on a developer's working directory."""

from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
import json
import tempfile
import unittest

from facial_attendance.config import load_paths
from facial_attendance.environment import main, read_pins
from facial_attendance.recognition.download import COMPETITION


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "project.toml"

    def write_config(self, **overrides):
        values = dict(
            kaggle_cache="data/.kagglehub",
            face_data="data/raw/face",
            liveness_data="data/raw/liveness",
            emotion_data="data/raw/emotion",
            checkpoints="artifacts/checkpoints",
            attendance="artifacts/attendance",
        )
        values.update(overrides)
        # Escape backslashes/quotes for TOML basic strings. Keep Unicode literal
        # so non-BMP characters are not encoded as JSON-only surrogate pairs.
        lines = ["[paths]"] + [
            f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in values.items()
        ]
        self.config.write_text("\n".join(lines), encoding="utf-8")

    def test_resolves_paths_without_creating_data(self):
        self.write_config()
        paths = load_paths(self.config, root=self.root)
        # The loader returns canonical paths; macOS temp roots can contain symlinks.
        self.assertEqual(paths.face_data, (self.root / "data/raw/face").resolve())
        self.assertFalse(paths.face_data.exists())
        self.assertFalse(paths.checkpoints.exists())

    def test_supports_external_dataset_storage(self):
        # Exercise Windows backslash escapes even when this suite runs on Linux.
        for folder in ("external-face-data", r"external\Users\O'Brien\face-data"):
            with self.subTest(folder=folder):
                external = self.root / folder
                self.write_config(face_data=str(external))
                self.assertEqual(
                    load_paths(self.config, root=self.root).face_data,
                    external.resolve(),
                )

    def test_repository_dataset_default_matches_download_cache(self):
        # This checks configuration coherence without requiring private data in CI.
        repo = Path(__file__).resolve().parents[1]
        paths = load_paths(repo / "configs/project.toml", root=repo)
        self.assertEqual(
            paths.face_data, paths.kaggle_cache / "competitions" / COMPETITION
        )

    def test_rejects_missing_paths(self):
        self.config.write_text('[paths]\nface_data = "data/face"\n')
        with self.assertRaises(ValueError):
            load_paths(self.config, root=self.root)

    def test_rejects_empty_path(self):
        self.write_config(checkpoints="")
        with self.assertRaises(ValueError):
            load_paths(self.config, root=self.root)

    def test_exact_pins_accept_comments_and_cuda_versions(self):
        path = self.root / "requirements.txt"
        path.write_text(
            "# Project pins\ntorch==2.13.0+cu126\n\nnumpy==2.2.6 # arrays\n"
        )
        self.assertEqual(read_pins(path), {"torch": "2.13.0+cu126", "numpy": "2.2.6"})

    def test_rejects_unpinned_dependencies(self):
        path = self.root / "requirements.txt"
        path.write_text("torch>=2\n")
        with self.assertRaises(ValueError):
            read_pins(path)

    def test_cli_preserves_existing_evidence(self):
        output = self.root / "earlier-run.json"
        output.write_text('{"status": "earlier evidence"}')
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main(["--output", str(output), "--operator", "test"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(output.read_text(), '{"status": "earlier evidence"}')


if __name__ == "__main__":
    unittest.main()
