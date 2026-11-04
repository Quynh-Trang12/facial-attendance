"""The local and API entry points must actually consume the shared configuration."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DataCliTests(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_face_data.py"
        spec = importlib.util.spec_from_file_location("data_check_for_test", script)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = str(Path(self.temp.name) / "result.json")

    def test_no_input_flag_uses_configured_dataset(self):
        args = self.cli.parse_arguments(["--output", self.output])
        with patch.object(
            self.cli.FaceLayout, "discover", return_value="layout"
        ) as discover:
            layout, paths = self.cli.resolve_layout(args)
        discover.assert_called_once_with(paths.face_data)
        self.assertEqual(layout, "layout")

    def test_api_mode_uses_configured_cache(self):
        args = self.cli.parse_arguments(["--download", "--output", self.output])
        with patch.object(
            self.cli, "download_face_data", return_value="layout"
        ) as download:
            layout, paths = self.cli.resolve_layout(args)
        download.assert_called_once_with(cache_dir=paths.kaggle_cache)

    def test_explicit_dataset_path_overrides_config(self):
        override = Path(self.temp.name) / "mounted-colab-data"
        args = self.cli.parse_arguments(
            ["--data-root", str(override), "--output", self.output]
        )
        with patch.object(
            self.cli.FaceLayout, "discover", return_value="layout"
        ) as discover:
            self.cli.resolve_layout(args)
        discover.assert_called_once_with(override)


if __name__ == "__main__":
    unittest.main()
