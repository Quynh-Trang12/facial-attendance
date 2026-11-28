"""Plain-Python notebook execution retains rich output and primary failures."""

import ast
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from facial_attendance import notebook_runtime as runtime


def fixture(path, sources):
    value = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": f"case-{index}",
                "metadata": {},
                "source": source,
                "outputs": [],
                "execution_count": None,
            }
            for index, source in enumerate(sources)
        ],
    }
    path.write_text(json.dumps(value))


class NotebookRuntimeTests(unittest.TestCase):
    def test_source_snapshot_preserves_existing_bytes_and_rejects_destination_escape(
        self,
    ):
        from facial_attendance.liveness.dataset_files import (
            file_sha256,
            snapshot_sources,
            source_path,
        )

        relative = "models/binary_model.py"
        binding = {relative: file_sha256(source_path(relative))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "snapshot"
            snapshot_sources(destination, binding)
            target = destination / relative
            target.write_bytes(b"existing source evidence")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                snapshot_sources(destination, binding)
            self.assertEqual(target.read_bytes(), b"existing source evidence")
            escaped = root / "escaped-snapshot"
            outside = root / "outside"
            escaped.mkdir()
            outside.mkdir()
            try:
                (escaped / "models").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest(
                    "Snapshot copy passed; symlink escape check needs symlink privileges"
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                snapshot_sources(escaped, binding)
            self.assertFalse((outside / "binary_model.py").exists())

    def test_liveness_dependencies_follow_the_declared_layers(self):
        package = Path(__file__).resolve().parents[1] / "src/facial_attendance/liveness"
        forbidden = {
            "models": {"datasets", "training", "experiments"},
            "datasets": {"training", "experiments"},
            "training": {"experiments"},
        }
        for layer, blocked in forbidden.items():
            for source in (package / layer).glob("*.py"):
                for node in ast.walk(ast.parse(source.read_text())):
                    modules = (
                        [node.module]
                        if isinstance(node, ast.ImportFrom)
                        else [item.name for item in node.names]
                        if isinstance(node, ast.Import)
                        else []
                    )
                    for module in modules:
                        if module and module.startswith("facial_attendance.liveness."):
                            dependency = module.split(".")[2]
                            self.assertNotIn(
                                dependency,
                                blocked,
                                f"{source.name} imports higher layer {module}",
                            )

    def test_colab_staging_preserves_script_packages_and_resolves_audit_tests(self):
        project = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project / "notebooks/liveness-training-and-evaluation.ipynb").read_text()
        )
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("id") == "release-script-verification"
        )
        scope = {
            "Path": Path,
            "verified_source": lambda root, record: (
                root / record["delivery_path"]
            ).read_bytes(),
        }
        exec(compile(source, "notebook-script-staging", "exec"), scope)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "bootstrap"
            for relative in (
                "scripts/evaluation/score_liveness_mp4_folder.py",
                "scripts/liveness_camera_inputs.py",
            ):
                target = scope["stage_release_script"](
                    project, {"delivery_path": relative}, destination
                )
                self.assertEqual(
                    target.relative_to(destination),
                    Path(relative).relative_to("scripts"),
                )
                self.assertEqual(target.read_bytes(), (project / relative).read_bytes())
            scorer = destination / "evaluation/score_liveness_mp4_folder.py"
            self.assertTrue((scorer.parents[1] / "liveness_camera_inputs.py").is_file())
            audit = json.loads(
                (
                    project / "notebooks/liveness-data-preparation-and-audit.ipynb"
                ).read_text()
            )
            source = next(
                "".join(cell["source"])
                for cell in audit["cells"]
                if cell.get("id") == "b118ea5561624da68c537baed56e602f"
            )
            commands = []
            # Exercise stage routing without downloading data or requiring an
            # 85 GiB extraction VM. The real cache-publication test uses real disk.
            scope = {
                "IS_COLAB": True,
                "ACTION": "audit",
                "stage": lambda name, *args: commands.append((name, args)),
                "shutil": SimpleNamespace(
                    disk_usage=lambda _: SimpleNamespace(free=90 * 2**30)
                ),
                "data": destination / "unextracted-data",
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "audit-test-routing", "exec"), scope)
            self.assertEqual(
                {name for name, _ in commands}, {"selection-tests", "audit-tests"}
            )
            for name, arguments in commands:
                self.assertEqual(arguments[:2], ("-m", "unittest"))
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromName(arguments[2])
                self.assertFalse(loader.errors, (name, loader.errors))
                self.assertGreater(suite.countTestCases(), 0)

    def test_shared_cli_retains_specialized_training_arguments_and_epoch_archives(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "shared_liveness_cli", scripts / "run_liveness_notebook.py"
        )
        cli = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(cli)
        backend = Mock()
        with patch.object(cli, "import_module", return_value=backend) as load:
            cli.main(["--action", "train-source", "--", "--epochs", "25"])
            load.assert_called_with("training.train_liveness_aenet")
            backend.main.assert_called_once_with(
                ["--epochs", "25", "--retain-epoch-checkpoints"]
            )
        for action in ("train-baseline", "train-multitask"):
            before = list(sys.argv)
            forwarded = []
            backend = Mock()
            backend.main.side_effect = lambda arguments: forwarded.extend(arguments)
            with patch.object(cli, "import_module", return_value=backend):
                cli.main(["--action", action, "--", "--epochs", "2"])
            self.assertEqual(forwarded, ["--epochs", "2"])
            self.assertEqual(sys.argv, before)

    def test_shared_cli_inspect_does_not_import_training_or_gpu_staging(self):
        with tempfile.TemporaryDirectory(prefix="inspect path's ") as temporary:
            root = Path(temporary)
            notebook, output = root / "input.ipynb", root / "rendered.ipynb"
            fixture(
                notebook,
                [
                    "import sys\n"
                    "assert 'torch' not in sys.modules\n"
                    "assert 'run_liveness_public_test' not in sys.modules\n"
                    "assert 'stage_liveness_training_runtime' not in sys.modules\n"
                    "from facial_attendance.notebook_runtime import HTML, display\n"
                    "display(HTML('<b>saved evidence</b>'))"
                ],
            )
            before = notebook.read_bytes()
            project = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/run_liveness_notebook.py"),
                    "--notebook",
                    str(notebook),
                    "--output",
                    str(output),
                ],
                cwd=project,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["metadata"]["local_execution"]["completed"])
            self.assertEqual(
                result["cells"][0]["outputs"][0]["data"]["text/html"],
                "<b>saved evidence</b>",
            )
            self.assertEqual(notebook.read_bytes(), before)

    def test_html_and_shared_state_without_ipython_keep_output_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "analysis.ipynb"
            fixture(
                path,
                [
                    'print("before"); value = 4\n'
                    "from facial_attendance.notebook_runtime import HTML, display\n"
                    'display(HTML("<b>result</b>")); print("after")',
                    "assert value == 4; assert LOCAL_PROJECT.is_dir(); print(ACTION)",
                ],
            )
            with (
                patch.object(runtime, "get_ipython", None),
                patch.object(runtime, "ipython_display", None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = runtime.execute_notebook(path, project=root)
            outputs = result["cells"][0]["outputs"]
            self.assertEqual(
                [o["output_type"] for o in outputs],
                ["stream", "display_data", "stream"],
            )
            self.assertEqual(outputs[1]["data"]["text/html"], "<b>result</b>")
            self.assertTrue(result["metadata"]["local_execution"]["completed"])

    def test_failure_retains_output_and_stops_later_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "analysis.ipynb"
            fixture(
                path,
                [
                    'print("evidence before failure"); raise ValueError("bad input")',
                    '(LOCAL_PROJECT / "unexpected").write_text("must not run")',
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "stopped at case-0"):
                    runtime.execute_notebook(path, project=root)
            result = json.loads(path.read_text())
            self.assertFalse((root / "unexpected").exists())
            self.assertFalse(result["metadata"]["local_execution"]["completed"])
            self.assertEqual(result["cells"][0]["outputs"][-1]["ename"], "ValueError")
            self.assertIn(
                "evidence before failure", result["cells"][0]["outputs"][0]["text"]
            )

    def test_plot_import_and_show_in_one_cell_is_saved_without_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "analysis.ipynb"
            fixture(
                path,
                [
                    'import matplotlib\nmatplotlib.use("Agg")\n'
                    "import matplotlib.pyplot as plt\nplt.plot([0, 1], [0, 1]); plt.show()"
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = runtime.execute_notebook(path, project=root)
            self.assertTrue(
                any(
                    "image/png" in o.get("data", {})
                    for o in result["cells"][0]["outputs"]
                )
            )


if __name__ == "__main__":
    unittest.main()
