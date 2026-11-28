"""Recovery must preserve partial transfers and surface computation failures."""

import ast
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# These tests exercise the Linux/Colab process and cache-staging contract.
if sys.platform != "linux":
    raise unittest.SkipTest(
        "Linux/Colab runtime staging; portable inspect/inference are tested separately"
    )

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = load("liveness_training_health")
with patch.dict(sys.modules, {"liveness_training_health": health}):
    restore = load("stage_liveness_training_runtime")
launcher = load("supervise_liveness_training_process")


class TrainingRestoreTests(unittest.TestCase):
    def test_notebook_progress_counts_sequential_prefix_but_not_sparse_rclone_size(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            for split in ("train", "validation"):
                (source / split).mkdir(parents=True)
                (source / split / "decoded-faces.json").write_text("{}")
                (source / split / "decoded-faces.npy").write_bytes(b"x" * 100)
                (target / split).mkdir(parents=True)
            (target / "train/decoded-faces.npy.copying").write_bytes(b"x" * 40)
            with (target / "validation/decoded-faces.npy.rclone-copying").open(
                "wb"
            ) as file:
                file.truncate(100)
            progress = launcher.CacheStagingProgress(
                ["--prepared-cache-root", str(source), "--decoded-cache", str(target)]
            )
            self.assertEqual(progress.copied_bytes(), 40)
            progress.record(
                {
                    "cache_transfer": "rclone-progress",
                    "split": "validation",
                    "file": "decoded-faces.npy",
                    "copied_bytes": 25,
                }
            )
            self.assertEqual(progress.copied_bytes(), 65)
            self.assertIn("(32.5%)", progress.message())
            progress.record(
                {
                    "phase": "cache-sha256",
                    "split": "train",
                    "file": "decoded-faces.npy",
                    "checked_bytes": 25,
                    "total_bytes": 100,
                    "bytes_per_second": 5,
                }
            )
            text = progress.message()
            self.assertIn("SHA-256: train/decoded-faces.npy 25.0%", text)
            self.assertIn("check ETA ~0m 15s", text)
            self.assertNotIn(temporary, text)
            self.assertIn("unavailable", progress.eta(10, 0))

    def test_reported_hash_matches_native_digest_without_exposing_paths(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train/decoded-faces.npy"
            path.parent.mkdir()
            path.write_bytes(b"verified pixels")
            output = io.StringIO()
            with redirect_stdout(output):
                digest = restore.sha(path, report_progress=True)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[0]["checked_bytes"], 0)
            self.assertEqual(events[-1]["checked_bytes"], path.stat().st_size)
            self.assertNotIn(temporary, output.getvalue())

    def test_notebook_status_updates_one_display_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            for split in ("train", "validation"):
                (source / split).mkdir(parents=True)
                (source / split / "decoded-faces.json").write_text("{}")
                (source / split / "decoded-faces.npy").write_bytes(b"x")
            status = Mock()
            options = [
                "--prepared-cache-root",
                str(source),
                "--decoded-cache",
                str(target),
            ]
            launcher.launch(
                [sys.executable, "-c", "pass", *options],
                root / "ok.log",
                on_status=status,
            )
            self.assertIn("complete", status.call_args.args[0])
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.launch(
                    [sys.executable, "-c", "raise SystemExit(3)", *options],
                    root / "failed.log",
                    on_status=status,
                )
            self.assertIn("failed", status.call_args.args[0])

    def test_workflow_status_separates_resume_training_and_saved_validation(self):
        clock = [0.0]
        command = [
            "--arms",
            "binary-classification",
            "aenet-full",
            "--screening-epochs",
            "15",
            "--validation-only",
        ]
        with patch.object(launcher.time, "monotonic", side_effect=lambda: clock[0]):
            progress = launcher.WorkflowProgress(command)
            progress.record(
                {
                    "phase": "resume-plan",
                    "arm": "binary-classification",
                    "completed_epochs": 7,
                }
            )
            progress.record(
                {
                    "phase": "workflow-phase",
                    "step": "training",
                    "arm": "binary-classification",
                }
            )
            for elapsed, count in ((30, 12000), (60, 24000)):
                clock[0] = elapsed
                progress.record(
                    {
                        "phase": "training",
                        "epoch": 8,
                        "samples": count,
                        "total_samples": 444000,
                        "elapsed_seconds": elapsed,
                    }
                )
            text = progress.message()
            self.assertIn("epoch 8/15", text)
            self.assertIn("24,000/444,000", text)
            self.assertIn("~17m 30s", text)
            self.assertIn("7/30 epochs saved", text)
            self.assertNotIn("Latest saved validation", text)
            clock[0] = 120
            self.assertIn("last progress 60s ago", progress.message())
            progress.record(
                {
                    "arm": "binary-classification",
                    "epoch": 8,
                    "total_updates": 100,
                    "validation_ACER": 0.02,
                }
            )
            self.assertIn("8/30 epochs saved", progress.message())
            self.assertIn("ACER 2.00%", progress.message())
            self.assertIn("deferred", progress.finished(False))

    def test_evaluation_progress_does_not_require_training_epoch_fields(self):
        progress = launcher.WorkflowProgress(
            ["python", "-u", "evaluation/score_liveness_image_split.py"]
        )
        progress.record(
            {
                "phase": "evaluation",
                "samples": 250,
                "total_samples": 1000,
                "elapsed_seconds": 5,
                "unit": "images",
            }
        )
        text = progress.message()
        self.assertIn("250/1,000 images (25.0%)", text)
        self.assertIn("~0m 15s", text)
        self.assertNotIn("epochs saved", text)
        self.assertIn("Evaluation task complete", progress.finished(False))

    def test_environment_status_does_not_claim_cache_or_training_completion(self):
        progress = launcher.WorkflowProgress(["--runtime-only"])
        progress.record({"phase": "source-verification", "completed": 2, "total": 8})
        self.assertIn("2/8 files", progress.message())
        progress.record({"phase": "workflow-phase", "step": "environment"})
        self.assertIn("Environment", progress.message())
        self.assertNotIn("Cache staging complete", progress.finished(False))
        self.assertIn("Environment failed", progress.finished(True))

    def test_launcher_reports_fallback_once_without_streaming_verbose_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "native.log"
            output = io.StringIO()
            code = "import json; event=json.dumps({'cache_transfer':'DriveFS-fallback','reason':'HTTPError'}); print(event); print(event)"
            with redirect_stdout(output):
                launcher.launch([sys.executable, "-c", code], log)
            self.assertEqual(output.getvalue().count("Cache transfer fallback:"), 1)
            self.assertEqual(log.read_text().count("HTTPError"), 2)

    def test_launcher_keeps_verbose_log_out_of_notebook_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "native.log"
            output = io.StringIO()
            code = "import json; print('verbose dependency details'); print(json.dumps({'arm':'binary-classification','epoch':7,'total_updates':12152,'validation_ACER':0.0043}))"
            with redirect_stdout(output):
                launcher.launch([sys.executable, "-c", code], log)
            self.assertNotIn("verbose dependency details", output.getvalue())
            self.assertNotIn("validation_ACER", output.getvalue())
            self.assertIn("verbose dependency details", log.read_text())

    def test_ram_cache_requires_capacity_and_training_headroom(self):
        gib = 2**30
        self.assertTrue(
            restore.ram_cache_fits(72 * gib, 88 * gib, 160 * gib, 180 * gib)
        )
        self.assertFalse(
            restore.ram_cache_fits(72 * gib, 64 * gib, 160 * gib, 180 * gib)
        )
        self.assertFalse(
            restore.ram_cache_fits(72 * gib, 88 * gib, 80 * gib, 180 * gib)
        )
        self.assertFalse(restore.ram_cache_fits(72 * gib, 88 * gib, 83 * gib, 83 * gib))

    def test_storage_selection_preserves_existing_ssd_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ssd = root / "ssd"
            (ssd / "train").mkdir(parents=True)
            (ssd / "train/decoded-faces.npy.copying").write_bytes(b"prefix")
            chosen, reason = restore.choose_cache_directory(root / "unread-source", ssd)
            self.assertEqual(chosen, ssd)
            self.assertIn("reusing", reason)

    def test_rclone_setup_checks_official_archive_before_installing(self):
        import hashlib
        import zipfile

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("rclone-v1.60.1-linux-amd64/rclone", b"fixture executable")
        payload = archive.getvalue()
        inventory = (
            "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\n"
            f"{hashlib.sha256(payload).hexdigest()}  rclone-v1.60.1-linux-amd64.zip\n"
            "\n-----BEGIN PGP SIGNATURE-----\nfixture\n"
        ).encode()
        original_temporary = tempfile.TemporaryDirectory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "drive/MyDrive").mkdir(parents=True)

            def runtime_path(value):
                return root / str(value).removeprefix("/content/")

            def download(request, **kwargs):
                url = request.full_url
                self.assertTrue(url.startswith("https://downloads.rclone.org/v1.60.1/"))
                self.assertEqual(
                    request.get_header("User-agent"),
                    "facial-attendance-cache-setup/1.0",
                )
                return io.BytesIO(inventory if url.endswith("SHA256SUMS") else payload)

            with (
                patch.object(restore.shutil, "which", return_value=None),
                patch.object(restore, "Path", side_effect=runtime_path),
                patch.object(restore, "urlopen", side_effect=download),
                patch.object(
                    restore.tempfile,
                    "TemporaryDirectory",
                    side_effect=lambda **kwargs: original_temporary(dir=root),
                ),
            ):
                self.assertEqual(restore.prepare_rclone(), str(root / "rclone"))
            self.assertEqual((root / "rclone").read_bytes(), b"fixture executable")
            self.assertTrue((root / "rclone").stat().st_mode & 0o100)

    def test_rclone_setup_corruption_keeps_drivefs_fallback(self):
        original_temporary = tempfile.TemporaryDirectory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "drive/MyDrive").mkdir(parents=True)

            def runtime_path(value):
                return root / str(value).removeprefix("/content/")

            responses = [
                io.BytesIO(b"0000  rclone-v1.60.1-linux-amd64.zip\n"),
                io.BytesIO(b"corrupt release bytes"),
            ]
            with (
                patch.object(restore.shutil, "which", return_value=None),
                patch.object(restore, "Path", side_effect=runtime_path),
                patch.object(restore, "urlopen", side_effect=responses),
                patch.object(
                    restore.tempfile,
                    "TemporaryDirectory",
                    side_effect=lambda **kwargs: original_temporary(dir=root),
                ),
            ):
                self.assertIsNone(restore.prepare_rclone())
            self.assertFalse((root / "rclone").exists())

    def test_epoch_completion_replaces_stale_batch_progress_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publisher = health.TrainingHealth(root, "fixture", clock=lambda: 1000)
            publisher.state["arm"] = "binary-classification"
            publisher.record(
                json.dumps(
                    {
                        "phase": "training",
                        "epoch": 7,
                        "samples": 439808,
                        "total_samples": 444357,
                        "optimizer_updates": 1718,
                    }
                )
            )
            publisher.record(
                json.dumps(
                    {
                        "arm": "binary-classification",
                        "epoch": 7,
                        "total_updates": 12152,
                        "train_samples": 444357,
                    }
                )
            )
            saved = json.loads((root / "training-health.json").read_text())
            self.assertEqual(saved["completed_epoch"], 7)
            self.assertEqual(saved["samples"], 444357)
            self.assertEqual(saved["total_updates"], 12152)
            self.assertNotIn("epoch_optimizer_updates", saved)

    def test_rclone_transfer_verifies_bytes_before_committing(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/train/decoded-faces.npy"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"verified cache bytes")
            target = root / "target/train/decoded-faces.npy"
            credentials = SimpleNamespace(
                valid=True, token="fixture", expiry=datetime.now(UTC)
            )
            transfer = restore.RcloneCacheDownload("rclone", "folder-id", credentials)
            progress = Mock()

            def download(command, **kwargs):
                Path(command[3]).write_bytes(source.read_bytes())
                self.assertEqual(
                    command[command.index("--multi-thread-streams") + 1], "16"
                )
                self.assertNotIn("fixture", " ".join(command))
                return 0

            optional = {
                "google.auth.transport.requests": SimpleNamespace(Request=Mock())
            }
            with (
                patch.dict(sys.modules, optional),
                patch.object(
                    restore.RcloneCacheDownload, "run_download", side_effect=download
                ),
            ):
                self.assertTrue(
                    transfer.copy(source, target, restore.sha(source), progress)
                )
            self.assertEqual(target.read_bytes(), source.read_bytes())
            progress.put.assert_called_once_with(source.stat().st_size)

    def test_rclone_sparse_failure_and_corrupt_download_fall_back_safely(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/train/decoded-faces.npy"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original")
            credentials = SimpleNamespace(
                valid=True, token="fixture", expiry=datetime.now(UTC)
            )
            transfer = restore.RcloneCacheDownload("rclone", "folder-id", credentials)
            optional = {
                "google.auth.transport.requests": SimpleNamespace(Request=Mock())
            }
            for corrupt in (False, True):
                target = root / str(corrupt) / "train/decoded-faces.npy"

                def download(command, **kwargs):
                    pending = Path(command[3])
                    if corrupt:
                        pending.write_bytes(b"changed!")
                    else:
                        with pending.open("wb") as handle:
                            handle.seek(4096)
                            handle.write(b"x")
                        raise subprocess.CalledProcessError(1, command)

                with (
                    patch.dict(sys.modules, optional),
                    patch.object(
                        restore.RcloneCacheDownload,
                        "run_download",
                        side_effect=download,
                    ),
                ):
                    self.assertFalse(
                        transfer.copy(source, target, restore.sha(source), Mock())
                    )
                self.assertFalse(target.exists())
                self.assertFalse(
                    target.with_name(target.name + ".rclone-copying").exists()
                )
                restore.copy_resumable(source, target, source.stat().st_size)
                self.assertEqual(restore.sha(target), restore.sha(source))

    def test_existing_sequential_partial_is_preserved_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/train/decoded-faces.npy"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original")
            target = root / "target/train/decoded-faces.npy"
            target.parent.mkdir(parents=True)
            target.with_name(target.name + ".copying").write_bytes(b"orig")
            transfer = restore.RcloneCacheDownload("rclone", "folder-id", None)
            with patch.object(restore.subprocess, "run") as download:
                self.assertFalse(
                    transfer.copy(source, target, restore.sha(source), Mock())
                )
                download.assert_not_called()
            restore.copy_resumable(source, target, source.stat().st_size)
            self.assertEqual(target.read_bytes(), b"original")

    def test_worker_policy_respects_affinity_quota_and_explicit_serial_loading(self):
        for cpu_count, quota, requested, expected in (
            (24, "400000 100000", None, 4),
            (8, "max 100000", None, 8),
            (8, "250000 100000", None, 3),
            (24, "max 100000", None, 12),
            (24, "max 100000", 20, 20),
            (8, "200000 100000", 12, 2),
            (8, "200000 100000", 0, 0),
            (None, None, None, 1),
        ):
            quota_file = Mock()
            quota_file.is_file.return_value = quota is not None
            quota_file.read_text.return_value = quota
            with patch.object(health.os, "process_cpu_count", return_value=cpu_count):
                self.assertEqual(
                    health.runtime_loader_workers(requested, quota_file=quota_file),
                    expected,
                )
        with self.assertRaises(ValueError):
            health.runtime_loader_workers(-1)

    def test_notebook_executes_the_shared_worker_policy(self):
        notebook = json.loads(
            (
                SCRIPTS.parent / "notebooks/liveness-training-and-evaluation.ipynb"
            ).read_text()
        )
        cell = next(
            c for c in notebook["cells"] if c.get("id") == "runtime-loader-workers"
        )
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "bootstrap": Path(directory),
                "DRIVE_PROJECT": SCRIPTS.parent,
                "release": {"files": {"scripts/liveness_training_health.py": {}}},
                "stage_release_script": lambda *args: (
                    SCRIPTS / "liveness_training_health.py"
                ),
                "importlib": importlib,
                "experiment": {"loader_workers": 12},
            }
            with (
                patch.object(health.os, "process_cpu_count", return_value=2),
                redirect_stdout(io.StringIO()),
            ):
                exec(
                    compile("".join(cell["source"]), "notebook-workers", "exec"),
                    namespace,
                )
            self.assertEqual(namespace["LOADER_WORKERS"], 2)
            self.assertEqual(namespace["MAX_LOADER_WORKERS"], 2)
        command = next(
            "".join(row["source"])
            for row in notebook["cells"]
            if row["cell_type"] == "code"
            and '"--loader-workers"' in "".join(row["source"])
        )
        self.assertIn(
            "'--loader-workers', str(LOADER_WORKERS)", ast.unparse(ast.parse(command))
        )

    def test_notebook_uses_folder_id_after_project_moves(self):
        from tests.liveness.test_notebook_drive_resolver import (
            API,
            bootstrap_scope,
            folder,
        )

        api = API(
            {
                "project": folder("project", "facial-attendance", "course"),
                "course": folder("course", "Team files", "mine"),
            }
        )
        resolve = bootstrap_scope()["resolve_drive_folder"]
        url = "https://drive.google.com/drive/folders/project"
        self.assertEqual(
            resolve(api, url, "mine")[1], ("Team files", "facial-attendance")
        )
        api.nodes["course"]["name"] = "Moved course folder"
        self.assertEqual(
            resolve(api, url, "mine")[1], ("Moved course folder", "facial-attendance")
        )
        self.assertFalse(api.created)

    def test_cli_holds_the_process_lock_for_the_entire_restore(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "saved"
            run.mkdir()
            (run / "manifest.json").write_text("{}")
            args = [
                "stage_liveness_training_runtime.py",
                "--project-root",
                str(root),
                "--run-root",
                str(run),
                "--prepare-only",
            ]
            lock_path = root / "transfer/training.lock"

            def competing_restore(*arguments):
                with lock_path.open("a") as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)

            with (
                patch.object(sys, "argv", args),
                patch.object(sys, "stdout", io.StringIO()),
                patch.object(sys, "stderr", io.StringIO()),
                patch.object(restore, "restore", side_effect=competing_restore),
            ):
                restore.main()
            with lock_path.open("a") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_current_release_creates_output_without_historical_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-output"
            inventory = root / "release.json"
            inventory.write_text('{"files": {}}')
            args = [
                "stage_liveness_training_runtime.py",
                "--project-root",
                str(root / "project"),
                "--run-root",
                str(output),
                "--release-inventory",
                str(inventory),
                "--prepare-only",
                "--arms",
                "aenet-full",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(sys, "stdout", io.StringIO()),
                patch.object(sys, "stderr", io.StringIO()),
                patch.object(restore, "restore") as execute,
            ):
                restore.main()
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0].release_inventory, inventory)
            self.assertFalse((output / "manifest.json").exists())

    def test_release_inventory_stages_without_any_historical_inputs(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "delivery"
            project = root / "project"
            source.mkdir()
            (source / "input.txt").write_text("current input")
            digest = hashlib.sha256((source / "input.txt").read_bytes()).hexdigest()
            inventory = source / "release.json"
            inventory.write_text(
                json.dumps(
                    {
                        "files": {
                            "data/current.txt": {
                                "delivery_path": "input.txt",
                                "bytes": 13,
                                "sha256": digest,
                            },
                            "notebooks/liveness-training-and-evaluation.ipynb": {
                                "delivery_path": "interactive.ipynb",
                                "bytes": 1,
                                "sha256": "old",
                            },
                        }
                    }
                )
            )
            # Notebook outputs are mutable; runtime inputs must still match.
            (source / "interactive.ipynb").write_text("autosaved output")
            args = SimpleNamespace(
                project_root=project,
                release_inventory=inventory,
                source_root=source,
                run_root=root / "no-old-run",
            )
            progress = SimpleNamespace(state={}, publish=Mock())
            restore.stage_verified_inputs(args, progress)
            self.assertEqual(
                (project / "data/current.txt").read_text(), "current input"
            )
            self.assertTrue((project / "transfer/release-inventory.json").is_file())
            (project / "data/current.txt").write_text("modified")
            with self.assertRaisesRegex(ValueError, "Existing staged input differs"):
                restore.stage_verified_inputs(args, progress)

    def test_notebook_uses_one_output_without_historical_selection(self):
        notebook = json.loads(
            (
                SCRIPTS.parent / "notebooks/liveness-training-and-evaluation.ipynb"
            ).read_text()
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for historical_name in (
            "RUN_NAME",
            "COMPARISON_NAME",
            "choose_experiment",
            "original-control",
        ):
            self.assertNotIn(historical_name, code)
        self.assertIn('"--release-inventory"', code)
        self.assertIn(
            "'--run-root', str(TRAINING_OUTPUT)", ast.unparse(ast.parse(code))
        )

    def test_storage_preflight_accounts_for_both_splits_before_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for split in ("train", "validation"):
                directory = source / split
                directory.mkdir(parents=True)
                (directory / "decoded-faces.json").write_text(
                    json.dumps({"shape": [3, 224, 224, 3]})
                )
            with patch.object(
                restore.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=2 * 2**30 + 700000),
            ):
                with self.assertRaisesRegex(ValueError, "both face arrays"):
                    restore.ensure_cache_storage(source, root / "destination")
            with patch.object(
                restore.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=3 * 2**30),
            ):
                restore.ensure_cache_storage(source, root / "destination")

    def test_supported_cuda_builds_reuse_existing_environment_without_installation(
        self,
    ):
        for build in ("2.13.0+cu126", "2.13.0+cu130"):
            with self.subTest(build=build), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                python = root / ".venv/bin/python"
                python.parent.mkdir(parents=True)
                python.touch()
                modules = {
                    "sys": SimpleNamespace(version_info=(3, 13)),
                    "torch": SimpleNamespace(__version__=build),
                    "numpy": SimpleNamespace(__version__="2.2.6"),
                    "PIL": SimpleNamespace(__version__="11.2.1"),
                }

                def probe(command, **kwargs):
                    def import_fixture(name, *args, **kw):
                        return modules[name]

                    exec(
                        command[2],
                        {"__builtins__": {"__import__": import_fixture, "str": str}},
                    )
                    return SimpleNamespace(returncode=0)

                with (
                    patch.object(restore.subprocess, "run", side_effect=probe),
                    patch.object(restore, "execute") as execute,
                ):
                    result = restore.prepare_environment(
                        SimpleNamespace(project_root=root, prepare_only=True),
                        SimpleNamespace(phase=Mock()),
                    )
                self.assertEqual(result, python)
                execute.assert_not_called()

    def test_unavailable_cuda_does_not_trigger_dependency_reinstallation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            args = SimpleNamespace(project_root=root, prepare_only=False)
            failure = subprocess.CalledProcessError(
                1, [str(python), "-c", "CUDA probe"]
            )
            with (
                patch.object(
                    restore.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                patch.object(restore, "execute", side_effect=failure) as execute,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    restore.prepare_environment(args, SimpleNamespace(phase=Mock()))
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0][0], python)

    def test_cache_publication_and_staging_verify_bytes_and_reject_corruption(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            inventory = data / "audits/current-schema-2026-11-12/inventory.csv"
            inventory.parent.mkdir(parents=True)
            inventory.write_text("fixture inventory")
            source, published, staged = (
                root / "source",
                root / "published",
                root / "staged",
            )
            for split, name in (
                ("train", "reviewed_project_train.json"),
                ("validation", "project_validation.json"),
            ):
                annotation = data / name
                annotation.write_text("fixture annotations")
                directory = source / split
                directory.mkdir(parents=True)
                np.save(
                    directory / "decoded-faces.npy",
                    np.zeros((2, 224, 224, 3), dtype=np.uint8),
                )
                receipt = {
                    "schema": "liveness-decoded-faces-v1",
                    "dtype": "uint8",
                    "shape": [2, 224, 224, 3],
                    "sha256": restore.sha(directory / "decoded-faces.npy"),
                    "data": {
                        "inventory_sha256": restore.sha(inventory),
                        "annotations_sha256": restore.sha(annotation),
                        "samples": 2,
                        "crop_recipe": "fixture-source-crop-v1",
                    },
                }
                (directory / "decoded-faces.json").write_text(json.dumps(receipt))
            publisher = SimpleNamespace(state={}, publish=Mock())
            restore.copy_decoded_face_caches(
                source, published, data, publisher, verify_destination=False
            )
            restore.copy_decoded_face_caches(
                published, staged, data, publisher, verify_destination=True
            )
            for split in ("train", "validation"):
                self.assertEqual(
                    restore.sha(source / split / "decoded-faces.npy"),
                    restore.sha(staged / split / "decoded-faces.npy"),
                )
                self.assertIn(
                    "array_identity",
                    json.loads((staged / split / "decoded-faces.json").read_text()),
                )
            path = published / "train/decoded-faces.json"
            original = path.read_bytes()
            legacy = json.loads(original)
            legacy["data"].pop("crop_recipe")
            path.write_text(json.dumps(legacy))
            with patch.object(restore, "copy_resumable") as copy:
                with self.assertRaisesRegex(ValueError, "lacks its crop recipe"):
                    restore.copy_decoded_face_caches(
                        published,
                        root / "legacy",
                        data,
                        publisher,
                        verify_destination=True,
                    )
                copy.assert_not_called()
            path.write_bytes(original)
            with (published / "train/decoded-faces.npy").open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                restore.copy_decoded_face_caches(
                    published,
                    root / "corrupt",
                    data,
                    publisher,
                    verify_destination=True,
                )

    def test_child_progress_works_with_captured_stdout_and_preserves_failure(self):
        output = io.StringIO()
        with redirect_stdout(output):
            health.run_logged_command([sys.executable, "-c", "print('progress')"])
            with self.assertRaises(subprocess.CalledProcessError):
                health.run_logged_command([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertIn("progress", output.getvalue())

    def test_cpu_preparation_materializes_cache_without_training_or_zip_extraction(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/project.toml").write_text(
                (SCRIPTS.parent / "configs/project.toml").read_text()
            )
            args = SimpleNamespace(
                project_root=root,
                run_root=root / "saved-run",
                face_cache_transport=root / "face-shards",
                decoded_cache=root / "decoded",
                prepared_cache_root=None,
                publish_cache_root=None,
                screening_epochs=5,
                parts_root=None,
                arms=["binary-semantic"],
                prepare_only=True,
            )
            publisher = SimpleNamespace(
                phase=Mock(), state={"job_id": "fixture"}, finish=Mock()
            )
            with (
                patch.object(restore, "stage_verified_inputs"),
                patch.object(
                    restore, "prepare_environment", return_value=root / "python"
                ),
                patch.object(restore, "prepare_face_caches") as materialize,
                patch.object(restore, "restore_original_images") as extract,
                patch.object(restore, "execute"),
                patch.object(restore.os, "execv") as train,
            ):
                restore.restore(args, publisher)
            materialize.assert_called_once()
            extract.assert_not_called()
            train.assert_not_called()
            self.assertTrue(publisher.state["preparation_complete"])

    def test_screening_without_face_transport_cannot_fall_back_to_zip_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/project.toml").write_text(
                (SCRIPTS.parent / "configs/project.toml").read_text()
            )
            args = SimpleNamespace(
                project_root=root,
                run_root=root / "saved-run",
                face_cache_transport=None,
                prepared_cache_root=None,
                screening_epochs=5,
                parts_root=root / "zip-parts",
            )
            with (
                patch.object(restore, "stage_verified_inputs"),
                patch.object(
                    restore, "prepare_environment", return_value=root / "python"
                ),
                patch.object(restore, "restore_original_images") as extract,
            ):
                with self.assertRaisesRegex(ValueError, "no ZIP fallback"):
                    restore.restore(args, SimpleNamespace(phase=Mock()))
            extract.assert_not_called()

    def test_partial_copy_resumes_and_size_errors_remain_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.write_bytes(b"abcdefgh")
            (root / "target.copying").write_bytes(b"abc")
            restore.copy_resumable(source, target, 8)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                restore.copy_resumable(source, root / "other", 9)

    def test_launcher_uses_regular_file_and_preserves_exit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native.log"
            launcher.launch(
                [
                    sys.executable,
                    "-c",
                    "import os; assert os.path.isfile('/proc/self/fd/1'); print('native output')",
                ],
                log,
            )
            self.assertEqual(log.read_text().strip(), "native output")
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.launch([sys.executable, "-c", "raise SystemExit(7)"], log)

    def test_recent_producer_blocks_duplicate_except_same_process_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = health.TrainingHealth(root, "fixture", clock=lambda: 1000)
            with self.assertRaises(RuntimeError):
                health.check_producer_available(root, clock=lambda: 1001)
            publisher.phase("ready-for-training")
            self.assertEqual(
                health.check_producer_available(
                    root, clock=lambda: 1001, allow_handoff=True
                )["job_id"],
                "fixture",
            )
            self.assertIsNone(health.check_producer_available(root, clock=lambda: 1400))
