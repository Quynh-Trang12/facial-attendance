"""Behavior contracts for liveness public test workflow."""

import hashlib
import importlib.util
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


sys.path.insert(0, str(SCRIPTS))

from liveness_training_health import InferenceProgress  # noqa: E402 — repository path is configured before local imports.

spec = importlib.util.spec_from_file_location(
    "evaluation_runner", SCRIPTS / "run_liveness_public_test.py"
)


runner = importlib.util.module_from_spec(spec)


spec.loader.exec_module(runner)


class EvaluationTests(unittest.TestCase):
    def test_commenter_results_do_not_modify_shared_checkpoint_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, private = root / "shared", root / "personal"
            checkpoint = shared / "aenet-full/liveness-inference.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic frozen checkpoint")
            checksum = runner.digest(checkpoint)
            choice = shared / "model-choice.json"
            choice.write_text(
                json.dumps(
                    {
                        "chosen": "aenet-full",
                        "arms": {"aenet-full": {"checkpoint_sha256": checksum}},
                    }
                )
            )
            original = {p: p.read_bytes() for p in shared.rglob("*") if p.is_file()}

            def score(command, *args, **kwargs):
                output = Path(command[command.index("--output-dir") + 1])
                self.assertTrue(output.is_relative_to(private))
                output.mkdir(parents=True)
                (output / "evaluation.json").write_text(
                    json.dumps({"checkpoint_sha256": checksum})
                )

            with (
                patch.object(
                    runner, "evaluation_input_signatures", return_value=["signature"]
                ),
                patch.object(runner, "observe_active_inference"),
                patch.object(
                    runner,
                    "prepare",
                    return_value=(
                        "python",
                        root / "cache",
                        "aenet-full",
                        checkpoint,
                        checksum,
                    ),
                ),
                patch.object(runner, "launch", side_effect=score),
            ):
                output = runner.evaluate_frozen(
                    root,
                    root / "runtime",
                    shared,
                    lambda text: None,
                    output_root=private,
                )
            self.assertTrue((output / "evaluation-results.json").is_file())
            self.assertFalse((shared / "evaluation").exists())
            self.assertEqual(
                original, {p: p.read_bytes() for p in shared.rglob("*") if p.is_file()}
            )

    def test_notebook_diagnostics_use_private_output_even_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, private = root / "shared", root / "personal"
            display_module = ModuleType("IPython.display")
            display_module.Markdown = str
            display_module.display = Mock(return_value=SimpleNamespace(update=Mock()))
            with (
                patch.dict(
                    sys.modules,
                    {
                        "IPython": ModuleType("IPython"),
                        "IPython.display": display_module,
                    },
                ),
                patch.object(
                    runner, "evaluate", side_effect=ValueError("fixture failure")
                ),
                self.assertRaisesRegex(RuntimeError, "Evaluation failed"),
            ):
                runner.run_notebook(root, root / "runtime", shared, output_root=private)
            self.assertTrue((private / "evaluation/input-preparation.log").is_file())
            self.assertFalse(shared.exists())

    def test_selection_uses_checksum_not_a_historical_host_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "aenet-full/liveness-inference.pt"
            path.parent.mkdir()
            path.write_bytes(b"frozen checkpoint")
            checksum = runner.digest(path)
            (root / "model-choice.json").write_text(
                json.dumps(
                    {
                        "chosen": "aenet-full",
                        "directories": {"aenet-full": "/another-host/output"},
                        "arms": {"aenet-full": {"checkpoint_sha256": checksum}},
                    }
                )
            )
            self.assertEqual(
                runner.selected_checkpoint(root), ("aenet-full", path, checksum)
            )
            path.write_bytes(b"changed checkpoint")
            with self.assertRaises(ValueError):
                runner.selected_checkpoint(root)

    @unittest.skipUnless(os.name == "posix", "POSIX/Colab evaluation staging")
    def test_changed_split_is_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, project = root / "source", root / "runtime"
            (source / runner.CACHE).mkdir(parents=True)
            for name, data in (
                (runner.INVENTORY, b"inventory"),
                (runner.ANNOTATIONS, b'{"image":[]}'),
            ):
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            receipt = {
                "schema": "liveness-decoded-faces-v1",
                "dtype": "uint8",
                "shape": [1, 224, 224, 3],
                "data": {"annotations_sha256": "wrong"},
            }
            (source / runner.CACHE / "decoded-faces.json").write_text(
                json.dumps(receipt)
            )
            with patch.object(runner, "rclone_cache_download") as download:
                with self.assertRaises(ValueError):
                    runner.stage_test_cache(source, project, lambda message: None)
                download.assert_not_called()

    def test_staging_rejects_source_directory_alias_before_rebinding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, project = root / "source", root / "runtime"
            (source / runner.CACHE).mkdir(parents=True)
            (project / runner.CACHE).parent.mkdir(parents=True)
            try:
                (project / runner.CACHE).symlink_to(
                    source / runner.CACHE, target_is_directory=True
                )
            except OSError:
                self.skipTest("Directory symlinks are unavailable on this host")
            with self.assertRaisesRegex(ValueError, "directories must differ"):
                runner.stage_test_cache(source, project, lambda message: None)

    def test_short_final_batch_preserves_the_reported_sample_count(self):
        output = io.StringIO()
        with redirect_stdout(output):
            progress = InferenceProgress(3, 2)
            progress()
            progress()
        event = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(event["samples"], 3)
        self.assertEqual(event["total_samples"], 3)


class EvaluationReuseTests(unittest.TestCase):
    def test_dataset_evaluator_rename_preserves_saved_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            script = source / "scripts/evaluation/score_liveness_image_split.py"
            script.parent.mkdir(parents=True)
            script.write_text("frozen evaluator")
            receipt = source / runner.CACHE / "decoded-faces.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}")
            release = {
                "files": {
                    "scripts/evaluation/score_liveness_image_split.py": {
                        "delivery_path": "scripts/evaluation/score_liveness_image_split.py",
                        "sha256": runner.digest(script),
                    }
                },
                "evaluation": {
                    "test_cache_receipt": {"sha256": runner.digest(receipt)}
                },
            }
            config = source / "configs/liveness-release.json"
            config.parent.mkdir()
            config.write_text(json.dumps(release))
            historical = {
                "scripts/evaluate_liveness.py": runner.digest(script),
                "test-cache-receipt": runner.digest(receipt),
            }
            expected = hashlib.sha256(
                json.dumps(historical, sort_keys=True).encode()
            ).hexdigest()
            self.assertIn(expected, runner.evaluation_input_signatures(source))
            script.write_text("changed evaluator")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.evaluation_input_signatures(source)

    @unittest.skipUnless(os.name == "posix", "POSIX/Colab evaluation staging")
    def test_complete_results_skip_environment_and_pixel_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            run = root / "run"
            (source / "configs").mkdir(parents=True)
            receipt = source / runner.CACHE / "decoded-faces.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}")
            release = {
                "files": {"data/liveness-trials/missing.mp4": {"sha256": "legacy"}},
                "evaluation": {
                    "test_cache_receipt": {"sha256": runner.digest(receipt)}
                },
            }
            (source / "configs/liveness-release.json").write_text(json.dumps(release))
            checkpoint = run / "aenet-full/liveness-inference.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen")
            checksum = runner.digest(checkpoint)
            (run / "model-choice.json").write_text(
                json.dumps(
                    {
                        "chosen": "aenet-full",
                        "arms": {"aenet-full": {"checkpoint_sha256": checksum}},
                    }
                )
            )
            output = run / "evaluation/aenet-full" / checksum[:16]
            path = output / "public-test/evaluation.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"checkpoint_sha256": checksum}))
            signature, *_, legacy_signature = runner.evaluation_input_signatures(source)
            self.assertNotEqual(signature, legacy_signature)
            # A folder move must not invalidate an otherwise compatible saved result.
            release["files"] = {
                "data/liveness/camera-trials/missing.mp4": {"sha256": "legacy"}
            }
            (source / "configs/liveness-release.json").write_text(json.dumps(release))
            self.assertEqual(
                runner.evaluation_input_signatures(source)[-1], legacy_signature
            )
            release["files"] = {}
            release["evaluation"]["legacy_camera_evaluation_bindings"] = {
                "data/liveness-trials/missing.mp4": "legacy"
            }
            (source / "configs/liveness-release.json").write_text(json.dumps(release))
            self.assertEqual(
                runner.evaluation_input_signatures(source)[-1], legacy_signature
            )
            index = {
                "checkpoint_sha256": checksum,
                "input_signature": legacy_signature,
                "results": {
                    "public-test": "public-test",
                    "physical-trials": "physical-trials",
                },
            }
            index_path = output / "evaluation-results.json"
            index_path.write_text(json.dumps(index))
            with patch.object(
                runner,
                "prepare",
                side_effect=AssertionError("Must reuse saved results"),
            ):
                self.assertEqual(
                    runner.evaluate(
                        source, root / "runtime", run, lambda message: None
                    ),
                    output,
                )
            self.assertEqual(json.loads(index_path.read_text()), index)

    @unittest.skipUnless(os.name == "posix", "POSIX/Colab evaluation staging")
    def test_fresh_evaluation_runs_only_the_public_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, project, run = (
                root / name for name in ("source", "runtime", "run")
            )
            checkpoint = run / "model.pt"
            checksum = "a" * 64
            output = run / "evaluation" / "aenet-full" / checksum[:16]
            camera = run / "camera-evaluation" / checksum / "latest.json"
            camera.parent.mkdir(parents=True)
            camera.write_text("saved camera results")

            def publish(command, log, **kwargs):
                self.assertEqual(
                    Path(command[2]).parts[-2:],
                    ("evaluation", "score_liveness_image_split.py"),
                )
                result = (
                    Path(command[command.index("--output-dir") + 1]) / "evaluation.json"
                )
                result.parent.mkdir()
                result.write_text(json.dumps({"checkpoint_sha256": checksum}))

            with (
                patch.object(
                    runner,
                    "evaluation_input_signatures",
                    return_value=("public", "previous", "legacy"),
                ),
                patch.object(
                    runner,
                    "selected_checkpoint",
                    return_value=("aenet-full", checkpoint, checksum),
                ),
                patch.object(
                    runner,
                    "prepare",
                    return_value=(
                        "python",
                        project / "cache",
                        "aenet-full",
                        checkpoint,
                        checksum,
                    ),
                ),
                patch.object(runner, "launch", side_effect=publish) as launch,
            ):
                self.assertEqual(
                    runner.evaluate(source, project, run, lambda message: None), output
                )
                launch.assert_called_once()

            index = json.loads((output / "evaluation-results.json").read_text())
            self.assertEqual(index["results"], {"public-test": "public-test"})
            self.assertEqual(camera.read_text(), "saved camera results")


class CameraInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS.parent / "src"))
        cls.recordings = staticmethod(
            runpy.run_path(str(SCRIPTS / "evaluation/score_liveness_mp4_folder.py"))[
                "recordings"
            ]
        )

    def test_accepts_root_clips_only_and_normalizes_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (root / "First-20261122-122330-live-bright.mp4").write_bytes(b"first")
            (nested / "missing-timestamp.mp4").write_bytes(b"ignored")
            self.assertEqual(
                set(self.recordings(root)),
                {
                    "First-20261122_122330-live-bright",
                },
            )

    def test_reports_missing_and_impossible_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = ("Subject-live-bright.mp4", "Subject-20260230_122330-live-dim.mp4")
            for name in bad:
                (root / name).touch()
            with self.assertRaises(ValueError) as failure:
                self.recordings(root)
            for name in bad:
                self.assertIn(name, str(failure.exception))

    def test_rejects_duplicate_id_and_duplicate_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Subject-20261122-122330-live-bright.mp4").write_bytes(b"one")
            duplicate = root / "Subject-20261122_122330-live-bright.mp4"
            duplicate.write_bytes(b"another")
            with self.assertRaisesRegex(ValueError, "Duplicate trial ID"):
                self.recordings(root)
            duplicate.rename(
                root / "Other-20261122_122330-live-bright.mp4"
            ).write_bytes(b"one")
            with self.assertRaisesRegex(ValueError, "Duplicate video content"):
                self.recordings(root)


class EvaluationSourceTests(unittest.TestCase):
    def test_source_bootstrap_accepts_matching_files_and_rejects_changed_digest(self):
        repository = Path(__file__).resolve().parents[2]
        source = repository / "src/facial_attendance/liveness"
        hashes = {
            path.relative_to(source).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source.rglob("*.py")
        }
        with tempfile.TemporaryDirectory() as temporary:
            from facial_attendance.liveness.dataset_files import snapshot_sources

            snapshot = Path(temporary) / "source"
            snapshot_sources(snapshot, hashes)
            checkpoint = Path(temporary) / "metadata.pt"
            torch.save({"meta": {"source_sha256": hashes}}, checkpoint)
            command = [
                sys.executable,
                str(repository / "scripts/evaluation/score_liveness_image_split.py"),
                "--checkpoint",
                str(checkpoint),
                "--source-snapshot",
                str(snapshot),
                "--help",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            hashes["datasets/source_dataset.py"] = "0" * 64
            torch.save({"meta": {"source_sha256": hashes}}, checkpoint)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Producing source checksum mismatch: datasets/source_dataset.py",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
