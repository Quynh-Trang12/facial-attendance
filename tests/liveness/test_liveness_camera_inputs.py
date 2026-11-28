"""Flat camera inputs preserve footage, labels and capture provenance."""

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from liveness_camera_inputs import count_frames, recordings, run_colab_inference  # noqa: E402 — repository path is configured before local imports.


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_script("capture_liveness_trial")
legacy = load_script("evaluation/score_liveness_trial_inventory")
batch = load_script("evaluation/score_liveness_mp4_folder")


class FlatCameraTests(unittest.TestCase):
    def capture_clip(self, directory, *, default_metadata=False):
        root = Path(directory)
        frames = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.full((32, 32, 3), 255, dtype=np.uint8),
        ]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.return_value = 20.0
        camera.read.side_effect = [(True, frame) for frame in frames]
        argv = [
            "--output-root",
            str(root / "clips"),
            "--metadata-root",
            str(root / "metadata"),
            "--subject-id",
            "person",
            "--trial-id",
            "person-20261125_120000-live-bright",
            "--kind",
            "live",
            "--illumination",
            "bright",
            "--seconds",
            "1",
        ]
        if default_metadata:
            index = argv.index("--metadata-root")
            del argv[index : index + 2]
        with (
            patch.object(capture.cv2, "VideoCapture", return_value=camera),
            patch.object(capture.time, "sleep"),
            patch.object(capture.time, "monotonic", side_effect=[0, 0, 0.2, 2, 2]),
        ):
            capture.main(argv)
        return root / "clips", (
            root / "clips/metadata" if default_metadata else root / "metadata"
        )

    def test_capture_writes_flat_mp4_and_separate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            clips, metadata = self.capture_clip(directory)
            self.assertEqual([p.suffix for p in clips.iterdir()], [".mp4"])
            clip = next(clips.iterdir())
            receipt = json.loads(next(metadata.iterdir()).read_text())
            self.assertEqual(count_frames(clip), 2)
            self.assertEqual(receipt["frames"], 2)
            self.assertEqual(receipt["video"], clip.name)
            self.assertEqual(recordings(clips)[clip.stem][1], receipt["video_sha256"])
            self.assertIn("unverified", receipt["session_source"])

    def test_default_capture_metadata_follows_custom_input_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            clips, metadata = self.capture_clip(directory, default_metadata=True)
            self.assertEqual(metadata.parent, clips)
            receipt = json.loads(next(metadata.iterdir()).read_text())
            self.assertEqual(len(recordings(clips)), 1)
            self.assertEqual(
                receipt["video_sha256"], next(iter(recordings(clips).values()))[1]
            )

    def test_legacy_scorer_view_preserves_flat_input_bytes_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            clips, _ = self.capture_clip(directory)
            clip = next(clips.iterdir())
            original = clip.read_bytes()
            with legacy.trial_metadata_view(clips) as view:
                receipt_file = next(view.glob("*/trial.json"))
                receipt = json.loads(receipt_file.read_text())
                self.assertEqual(receipt["binary_label"], 0)
                self.assertEqual(receipt["frames"], 2)
                self.assertEqual(
                    (receipt_file.parent / receipt["video"]).read_bytes(), original
                )
                self.assertNotEqual(view, clips)
            self.assertFalse(view.exists())
            self.assertEqual(clip.read_bytes(), original)
            self.assertEqual(len(list(clips.iterdir())), 1)

    def test_conflicting_filename_labels_stop_before_camera_access(self):
        with patch.object(capture.cv2, "VideoCapture") as camera:
            with self.assertRaises(SystemExit):
                capture.main(
                    [
                        "--subject-id",
                        "person",
                        "--kind",
                        "live",
                        "--illumination",
                        "bright",
                        "--trial-id",
                        "other-20261125_120000-live-bright",
                    ]
                )
            camera.assert_not_called()

    def test_legacy_cli_retains_its_temporary_input_metadata_in_the_output(self):
        with tempfile.TemporaryDirectory() as directory:
            clips, _ = self.capture_clip(directory)
            output = Path(directory) / "report"

            def score(checkpoint, view, target, **kwargs):
                self.assertEqual(len(list(view.glob("*/trial.json"))), 1)
                target.mkdir()
                return {"synthetic_fixture": True}

            argv = [
                "evaluation/score_liveness_trial_inventory.py",
                "--checkpoint",
                str(Path(directory) / "model.pt"),
                "--trials-root",
                str(clips),
                "--output-dir",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(legacy, "evaluate_trials", side_effect=score),
            ):
                legacy.main()
            receipt = json.loads((output / "recording-inputs.json").read_text())
            record = receipt["recordings"][0]
            self.assertEqual(
                hashlib.sha256(record["original_metadata_utf8"].encode()).hexdigest(),
                record["metadata_sha256"],
            )
            self.assertEqual(
                json.loads(record["original_metadata_utf8"])["source_recording"],
                str(next(clips.iterdir()).resolve()),
            )

    def test_previous_camera_summary_is_preserved_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            original = b'{"historical": true}\n'
            latest = cache / "latest.json"
            latest.write_bytes(original)
            batch.preserve_previous_summary(cache)
            batch.preserve_previous_summary(cache)
            snapshots = list((cache / "summaries").glob("*.json"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].read_bytes(), original)
            self.assertEqual(latest.read_bytes(), original)

    def test_existing_recording_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person-20261125_120000-live-bright.mp4"
            path.write_bytes(b"original footage")
            with patch.object(capture.cv2, "VideoCapture") as camera:
                with self.assertRaises(FileExistsError):
                    capture.main(
                        [
                            "--output-root",
                            directory,
                            "--subject-id",
                            "person",
                            "--kind",
                            "live",
                            "--illumination",
                            "bright",
                            "--trial-id",
                            path.stem,
                        ]
                    )
                camera.assert_not_called()
            self.assertEqual(path.read_bytes(), b"original footage")

    def test_colab_camera_orchestration_preserves_private_outputs_and_interpreter(self):
        with tempfile.TemporaryDirectory(prefix="camera path's ") as temporary:
            root = Path(temporary)
            shared, runtime, private = (
                root / "shared",
                root / "runtime",
                root / "private",
            )
            shared.mkdir()
            source = shared / "src/model.py"
            source.parent.mkdir()
            source.write_text("# source fixture")
            release = {
                "files": {
                    "src/model.py": {
                        "delivery_path": "src/model.py",
                        "sha256": "fixture",
                    }
                }
            }
            health = Mock()
            workflow = SimpleNamespace(
                TrainingHealth=Mock(return_value=health),
                stage_verified_inputs=Mock(),
                prepare_environment=Mock(return_value="project-python"),
            )
            with patch("subprocess.run") as launch:
                python = run_colab_inference(
                    shared,
                    runtime,
                    shared / "model.pt",
                    "checksum",
                    shared / "clips",
                    private,
                    release,
                    shared / "batch.py",
                    workflow,
                )
                command = launch.call_args.args[0]
                self.assertEqual(python, "project-python")
                self.assertEqual(command[0], python)
                self.assertEqual(
                    Path(command[command.index("--cache") + 1]),
                    private / "camera-evaluation/checksum",
                )
                self.assertEqual(
                    Path(command[command.index("--checkpoint") + 1]),
                    shared / "model.pt",
                )
                args = workflow.stage_verified_inputs.call_args.args[0]
                self.assertEqual(args.source_root, shared)
                self.assertEqual(args.project_root, runtime)
                self.assertEqual(source.read_text(), "# source fixture")
            workflow.stage_verified_inputs.side_effect = ValueError("source changed")
            workflow.prepare_environment.reset_mock()
            with patch("subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    run_colab_inference(
                        shared,
                        runtime,
                        shared / "model.pt",
                        "checksum",
                        shared / "clips",
                        private,
                        release,
                        shared / "batch.py",
                        workflow,
                    )
                launch.assert_not_called()
                workflow.prepare_environment.assert_not_called()

    def test_cpu_camera_predictions_cannot_reuse_cuda_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, checkpoint, detector = (
                root / name for name in ("clip.mp4", "model.pt", "detector.onnx")
            )
            for path in (video, checkpoint, detector):
                path.write_bytes(path.name.encode())
            trial = "person-20261125_120000-live-bright"
            checksum = batch.file_sha256(video)
            clips = {trial: (video, checksum, "person", "live")}
            args = SimpleNamespace(
                checkpoint=checkpoint,
                detector=detector,
                cache=root / "cache",
                device="cuda",
            )
            adapter = Mock()
            adapter.predictor.meta = {"threshold": 0.5}
            result = dict(trial_id=trial, recording_sha256=checksum, kind="live")
            with (
                patch.object(batch, "count_frames", return_value=1),
                patch.object(batch, "LivenessAdapter", return_value=adapter) as factory,
                patch.object(
                    batch.evaluator, "score_recording", return_value=(result, [])
                ),
            ):
                batch.evaluate_batch(args, clips, [])
                cuda = json.loads((args.cache / "latest.json").read_text())
                args.device = "cpu"
                batch.evaluate_batch(args, clips, [])
                factory.assert_called_with(checkpoint, "cpu", detector_model=detector)
                cpu = json.loads((args.cache / "latest.json").read_text())
                self.assertEqual(cpu["inferred"], 1)
                self.assertNotEqual(cpu["bindings"], cuda["bindings"])
                self.assertEqual(cpu["bindings"]["device"], "cpu")
                self.assertNotIn("binary_training.py", cpu["bindings"]["source"])
                self.assertNotIn("audit/duplicate_scan.py", cpu["bindings"]["source"])
                batch.evaluate_batch(args, clips, [])
                self.assertEqual(
                    json.loads((args.cache / "latest.json").read_text())["reused"], 1
                )

    def test_historical_source_override_requires_the_approved_current_scorer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("clip.mp4", "model.pt", "detector.onnx"):
                (root / name).write_bytes(name.encode())
            trial = "person-20261125_120000-live-bright"
            checksum = batch.file_sha256(root / "clip.mp4")
            clips = {trial: (root / "clip.mp4", checksum, "person", "live")}
            args = SimpleNamespace(
                checkpoint=root / "model.pt",
                detector=root / "detector.onnx",
                cache=root / "cache",
            )
            adapter = Mock()
            adapter.predictor.meta = {"threshold": 0.5}
            result = dict(trial_id=trial, recording_sha256=checksum, kind="live")
            with (
                patch.object(batch, "count_frames", return_value=1),
                patch.object(batch, "LivenessAdapter", return_value=adapter),
                patch.object(
                    batch.evaluator, "score_recording", return_value=(result, [])
                ),
            ):
                batch.evaluate_batch(args, clips, [])
            binding = json.loads((args.cache / "latest.json").read_text())["bindings"]
            compatible = [
                dict(
                    batch=binding["batch"],
                    source=binding["source"],
                    input_helper=binding["input_helper"],
                    accepted_input_helper_sha256=binding["input_helper"],
                    accepted_batch_sha256=binding["batch"],
                    accepted_source_sha256=binding["source"],
                )
            ]
            original_hash = batch.file_sha256

            def changed(path):
                return (
                    "changed-model-implementation"
                    if Path(path).name == "binary_model.py"
                    else original_hash(path)
                )

            with (
                patch.object(batch, "file_sha256", side_effect=changed),
                patch.object(batch, "count_frames", return_value=1),
                patch.object(
                    batch,
                    "LivenessAdapter",
                    side_effect=RuntimeError("New source requires inference"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "New source requires inference"
                ):
                    batch.evaluate_batch(args, clips, compatible)

    def test_legacy_score_reuse_requires_the_reviewed_input_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, checkpoint, detector = (
                root / name for name in ("clip.mp4", "model.pt", "detector.onnx")
            )
            for path in (video, checkpoint, detector):
                path.write_bytes(path.name.encode())
            digest = batch.file_sha256(video)
            identity = "person-20261125_120000-live-bright"
            clips = {identity: (video, digest, "person", "live")}
            args = SimpleNamespace(
                checkpoint=checkpoint, detector=detector, cache=root / "scores"
            )
            args.cache.mkdir()
            adapter = Mock()
            adapter.predictor.meta = {"threshold": 0.5}
            result = {"trial_id": identity, "recording_sha256": digest, "kind": "live"}
            with (
                patch.object(batch, "count_frames", return_value=4),
                patch.object(batch, "LivenessAdapter", return_value=adapter),
                patch.object(
                    batch.evaluator, "score_recording", return_value=(result, [])
                ),
            ):
                batch.evaluate_batch(args, clips, [])
            bindings = json.loads((args.cache / "latest.json").read_text())["bindings"]
            current_key = batch.cache_key(bindings)
            historical = {
                key: value for key, value in bindings.items() if key != "input_helper"
            }
            historical["batch"] = "previous-batch"
            historical["source"] = {
                name.removeprefix("models/"): value
                for name, value in bindings["source"].items()
            }
            historical_key = batch.cache_key(historical)
            previous = args.cache / historical_key
            previous.mkdir()
            receipt = next((args.cache / current_key).glob("*.json"))
            record = json.loads(receipt.read_text())
            record["key"] = historical_key
            (previous / receipt.name).write_text(json.dumps(record))
            shutil.rmtree(args.cache / current_key)
            compatible = [
                {
                    "batch": "previous-batch",
                    "source": historical["source"],
                    "complete_source_bindings": True,
                    "input_helper": None,
                    "accepted_input_helper_sha256": bindings["input_helper"],
                    "accepted_source_sha256": bindings["source"],
                    "accepted_batch_sha256": bindings["batch"],
                }
            ]
            with patch.object(
                batch,
                "LivenessAdapter",
                side_effect=AssertionError("Expected cache reuse"),
            ):
                batch.evaluate_batch(args, clips, compatible)
            self.assertEqual(
                json.loads((args.cache / "latest.json").read_text())["reused"], 1
            )
            original_hash = batch.file_sha256

            def changed_helper(path):
                return (
                    "changed-helper"
                    if Path(path).name == "liveness_camera_inputs.py"
                    else original_hash(path)
                )

            with (
                patch.object(batch, "file_sha256", side_effect=changed_helper),
                patch.object(batch, "count_frames", return_value=4),
                patch.object(
                    batch,
                    "LivenessAdapter",
                    side_effect=RuntimeError("New input rules require a new score"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "New input rules"):
                    batch.evaluate_batch(args, clips, compatible)


if __name__ == "__main__":
    unittest.main()
