"""Frame integration and physical-trial accounting, using controlled fixtures."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

FULL = all(
    importlib.util.find_spec(name)
    for name in ("torch", "torchvision", "cv2", "numpy", "sklearn")
)
if FULL:
    import cv2
    import numpy as np

    from facial_attendance.liveness.dataset_files import file_sha256
    from facial_attendance.liveness.evaluation.presentation_evaluation import (
        evaluate_trials,
    )
    from facial_attendance.liveness.inference.liveness_adapter import LivenessAdapter
    from facial_attendance.liveness.inference.temporal_liveness import (
        TemporalLivenessConfig,
        TemporalLivenessGate,
    )
    from facial_attendance.liveness.models.face_detector import (
        DEFAULT_MODEL,
        YuNetFaceDetector,
    )


@unittest.skipUnless(FULL, "Requires the existing project runtime")
class FrameIntegrationTests(unittest.TestCase):
    def test_shared_cli_runs_cpu_checkpoint_without_training_imports(self):
        from facial_attendance.liveness.models.binary_model import (
            ARCHITECTURE,
            INPUT_SIZE,
            MEAN,
            STD,
            build_model,
            save_checkpoint,
            state_version,
        )

        with tempfile.TemporaryDirectory(prefix="liveness CLI's ") as temporary:
            root = Path(temporary)
            model = build_model().eval()
            meta = dict(
                architecture=ARCHITECTURE,
                input_size=INPUT_SIZE,
                mean=list(MEAN),
                std=list(STD),
                label_mapping={"live": 0, "spoof": 1},
                model_id="synthetic-CLI-fixture",
                model_version=state_version(model.state_dict()),
                optimizer_updates=1,
                threshold=0.5,
                threshold_source="validation EER operating point",
            )
            checkpoint = root / "fixture.pt"
            save_checkpoint(checkpoint, model.state_dict(), meta)
            from PIL import Image

            image = root / "face.png"
            Image.new("RGB", (80, 64), (40, 80, 120)).save(image)
            output = root / "prediction.json"
            project = Path(__file__).resolve().parents[2]
            command = [
                sys.executable,
                str(project / "scripts/run_liveness_notebook.py"),
                "--action",
                "inference",
                "--device",
                "cpu",
                "--face-crop",
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(image),
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command,
                cwd=project,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                capture_output=True,
                text=True,
                check=True,
            )
            prediction = json.loads(result.stdout)
            self.assertEqual(prediction, json.loads(output.read_text()))
            self.assertEqual(prediction["precision"], "cpu_float32")
            self.assertEqual(prediction["model_version"], meta["model_version"])
            self.assertEqual(prediction["threshold"], 0.5)
            self.assertEqual(
                prediction["allow_attendance"], prediction["spoof_score"] < 0.5
            )
            before = output.read_bytes()
            repeated = subprocess.run(
                command, cwd=project, capture_output=True, text=True
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            if DEFAULT_MODEL.is_file():
                # Exercise the actual detector boundary on a non-face image,
                # in addition to the upstream-crop route above.
                frame_command = [arg for arg in command if arg != "--face-crop"]
                frame_output = root / "frame-prediction.json"
                frame_command[-1] = str(frame_output)
                full_image = subprocess.run(
                    frame_command,
                    cwd=project,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                frame_prediction = json.loads(full_image.stdout)
                self.assertEqual(frame_prediction, json.loads(frame_output.read_text()))
                self.assertEqual(frame_prediction["status"], "uncertain")
                self.assertFalse(frame_prediction["allow_attendance"])

            # A fresh import proves the runtime boundary, independent of test import order.
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; import facial_attendance.liveness.evaluation.presentation_evaluation; "
                    "assert 'facial_attendance.liveness.experiments.binary_experiment' not in sys.modules; "
                    "assert 'facial_attendance.liveness.experiments.multitask_experiment' not in sys.modules",
                ],
                cwd=project,
                env={**os.environ, "PYTHONPATH": str(project / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_explicit_unavailable_cuda_never_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                LivenessAdapter("must-not-open.pt", device="cuda")

    def test_missing_precision_is_fp32_and_explicit_half_uses_autocast(self):
        for metadata, half in (
            ({}, False),
            ({"inference_precision": "float32"}, False),
            ({"inference_precision": "float16"}, True),
        ):
            adapter = LivenessAdapter.__new__(LivenessAdapter)
            adapter.predictor = SimpleNamespace(
                device=SimpleNamespace(type="cuda"),
                meta=metadata,
                predict_face=lambda image: {"spoof_score": 0.499999},
            )
            with patch("torch.autocast", return_value=nullcontext()) as autocast:
                result = adapter.predict_face(None)
            self.assertEqual(autocast.called, half)
            self.assertEqual(
                result["precision"], "cuda_float16_autocast" if half else "cuda_float32"
            )

    def test_adapter_accepts_temporal_policy_for_slower_frames(self):
        custom = TemporalLivenessConfig(3, 1.5, 1.0)
        for config, expected in ((TemporalLivenessConfig(), False), (custom, True)):
            with patch(
                "facial_attendance.liveness.inference.liveness_adapter.LivenessPredictor"
            ):
                adapter = LivenessAdapter(
                    "fixture", device="cpu", temporal_config=config
                )
            adapter.predict_frame = lambda frame: {
                "status": "live",
                "allow_attendance": True,
                "spoof_score": 0.1,
                "threshold": 0.5,
            }
            for timestamp in (0, 0.75, 1.5):
                result = adapter.predict_stream_frame(
                    None, stream_id="person", timestamp=timestamp
                )
            self.assertEqual(result["allow_attendance"], expected)
            self.assertEqual(
                result["temporal_policy"]["maximum_gap_seconds"],
                config.maximum_gap_seconds,
            )

    def test_installed_opencv_detector_api_when_asset_is_available(self):
        if not DEFAULT_MODEL.exists():
            self.skipTest("Optional YuNet asset not prepared")
        detector = YuNetFaceDetector()
        self.assertEqual(detector.detect(np.zeros((320, 320, 3), dtype=np.uint8)), [])

    def adapter(self, boxes):
        adapter = LivenessAdapter.__new__(LivenessAdapter)
        adapter.brightness_factor, adapter.crop_margin = 1.0, 0.0
        adapter.detector = SimpleNamespace(detect=lambda frame: boxes)
        adapter.predictor = SimpleNamespace(
            meta={"model_id": "fixture", "model_version": "fixture", "threshold": 0.5}
        )
        return adapter

    def test_no_face_and_multiple_faces_cannot_allow_attendance(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for boxes, reason in (
            ([], "no_face"),
            ([(0, 0, 60, 60), (70, 0, 60, 60)], "multiple_faces"),
        ):
            result = self.adapter(boxes).predict_frame(frame)
            self.assertEqual(result["status"], "uncertain")
            self.assertFalse(result["allow_attendance"])
            self.assertIsNone(result["spoof_score"])
            self.assertEqual(result["reason"], reason)
            self.assertEqual(result["face_box_format"], "source_pixels_xywh")
            self.assertEqual(result["threshold"], 0.5)

    def test_frame_crop_uses_rgb_and_expected_box(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:] = [10, 20, 30]
        adapter = self.adapter([(5, 7, 60, 70)])
        seen = []

        def face(image):
            seen.append((image.mode, image.size, image.getpixel((0, 0))))
            return {"status": "spoof", "allow_attendance": False, "spoof_score": 0.9}

        adapter.predict_face = face
        result = adapter.predict_frame(frame)
        self.assertEqual(result["face_box"], [5, 7, 60, 70])
        self.assertIsNone(result["reason"])
        self.assertEqual(seen, [("RGB", (60, 70), (30, 20, 10))])
        with self.assertRaises(ValueError):
            adapter.predict_frame(frame.astype(float))

    def test_expanded_crop_brightness_and_frame_boundary(self):
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        adapter = self.adapter([(0, 0, 50, 50)])
        adapter.brightness_factor, adapter.crop_margin = 1.25, 0.06
        seen = []
        adapter.predict_face = lambda image: (
            seen.append((image.size, image.getpixel((0, 0)))) or {}
        )
        result = adapter.predict_frame(frame)
        self.assertEqual(result["face_box"], [0, 0, 53, 53])
        self.assertEqual(seen, [((53, 53), (125, 125, 125))])

    def test_detector_maps_resized_boxes_to_original_frame_crop(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[60:300, 30:330] = [10, 20, 30]
        seen = []
        detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
        detector.engine = SimpleNamespace(
            setInputSize=lambda size: seen.append(size),
            detect=lambda image: (
                seen.append(image.shape),
                np.array([[10, 20, 100, 80]], dtype=float),
            ),
        )
        adapter = self.adapter([])
        adapter.detector = detector
        crops = []

        def predict(image):
            crops.append(
                (image.size, image.getpixel((0, 0)), image.getpixel((299, 239)))
            )
            return {"status": "live", "allow_attendance": True, "spoof_score": 0.1}

        adapter.predict_face = predict
        result = adapter.predict_frame(frame)
        self.assertEqual(seen, [(640, 360), (360, 640, 3)])
        self.assertEqual(result["face_box"], [30, 60, 300, 240])
        self.assertEqual(crops, [((300, 240), (30, 20, 10), (30, 20, 10))])

    def test_detector_maps_rounded_portrait_size_and_clips_source_bounds(self):
        frame = np.zeros((1001, 751, 3), dtype=np.uint8)
        detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
        seen = []
        detector.engine = SimpleNamespace(
            setInputSize=lambda size: seen.append(size),
            detect=lambda image: (
                None,
                np.array(
                    [[-1, -2, 482, 644], [48, 64, 48, 64], [500, 700, 20, 20]],
                    dtype=float,
                ),
            ),
        )
        boxes = detector.detect(frame)
        self.assertEqual(seen, [(480, 640)])
        self.assertEqual(boxes, [(0, 0, 751, 1001), (75, 100, 76, 101)])

    def test_small_input_is_not_upscaled_and_empty_detection_stays_empty(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
        seen = []

        def detect(image):
            self.assertIs(image, frame)
            return None, None

        detector.engine = SimpleNamespace(
            setInputSize=lambda size: seen.append(size), detect=detect
        )
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(seen, [(160, 120)])

    def test_missing_recordings_are_not_fabricated_physical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "No operator-recorded physical trials"
            ):
                evaluate_trials(
                    Path(directory) / "model.pt", directory, Path(directory) / "out"
                )

    def test_uncertain_presentations_are_not_reported_as_classifier_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "synthetic.pt"
            checkpoint.write_bytes(b"SYNTHETIC fixture, not model weights")
            for kind in ("live", "screen"):
                trial = root / "trials" / kind
                trial.mkdir(parents=True)
                video = trial / "capture.avi"
                writer = cv2.VideoWriter(
                    str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96)
                )
                self.assertTrue(writer.isOpened())
                for _ in range(8):
                    writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
                writer.release()
                (trial / "trial.json").write_text(
                    json.dumps(
                        {
                            "schema": "physical-liveness-trial-v1",
                            "trial_id": kind,
                            "kind": kind,
                            "binary_label": int(kind != "live"),
                            "frames": 8,
                            "video": "capture.avi",
                            "video_sha256": file_sha256(video),
                            "subject_id": "fixture",
                            "session_id": "fixture",
                        }
                    )
                )
            adapter = SimpleNamespace(
                predictor=SimpleNamespace(
                    meta={"threshold": 0.5, "model_version": "fixture"}
                ),
                predict_frame=lambda frame: {
                    "status": "uncertain",
                    "spoof_score": None,
                    "allow_attendance": False,
                },
            )
            with patch(
                "facial_attendance.liveness.evaluation.presentation_evaluation.LivenessAdapter",
                return_value=adapter,
            ):
                result = evaluate_trials(checkpoint, root / "trials", root / "result")
            self.assertIsNone(result["presentation_metrics_on_scored_trials"])
            self.assertEqual(result["uncertain_presentations"], 2)
            self.assertEqual(
                result["operational_errors_including_uncertain"]["live_not_accepted"], 1
            )
            self.assertTrue(
                all(
                    r["sampled_frames"] == 8 and r["scored_frames"] == 0
                    for r in result["trials"]
                )
            )
            self.assertTrue(
                all(
                    r["uncertain"] == 8 and r["live"] == 0 and r["spoof"] == 0
                    for r in result["frame_prediction_counts"]
                )
            )

    def test_clip_median_does_not_hide_mixed_frame_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "synthetic.pt"
            checkpoint.write_bytes(b"Synthetic fixture")
            trial = root / "trials/live01"
            trial.mkdir(parents=True)
            video = trial / "capture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
            writer.release()
            (trial / "trial.json").write_text(
                json.dumps(
                    {
                        "schema": "physical-liveness-trial-v1",
                        "trial_id": "live01",
                        "kind": "live",
                        "binary_label": 0,
                        "frames": 8,
                        "video": "capture.avi",
                        "video_sha256": file_sha256(video),
                        "subject_id": "fixture",
                        "session_id": "fixture",
                    }
                )
            )
            values = iter([0.1] * 4 + [0.9] * 3 + [None])

            def predict(frame):
                value = next(values)
                return {
                    "status": "uncertain"
                    if value is None
                    else "spoof"
                    if value >= 0.5
                    else "live",
                    "spoof_score": value,
                    "allow_attendance": value is not None and value < 0.5,
                }

            adapter = SimpleNamespace(
                predictor=SimpleNamespace(
                    meta={"threshold": 0.5, "model_version": "fixture"}
                ),
                predict_frame=predict,
            )
            with patch(
                "facial_attendance.liveness.evaluation.presentation_evaluation.LivenessAdapter",
                return_value=adapter,
            ):
                result = evaluate_trials(checkpoint, root / "trials", root / "result")
            self.assertEqual(result["trials"][0]["status"], "live")
            self.assertEqual(
                result["frame_prediction_counts"],
                [{"trial_id": "live01", "live": 4, "spoof": 3, "uncertain": 1}],
            )


@unittest.skipUnless(FULL, "Requires the project ML runtime")
class TemporalAdmissionTests(unittest.TestCase):
    def live(self):
        return {
            "status": "live",
            "allow_attendance": True,
            "spoof_score": 0.01,
            "threshold": 0.2,
        }

    def test_live_needs_frames_and_elapsed_time_and_resets_on_uncertain(self):
        gate = TemporalLivenessGate()
        for index in range(4):
            self.assertFalse(
                gate.observe(self.live(), stream_id="P1", timestamp=index * 0.25)[
                    "allow_attendance"
                ]
            )
        self.assertTrue(
            gate.observe(self.live(), stream_id="P1", timestamp=1.0)["allow_attendance"]
        )
        uncertain = {
            "status": "uncertain",
            "allow_attendance": False,
            "spoof_score": None,
            "threshold": 0.2,
        }
        self.assertFalse(
            gate.observe(uncertain, stream_id="P1", timestamp=1.1)["allow_attendance"]
        )
        self.assertFalse(
            gate.observe(self.live(), stream_id="P1", timestamp=1.2)["allow_attendance"]
        )

    def test_track_changes_gaps_and_clock_reversal_cannot_inherit_acceptance(self):
        gate = TemporalLivenessGate()
        for index in range(5):
            gate.observe(self.live(), stream_id="P1", timestamp=index * 0.25)
        self.assertFalse(
            gate.observe(self.live(), stream_id="P2", timestamp=1.1)["allow_attendance"]
        )
        self.assertFalse(
            gate.observe(self.live(), stream_id="P2", timestamp=3.0)["allow_attendance"]
        )
        with self.assertRaises(ValueError):
            gate.observe(self.live(), stream_id="P2", timestamp=2.0)
        spoof = {
            "status": "spoof",
            "allow_attendance": False,
            "spoof_score": 0.9,
            "threshold": 0.2,
        }
        self.assertEqual(
            gate.observe(spoof, stream_id="P2", timestamp=3.1)["status"], "spoof"
        )


if __name__ == "__main__":
    unittest.main()
