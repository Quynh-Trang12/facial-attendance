"""Exercise bundled inference through public entry points without training assets."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = all(
    importlib.util.find_spec(name) for name in ("torch", "torchvision", "cv2", "numpy")
)


@unittest.skipUnless(RUNTIME, "Requires the shared inference runtime")
class SubmissionInferenceTests(unittest.TestCase):
    def test_bundled_locator_and_default_liveness_cli_handle_no_face(self):
        import numpy as np
        from PIL import Image

        from facial_attendance.ui.face_detection import YuNetFaceDetector

        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        # The legacy UI locator must work in the shared OpenCV 5 runtime too.
        self.assertEqual(YuNetFaceDetector()(blank), [])
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "blank.png"
            Image.fromarray(blank).save(image)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_liveness_notebook.py"),
                    "--action",
                    "inference",
                    "--device",
                    "cpu",
                    "--input",
                    str(image),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "src"),
                    "OMP_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                check=True,
            )
        prediction = json.loads(result.stdout)
        self.assertEqual(prediction["status"], "uncertain")
        self.assertFalse(prediction["allow_attendance"])

    def test_default_emotion_package_matches_the_ui_adapter(self):
        import numpy as np

        from facial_attendance.emotion.final import EmotionPredictor
        from facial_attendance.emotion.resnet_inference import ResNetEmotionInference

        y, x = np.indices((80, 96))
        bgr = np.stack(((x + y) % 256, x * 2 % 256, y * 3 % 256), axis=-1).astype(
            "uint8"
        )
        portable = EmotionPredictor().predict_bgr(bgr)
        ui = ResNetEmotionInference(
            ROOT / "artifacts/models/emotion/best.pt"
        ).predict_bgr(bgr)
        self.assertEqual(portable["label"], ui.label)
        self.assertEqual(portable["class_id"], ui.class_id)
        np.testing.assert_allclose(
            list(portable["probabilities"].values()),
            ui.probabilities,
            atol=1e-6,
            rtol=0,
        )
