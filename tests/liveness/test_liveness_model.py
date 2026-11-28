"""Controlled metric and checkpoint contracts; these are not accuracy results."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

FULL = all(
    importlib.util.find_spec(p) for p in ("torch", "torchvision", "sklearn", "numpy")
)
if FULL:
    import numpy as np
    import torch

    from facial_attendance.liveness.evaluation.evaluation_metrics import (
        binary_metrics,
        family_metrics,
        published_benchmark_metrics,
        validation_threshold,
    )
    from facial_attendance.liveness.models.binary_model import (
        LivenessPredictor,
        build_model,
        load_checkpoint,
        save_checkpoint,
    )
    from facial_attendance.liveness.training.checkpoints import model_meta
    from facial_attendance.liveness.training.validation import validation_checkpoint


@unittest.skipUnless(FULL, "Requires the existing project runtime")
class LivenessModelTests(unittest.TestCase):
    def test_spoof_positive_error_direction(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
        self.assertEqual(result["attack_accepted_as_live"], 1)
        self.assertEqual(result["live_rejected_as_spoof"], 1)
        self.assertEqual(result["APCER"], 0.5)
        self.assertEqual(result["BPCER"], 0.5)
        self.assertEqual(result["ACER"], 0.5)
        self.assertEqual(
            binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)["ROC_AUC"], 1
        )

    def test_threshold_and_attack_families(self):
        labels, scores = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        threshold = validation_threshold(labels, scores)
        self.assertEqual(binary_metrics(labels, scores, threshold)["ACER"], 0)
        families = family_metrics(
            labels, scores, ["live", "live", "print", "screen"], threshold
        )
        self.assertEqual(families["print"]["APCER"], 0)
        self.assertIsNone(families["print"]["BPCER"])
        with self.assertRaises(ValueError):
            validation_threshold([0, 0], [0.1, 0.2])
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [np.nan, 0.2], 0.5)

    def test_published_benchmark_uses_author_strict_threshold_rule(self):
        result = published_benchmark_metrics([0, 0, 1, 1], [0.1, 0.2, 0.2, 0.9])
        point = result["TPR_at_FPR"]["0.01"]
        self.assertEqual(point["threshold_descriptive_only"], 0.2)
        self.assertEqual(point["TPR"], 0.5)
        self.assertEqual(point["observed_FPR"], 0)
        self.assertEqual(result["nearest_discrete_EER"], 0.25)
        with self.assertRaises(ValueError):
            published_benchmark_metrics([0, 0], [0.1, 0.2])

    def test_checkpoint_and_integration_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            model = build_model().eval()
            # Synthetic serialization fixture; actual optimizer updates are
            # verified separately through the GPU worker and real input data.
            run = {"data": {"fixture": True}, "training_config": {"fixture": True}}
            meta = model_meta(model, run, updates=1, epoch=1, threshold=0.5)
            save_checkpoint(path, model.state_dict(), meta)
            save_checkpoint(
                path, model.state_dict(), {**meta, "inference_precision": "float64"}
            )
            with self.assertRaisesRegex(ValueError, "inference precision"):
                load_checkpoint(path)
            save_checkpoint(path, model.state_dict(), meta)
            restored, loaded = load_checkpoint(path)
            self.assertEqual(loaded["model_version"], meta["model_version"])
            with torch.inference_mode():
                self.assertEqual(
                    tuple(restored(torch.zeros(2, 3, 224, 224)).shape), (2, 1)
                )
            predictor = LivenessPredictor(path)
            output = predictor.predict_face(Image.new("RGB", (64, 80)))
            self.assertEqual(output["allow_attendance"], output["status"] == "live")
            self.assertAlmostEqual(output["live_score"] + output["spoof_score"], 1)
            with self.assertRaises(ValueError):
                predictor.predict_face(Image.new("L", (64, 64)))
            save_checkpoint(path, model.state_dict(), {**meta, "optimizer_updates": 0})
            with self.assertRaisesRegex(ValueError, "Untrained"):
                load_checkpoint(path)
            save_checkpoint(
                path, model.state_dict(), {**meta, "model_version": "wrong"}
            )
            with self.assertRaisesRegex(ValueError, "weights do not match"):
                load_checkpoint(path)

    def test_checkpoint_selection_uses_deployment_operating_point(self):
        labels = [0, 0, 1, 1]
        overlapping = [0.1, 0.2, 0.15, 0.9]
        separated_but_shifted = [0.8, 0.85, 0.9, 0.95]
        # A fixed 0.5 cut-off favours the overlapping model. The shifted model
        # is perfect at its validation threshold and must win the actual rule.
        self.assertLess(
            binary_metrics(labels, overlapping, 0.5)["ACER"],
            binary_metrics(labels, separated_but_shifted, 0.5)["ACER"],
        )
        old_key, _ = validation_checkpoint(labels, overlapping, 0.3)
        new_key, selected = validation_checkpoint(labels, separated_but_shifted, 0.3)
        self.assertLess(new_key, old_key)
        self.assertEqual(selected["ACER"], 0)
        self.assertEqual(selected["threshold"], 0.9)
        with self.assertRaisesRegex(ValueError, "loss must be finite"):
            validation_checkpoint(labels, separated_but_shifted, np.nan)


if __name__ == "__main__":
    unittest.main()
