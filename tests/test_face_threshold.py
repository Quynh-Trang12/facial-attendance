"""Writing the selected threshold into a copy of the selected checkpoint."""

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this environment")

import json
from pathlib import Path
import tempfile

import torch

from facial_attendance.recognition.model import FaceEmbeddingNet, build_meta, load_checkpoint, save_checkpoint


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts/set_face_threshold.py"
    spec = importlib.util.spec_from_file_location("set_threshold_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()
        cls.directory = tempfile.TemporaryDirectory(prefix="face-threshold-fixture-")
        cls.root = Path(cls.directory.name)
        torch.manual_seed(0)
        model = FaceEmbeddingNet(None).eval()
        cls.meta = build_meta(model, model_id="run-m", approach="metric")
        cls.best = cls.root / "best.pt"
        save_checkpoint(cls.best, model, cls.meta)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def selection(self, **changes):
        decision = {"selected_approach": "metric", "selected_model_id": "run-m",
                    "selected_model_version": self.meta["model_version"], "threshold": 0.4619967254593038,
                    "threshold_rule": "lowest validation cosine with FAR <= 1%", "validation_far": 0.00998,
                    "validation_tar": 0.8797, "reject_all": False, **changes}
        file = self.root / f"selection-{len(list(self.root.glob('selection-*')))}.json"
        file.write_text(json.dumps({"decision": decision}))
        return file

    def output(self):
        return self.root / f"deploy-{len(list(self.root.glob('deploy-*')))}.pt"

    def test_copy_holds_the_threshold_and_the_same_weights(self):
        output = self.output()
        result = self.script.deploy(self.best, self.selection(), output)
        model, meta = load_checkpoint(output)
        self.assertEqual(meta["threshold"], 0.4619967254593038)
        self.assertEqual(meta["model_version"], self.meta["model_version"])
        self.assertIn("selection.json sha256", meta["threshold_source"])
        self.assertEqual(result["threshold"], meta["threshold"])
        _, source = load_checkpoint(self.best)
        self.assertIsNone(source["threshold"])  # the source checkpoint is not changed

    def test_another_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "model_version"):
            self.script.deploy(self.best, self.selection(selected_model_version="000000000000"), self.output())
        with self.assertRaisesRegex(ValueError, "approach"):
            self.script.deploy(self.best, self.selection(selected_approach="classification"), self.output())

    def test_no_usable_threshold_or_existing_output_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no usable threshold"):
            self.script.deploy(self.best, self.selection(reject_all=True, threshold=None), self.output())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.script.deploy(self.best, self.selection(), self.best)


if __name__ == "__main__":
    unittest.main()
