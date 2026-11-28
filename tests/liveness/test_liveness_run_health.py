"""Progress snapshots are durable, bounded, and do not modify trainer output."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "training_health_fixture",
    Path(__file__).resolve().parents[2] / "scripts/liveness_training_health.py",
)
health_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health_module)


class TrainingHealthTests(unittest.TestCase):
    def test_real_progress_persists_and_new_epoch_can_reset_sample_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            now = [1000.0]
            root = Path(temporary)
            health = health_module.TrainingHealth(root, "job", clock=lambda: now[0])
            health.phase(
                "training", arm="binary-classification", output=root / "control"
            )
            for epoch, samples in ((15, 440000), (16, 64)):
                now[0] += 61
                health.record(
                    json.dumps(
                        {
                            "phase": "training",
                            "epoch": epoch,
                            "samples": samples,
                            "total_samples": 444357,
                            "optimizer_updates": 1,
                        }
                    )
                )
            saved = json.loads((root / "training-health.json").read_text())
            self.assertEqual((saved["epoch"], saved["samples"]), (16, 64))
            self.assertNotIn("_progress_key", saved)
            health.finish(error=ValueError("integrity failure"))
            self.assertEqual(
                json.loads((root / "training-health.json").read_text())["state"],
                "failed",
            )

    def test_output_handles_fragments_and_preserves_native_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            health = health_module.TrainingHealth(root, "job")
            native = io.StringIO()
            log = root / "console.log"
            output = health_module.DurableTrainingOutput(native, log, health)
            for text in ("first", " line\nsecond", " line\n", "last fragment"):
                output.write(text)
            output.flush()
            self.assertEqual(log.read_text(), native.getvalue())
            self.assertEqual(log.read_text(), "first line\nsecond line\nlast fragment")

    def test_checkpoint_timestamp_and_write_throttle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "control").mkdir()
            checkpoint = root / "control/last-training.pt"
            checkpoint.write_bytes(b"fixture; no model training")
            now = [1000.0]
            health = health_module.TrainingHealth(root, "job", clock=lambda: now[0])
            health.phase(
                "training", arm="binary-classification", output=root / "control"
            )
            before = (root / "training-health.json").read_bytes()
            now[0] += 30
            health.publish()
            self.assertEqual((root / "training-health.json").read_bytes(), before)
            saved = json.loads(before)
            self.assertEqual(
                saved["checkpoint_relative_path"], "control/last-training.pt"
            )
            self.assertIn("checkpoint_modified_at", saved)

    def test_new_arm_clears_old_counters_and_extraction_reports_real_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = [1000.0]
            health = health_module.TrainingHealth(root, "fixture", clock=lambda: now[0])
            health.phase("training", arm="binary-classification")
            health.record(
                '{"arm":"binary-classification","epoch":50,"total_updates":347200}'
            )
            health.phase("training", arm="aenet-full")
            saved = json.loads((root / "training-health.json").read_text())
            self.assertNotIn("completed_epoch", saved)
            self.assertNotIn("total_updates", saved)
            now[0] += 61
            health.record("Full extraction: 50000 files, CRC checked")
            self.assertEqual(
                json.loads((root / "training-health.json").read_text())[
                    "restored_files"
                ],
                50000,
            )
