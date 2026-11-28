"""Behavior contracts for liveness source training."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

RUNTIME_0 = all(
    importlib.util.find_spec(name)
    for name in ("torch", "torchvision", "numpy", "sklearn")
)


if RUNTIME_0:
    import numpy as np

    from facial_attendance.liveness.dataset_files import file_sha256
    from facial_attendance.liveness.datasets.geometric_cache import (
        MAP_DTYPE,
        load_geometric_cache,
    )
    from facial_attendance.liveness.datasets.source_dataset import (
        check_reproduction_partitions,
    )
    from facial_attendance.liveness.experiments.aenet_experiment import (
        FullTrainingConfig,
    )
    from facial_attendance.liveness.training.batching import epoch_training_config


@unittest.skipUnless(RUNTIME_0, "Requires the existing project runtime")
class FullTrainingTests(unittest.TestCase):
    def test_worker_benchmark_preserves_weights_rng_and_uses_two_counts(self):
        from unittest.mock import patch

        import torch

        from facial_attendance.liveness.training import loader_workers as benchmark

        model = torch.nn.Linear(2, 1)
        model.binary_model = torch.nn.Linear(2, 1)
        original = {name: value.clone() for name, value in model.state_dict().items()}

        class Rows:
            samples = list(range(2048))

            def __len__(self):
                return len(self.samples)

        training = Rows()
        clock = [0.0]
        original_fork = torch.random.fork_rng
        rng = torch.get_rng_state().clone()

        def train(probe, dataset, optimizer, config, *, on_batch, **kwargs):
            with torch.no_grad():
                next(probe.parameters()).add_(10)
            torch.rand(1)
            for update in range(1, 6):
                clock[0] += config.workers
                on_batch({"optimizer_updates": update})

        def predict(model, dataset, config, *, on_batch, **kwargs):
            clock[0] += config.workers
            on_batch()
            return None, None, None, float(config.workers)

        with (
            patch.object(benchmark.torch.cuda, "current_device", return_value=0),
            patch.object(benchmark.torch.cuda, "synchronize"),
            patch.object(benchmark.torch.cuda, "empty_cache"),
            patch.object(
                benchmark.torch.cuda, "get_device_name", return_value="test GPU"
            ),
            patch.object(
                benchmark.torch.random,
                "fork_rng",
                side_effect=lambda **kwargs: original_fork(devices=[]),
            ),
            patch.object(benchmark, "epoch_loader", return_value=[]),
            patch.object(benchmark, "train_multitask_epoch", side_effect=train),
            patch.object(benchmark, "predict_dataset", side_effect=predict),
            patch.object(benchmark.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            result = benchmark.choose_loader_workers(
                training, training, model, FullTrainingConfig(batch_size=256, workers=4)
            )
            self.assertEqual(result["candidate_workers"], [4, 2])
            self.assertEqual(result["selected_workers"], 2)
            self.assertEqual(len(result["measurements"]), 2)
            limited = benchmark.choose_loader_workers(
                training,
                training,
                model,
                FullTrainingConfig(batch_size=256, workers=4),
                budget_seconds=0,
            )
            self.assertEqual(limited["measurements"], [])
            self.assertEqual(limited["selected_workers"], 4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(
            all(
                torch.equal(original[name], value)
                for name, value in model.state_dict().items()
            )
        )

    def test_submitted_recipe_rejects_retired_or_unrequested_epoch_budgets(self):
        FullTrainingConfig().validate()
        with self.assertRaisesRegex(ValueError, "15 or 25 epochs"):
            FullTrainingConfig(epochs=4).validate()
        FullTrainingConfig(epochs=1).validate(preflight=True)
        FullTrainingConfig(epochs=15).validate(screening=True)
        FullTrainingConfig(epochs=25).validate(screening=True)
        self.assertEqual(FullTrainingConfig().epochs, 15)
        for epochs in (5, 50):
            with self.assertRaisesRegex(ValueError, "15 or 25 epochs"):
                FullTrainingConfig(epochs=epochs).validate(screening=True)
        with self.assertRaisesRegex(ValueError, "15 or 25 epochs"):
            FullTrainingConfig(epochs=4).validate(screening=True)
        for values in (
            {"learning_rate": 0.001},
            {"batch_size": 1},
            {"workers": -1},
            {"momentum": float("nan")},
        ):
            with self.assertRaises(ValueError):
                FullTrainingConfig(**values).validate()

    def test_epoch_archive_retains_exact_complete_state_and_metrics(self):
        from facial_attendance.liveness.training.checkpoints import (
            archive_epoch_checkpoint,
        )

        spec = importlib.util.spec_from_file_location(
            "archive_verification",
            Path(__file__).resolve().parents[2] / "scripts/liveness_training_health.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify_epoch_checkpoints = module.verify_epoch_checkpoints
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "binary-classification"
            output.mkdir()
            state = {
                "state_dict": {"binary_model.weight": torch.tensor([1.0])},
                "optimizer": {"state": {}},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [],
                "resume_epoch": 2,
                "progress": None,
                "updates": 3,
                "epoch": 1,
                "run": {"arm": "binary-classification"},
            }
            torch.save(state, output / "last-training.pt")
            dataset = SimpleNamespace(
                samples=[
                    SimpleNamespace(
                        relative_path="Data/val/face.jpg",
                        split="validation",
                        source_type=0,
                        attack_family="live",
                        binary_label=0,
                    )
                ]
            )
            archive_epoch_checkpoint(
                output,
                "binary-classification",
                1,
                {"validation": {"ACER": 0.2}},
                {"threshold": 0.3, "epoch": 1},
                validation_predictions=(dataset, np.array([0]), np.array([0.1])),
            )
            folder = output.parent / "binary-classification-epoch-checkpoints"
            receipt = json.loads((folder / "epoch-001.json").read_text())
            self.assertEqual(
                receipt["sha256"], file_sha256(output / "last-training.pt")
            )
            restored = torch.load(folder / receipt["checkpoint"], weights_only=True)
            self.assertEqual(restored["resume_epoch"], 2)
            self.assertTrue(torch.equal(restored["torch_rng"], state["torch_rng"]))
            self.assertEqual(receipt["inference_metadata"]["threshold"], 0.3)
            verify_epoch_checkpoints(folder, "binary-classification", 1)
            predictions = folder / receipt["validation_predictions"]["file"]
            original = predictions.read_bytes()
            self.assertIn(b"Data/val/face.jpg", original)
            predictions.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Archived validation"):
                verify_epoch_checkpoints(folder, "binary-classification", 1)
            predictions.write_bytes(original)
            with self.assertRaises(FileNotFoundError):
                verify_epoch_checkpoints(folder, "binary-classification", 2)
            with self.assertRaisesRegex(ValueError, "Incomplete archived"):
                verify_epoch_checkpoints(folder, "aenet-full", 1)
            (folder / receipt["checkpoint"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum/receipt"):
                verify_epoch_checkpoints(folder, "binary-classification", 1)
            torch.save({"changed": True}, output / "last-training.pt")
            with self.assertRaisesRegex(ValueError, "replace a different"):
                archive_epoch_checkpoint(output, "binary-classification", 1, {}, {})

    def test_larger_batch_begins_only_at_declared_epoch(self):
        config = FullTrainingConfig(
            epochs=25, large_batch_size=256, large_batch_start_epoch=19
        )
        config.validate()
        self.assertEqual(epoch_training_config(config, 18).batch_size, 64)
        self.assertEqual(epoch_training_config(config, 19).batch_size, 256)
        self.assertEqual(config.batch_size, 64)
        with self.assertRaises(ValueError):
            FullTrainingConfig(
                large_batch_size=256, large_batch_start_epoch=51
            ).validate()

    def test_train_validation_subject_overlap_is_rejected(self):
        train = SimpleNamespace(
            samples=[SimpleNamespace(subject_id="1", relative_path="train.jpg")]
        )
        validation = SimpleNamespace(
            samples=[SimpleNamespace(subject_id="2", relative_path="val.jpg")]
        )
        check_reproduction_partitions(train, validation)
        validation.samples[0].subject_id = "1"
        with self.assertRaisesRegex(ValueError, "subjects overlap"):
            check_reproduction_partitions(train, validation)

    def test_cache_requires_complete_bound_unchanged_data(self):
        class Source:
            def __len__(self):
                return 2

            def bindings(self):
                return {"fixture": "A"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(
                root / "geometric-targets.npy",
                np.zeros((2, 4, 14, 14), dtype=MAP_DTYPE),
            )
            receipt = {
                "completed": True,
                "completed_rows": 2,
                "bindings": {"data": {"fixture": "A"}},
                "array_sha256": file_sha256(root / "geometric-targets.npy"),
            }

            def save():
                (root / "geometric-cache.json").write_text(json.dumps(receipt))

            save()
            maps, _ = load_geometric_cache(root, Source())
            self.assertEqual(maps.shape, (2, 4, 14, 14))
            self.assertEqual(maps.dtype, np.dtype("float32"))
            receipt["completed"] = False
            save()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_geometric_cache(root, Source())
            receipt["completed"] = True
            receipt["bindings"]["data"]["fixture"] = "B"
            save()
            with self.assertRaisesRegex(ValueError, "another source"):
                load_geometric_cache(root, Source())
            receipt["bindings"]["data"]["fixture"] = "A"
            save()
            np.save(
                root / "geometric-targets.npy", np.ones((2, 4, 14, 14), dtype=MAP_DTYPE)
            )
            with self.assertRaisesRegex(ValueError, "array changed"):
                load_geometric_cache(root, Source())
            np.save(
                root / "geometric-targets.npy", np.ones((2, 4, 14, 14), dtype="<f2")
            )
            receipt["array_sha256"] = file_sha256(root / "geometric-targets.npy")
            save()
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                load_geometric_cache(root, Source())


RUNTIME_1 = all(importlib.util.find_spec(n) for n in ("torch", "torchvision"))


if RUNTIME_1:
    import torch

    from facial_attendance.liveness.training.batching import (
        SeededTrainingImages,
        epoch_loader,
    )


@unittest.skipUnless(RUNTIME_1, "Requires project ML runtime")
class TrainingBatchTests(unittest.TestCase):
    def test_replay_reproduces_augmentation_without_reprocessing_completed_batches(
        self,
    ):
        class Images:
            samples = list(range(10))

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return index, torch.rand(3)

        dataset = SeededTrainingImages(Images(), seed=42)
        config = SimpleNamespace(seed=42, batch_size=4, workers=0)
        full = list(epoch_loader(dataset, config, epoch=3))
        resumed = list(epoch_loader(dataset, config, epoch=3, start_sample=4))
        self.assertEqual(
            [x.tolist() for x, _ in full[1:]], [x.tolist() for x, _ in resumed]
        )
        self.assertTrue(all(torch.equal(a[1], b[1]) for a, b in zip(full[1:], resumed)))
        self.assertEqual(len(set(torch.cat([x for x, _ in full]).tolist())), 10)
        self.assertEqual(
            len(list(epoch_loader(dataset, config, epoch=3, start_sample=10))), 0
        )
        with self.assertRaises(ValueError):
            epoch_loader(dataset, config, epoch=3, start_sample=3)


RUNTIME_2 = all(
    importlib.util.find_spec(p) for p in ("torch", "torchvision", "sklearn")
)


if RUNTIME_2:
    import torch

    from facial_attendance.liveness.training.batching import training_subset


@unittest.skipUnless(RUNTIME_2, "Requires the existing project runtime")
class TrainingBudgetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            samples=[SimpleNamespace(binary_label=i % 2) for i in range(100)]
        )

        # Subset only needs indexing during loading; this fixture tests sampling.
        class Pool:
            samples = self.dataset.samples

            def __len__(self):
                return len(self.samples)

        self.pool = Pool()

    def test_stratified_budget_and_rng_isolation(self):
        torch.manual_seed(42)
        state = torch.get_rng_state().clone()
        subset = training_subset(self.pool, 60, 42, 1, 35)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(subset), 60)
        self.assertEqual(len(set(subset.indices)), 60)
        self.assertEqual(
            sum(self.pool.samples[i].binary_label == 0 for i in subset.indices), 35
        )
        self.assertEqual(
            subset.indices, training_subset(self.pool, 60, 42, 1, 35).indices
        )
        self.assertNotEqual(
            subset.indices, training_subset(self.pool, 60, 42, 2, 35).indices
        )

    def test_full_pool_and_invalid_budgets(self):
        self.assertIs(training_subset(self.pool, None, 42, 1), self.pool)
        self.assertEqual(len(training_subset(self.pool, 100, 42, 1)), 100)
        for budget, live_budget in (
            (0, None),
            (101, None),
            (60, 51),
            (60, 9),
            (None, 10),
        ):
            with (
                self.subTest(budget=budget, live_budget=live_budget),
                self.assertRaises(ValueError),
            ):
                training_subset(self.pool, budget, 42, 1, live_budget)


if __name__ == "__main__":
    unittest.main()
