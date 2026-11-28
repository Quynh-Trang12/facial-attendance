"""Relocation checks preserve bytes and reject mismatched or escaped inputs."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/release/package_liveness_release.py"
)
SPEC = importlib.util.spec_from_file_location("liveness_transfer_fixture", SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class TrainingTransferTests(unittest.TestCase):
    def test_source_relocation_accepts_only_exact_registered_code_and_preserves_data_checks(
        self,
    ):
        from facial_attendance.liveness.dataset_files import (
            compatible_sources,
            implementation_matches,
        )
        from facial_attendance.liveness.datasets.geometric_cache import (
            compatible_cache_bindings,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "new.py").write_text("# combined implementation")
            old, new = {"old.py": "before"}, {"new.py": "after"}
            declaration = {
                "before_files": {"old.py": "before"},
                "after_files": {"new.py": transfer.sha(root / "new.py")},
                "relocations": {"old.py": "new.py"},
                "scopes": [{"before": old, "after": new}],
            }
            (root / "configs/liveness-release.json").write_text(
                json.dumps({"source_transitions": [declaration]})
            )
            self.assertTrue(compatible_sources(old, new, root))
            self.assertFalse(compatible_sources(old, {"new.py": "changed"}, root))
            self.assertTrue(implementation_matches(root / "old.py", "before", root))
            loop = root / "loop.py"
            loop.write_text("# extracted optimizer loop")
            declaration["source_units"] = {"old.py": ["new.py", "loop.py"]}
            declaration["after_files"]["loop.py"] = transfer.sha(loop)
            (root / "configs/liveness-release.json").write_text(
                json.dumps({"source_transitions": [declaration]})
            )
            self.assertTrue(implementation_matches(root / "old.py", "before", root))
            loop.write_text("# unreviewed optimizer loop")
            self.assertFalse(implementation_matches(root / "old.py", "before", root))
            (root / "new.py").write_text("# changed implementation")
            self.assertFalse(implementation_matches(root / "old.py", "before", root))
            # Even a registered source move cannot authorize different teacher/data inputs.
            from unittest.mock import patch

            with patch(
                "facial_attendance.liveness.datasets.geometric_cache.compatible_sources",
                return_value=True,
            ):
                before = dict(
                    source_sha256=old, teacher="teacher-a", data="fixed-split"
                )
                self.assertTrue(
                    compatible_cache_bindings(before, {**before, "source_sha256": new})
                )
                self.assertFalse(
                    compatible_cache_bindings(
                        before, {**before, "teacher": "teacher-b"}
                    )
                )
                self.assertFalse(
                    compatible_cache_bindings(
                        before, {**before, "data": "changed-split"}
                    )
                )

    def test_fifteen_to_twentyfive_extension_preserves_state_and_next_optimizer_update(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, original, output = (
                root / "repo",
                root / "old",
                root / "extended",
            )
            source = repository / "src/facial_attendance/liveness"
            source.mkdir(parents=True)
            (original / "source").mkdir(parents=True)
            current = (
                SCRIPT.parents[2]
                / "src/facial_attendance/liveness/experiments/aenet_experiment.py"
            ).read_text()
            (source / "aenet_training.py").write_text(current)
            (original / "source/aenet_training.py").write_text(current)
            parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
            optimizer = torch.optim.SGD([parameter], lr=0.005, momentum=0.9)
            parameter.grad = torch.tensor([0.1, 0.2])
            optimizer.step()
            saved = {
                "run": {
                    "phase": "screening",
                    "arm": "binary-semantic",
                    "training_config": {"epochs": 15},
                    "protocol": {"epochs": 15},
                    "source_sha256": {
                        "aenet_training.py": transfer.sha(source / "aenet_training.py")
                    },
                },
                "state_dict": {"weight": parameter.detach().clone()},
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [torch.get_rng_state()],
                "updates": 10,
                "epoch": 15,
                "resume_epoch": 16,
                "progress": None,
                "best_epoch": 14,
                "best_key": [0.1, 0.2],
            }
            torch.save(saved, original / "last-training.pt")
            for name in (
                "liveness-inference.pt",
                "multitask-training.pt",
                "training-history.jsonl",
            ):
                (original / name).write_bytes(b"unchanged selected state/history")
            original_sha = transfer.sha(original / "last-training.pt")
            transfer.extend_screening(repository, original, output, 25)
            extended = torch.load(output / "last-training.pt", weights_only=True)
            self.assertEqual(extended["run"]["training_config"]["epochs"], 25)
            self.assertEqual(extended["run"]["protocol"]["epochs"], 25)
            self.assertEqual(original_sha, transfer.sha(original / "last-training.pt"))
            self.assertTrue(
                transfer.tensor_tree_equal(
                    {k: v for k, v in saved.items() if k != "run"},
                    {k: v for k, v in extended.items() if k != "run"},
                )
            )
            resumed = torch.nn.Parameter(extended["state_dict"]["weight"].clone())
            resumed_optimizer = torch.optim.SGD([resumed], lr=0.005, momentum=0.9)
            resumed_optimizer.load_state_dict(extended["optimizer"])
            parameter.grad = torch.tensor([0.1, 0.2])
            resumed.grad = parameter.grad.clone()
            optimizer.step()
            resumed_optimizer.step()
            self.assertTrue(torch.equal(parameter, resumed))
            for epochs in (15, 50):
                with (
                    self.subTest(epochs=epochs),
                    self.assertRaisesRegex(ValueError, "never shorten"),
                ):
                    transfer.extend_screening(
                        repository, output, root / "invalid", epochs
                    )
                self.assertFalse((root / "invalid").exists())
            (source / "aenet_training.py").write_text(current + "\n# unreviewed edit\n")
            with self.assertRaisesRegex(ValueError, "beyond its guard/report"):
                transfer.extend_screening(repository, original, root / "invalid", 25)
            self.assertFalse((root / "invalid").exists())

    def test_relocation_accepts_identical_content_and_canonical_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data with spaces"
            root.mkdir()
            inventory, annotations = root / "inventory.csv", root / "train.json"
            inventory.write_text("unchanged inventory")
            annotations.write_text("{}")
            old = {
                "root": "/original/data",
                "inventory": "/original/data/inventory.csv",
                "annotations": "/original/data/train.json",
                "samples": 2,
                "inventory_sha256": transfer.sha(inventory),
                "annotations_sha256": transfer.sha(annotations),
            }
            alias = Path(directory) / "alias"
            # Real relocation still runs without symlink privileges on Windows.
            alias = root
            try:
                candidate = Path(directory) / "alias"
                candidate.symlink_to(root, target_is_directory=True)
                alias = candidate
            except OSError:
                pass
            moved = transfer.relocate_binding(old, alias)
            self.assertEqual(moved["root"], str(root.resolve()))
            self.assertEqual(moved["annotations_sha256"], old["annotations_sha256"])
            self.assertEqual(old["root"], "/original/data")
            annotations.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                transfer.relocate_binding(old, root)

    def test_windows_bindings_relocate_to_native_paths_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, annotations = root / "inventory.csv", root / "train.json"
            inventory.write_text("same inventory")
            annotations.write_text("{}")
            binding = {
                "root": r"C:\Training Data",
                "inventory": r"c:\training data\inventory.csv",
                "annotations": r"C:\Training Data\train.json",
                "inventory_sha256": transfer.sha(inventory),
                "annotations_sha256": transfer.sha(annotations),
            }
            moved = transfer.relocate_binding(binding, root)
            self.assertEqual(moved["inventory"], str(inventory.resolve()))
            with self.assertRaises(ValueError):
                transfer.relocate_binding(
                    {**binding, "inventory": r"C:\Training Data\..\outside.csv"}, root
                )

    def test_relocation_rejects_inputs_outside_original_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                transfer.relocate_binding(
                    {"root": "/old/data", "inventory": "/other/private.csv"},
                    Path(directory),
                )

    def test_worker_change_requires_matching_real_input_proof(self):
        saved = {
            "run": {
                "training_config": {"workers": 4, "batch_size": 64, "seed": 42},
                "data": {
                    "train": {"annotations_sha256": "annotations"},
                    "inventory_sha256": "inventory",
                },
                "source_sha256": {"training.py": "source"},
            }
        }
        report = {
            "rows": [
                {
                    "workers": workers,
                    "batch_size": 64,
                    "inputs_bitwise_equal": True,
                    "input_target_sha256": "same tensors",
                }
                for workers in (4, 8)
            ],
            "seed": 42,
            "precision": "FP32",
            "training_annotations_sha256": "annotations",
            "inventory_sha256": "inventory",
            "source_sha256": {"training.py": "source"},
        }
        original = copy.deepcopy(saved)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loader-throughput.json"
            path.write_text(json.dumps(report))
            self.assertEqual(
                transfer.verified_loader_change(saved, 8, path)["new_workers"], 8
            )
            self.assertEqual(saved, original)
            for field, value in (
                ("inventory_sha256", "changed"),
                ("source_sha256", {}),
                ("precision", "FP16"),
            ):
                path.write_text(json.dumps({**report, field: value}))
                with self.assertRaises(ValueError):
                    transfer.verified_loader_change(saved, 8, path)
            different = copy.deepcopy(report)
            different["rows"][1]["input_target_sha256"] = "changed tensors"
            path.write_text(json.dumps(different))
            with self.assertRaises(ValueError):
                transfer.verified_loader_change(saved, 8, path)
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                transfer.verified_loader_change(saved, 12, path)
            with self.assertRaises(ValueError):
                transfer.verified_loader_change(saved, 8, None)
        self.assertIsNone(transfer.verified_loader_change(saved, None, None))

    def test_pipeline_change_requires_exact_sources_inputs_and_both_optimizer_arms(
        self,
    ):
        saved = {
            "run": {
                "source_sha256": {"aenet_training.py": "old"},
                "training_config": {"batch_size": 64},
                "data": {"inventory_sha256": "inventory"},
            }
        }
        sources = {"aenet_training.py": "new", "decoded_cache.py": "cache"}
        report = {
            "source_sha256": sources,
            "original_source_sha256": {"aenet_training.py": "old"},
            "training_input_target_bitwise_equal": True,
            "validation_input_bitwise_equal": True,
            "optimizer_state_bitwise_equal_by_arm": {
                "binary-classification": True,
                "aenet-full": True,
            },
            "precision": "FP32",
            "batch_size": 64,
            "inventory_sha256": "inventory",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            path.write_text(json.dumps(report))
            changed = {"aenet_training.py", "decoded_cache.py"}
            self.assertIsNotNone(
                transfer.verified_pipeline_change(saved, sources, changed, path)
            )
            for field, value in (
                ("validation_input_bitwise_equal", False),
                ("source_sha256", {}),
                (
                    "optimizer_state_bitwise_equal_by_arm",
                    {"binary-classification": True},
                ),
            ):
                path.write_text(json.dumps({**report, field: value}))
                with self.assertRaises(ValueError):
                    transfer.verified_pipeline_change(saved, sources, changed, path)
            with self.assertRaises(ValueError):
                transfer.verified_pipeline_change(
                    saved, sources, {"multitask_training.py"}, path
                )

    def test_source_ancestry_is_bound_to_real_evidence_and_cannot_escape(self):
        saved = {
            "run": {
                "source_sha256": {"aenet_training.py": "legacy"},
                "training_config": {"batch_size": 64},
                "data": {"inventory_sha256": "inventory"},
            }
        }
        sources = {"aenet_training.py": "new"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ancestor = root / "previous.json"
            ancestor.write_text(
                json.dumps({"source_sha256": {"aenet_training.py": "legacy"}})
            )
            report = {
                "source_sha256": sources,
                "original_source_sha256": {"aenet_training.py": "recent"},
                "source_ancestors": [
                    {
                        "file": "previous.json",
                        "sha256": transfer.sha(ancestor),
                        "field": "source_sha256",
                    }
                ],
                "training_input_target_bitwise_equal": True,
                "validation_input_bitwise_equal": True,
                "optimizer_state_bitwise_equal_by_arm": {
                    "binary-classification": True,
                    "aenet-full": True,
                },
                "precision": "FP32",
                "base_batch_size": 64,
                "inventory_sha256": "inventory",
            }
            path = root / "current.json"
            path.write_text(json.dumps(report))
            self.assertIsNotNone(
                transfer.verified_pipeline_change(
                    saved, sources, {"aenet_training.py"}, path
                )
            )
            ancestor.write_text("{}")
            with self.assertRaisesRegex(ValueError, "ancestry"):
                transfer.verified_pipeline_change(
                    saved, sources, {"aenet_training.py"}, path
                )
            report["source_ancestors"][0]["file"] = "../outside.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "ancestry"):
                transfer.verified_pipeline_change(
                    saved, sources, {"aenet_training.py"}, path
                )

    def test_state_comparison_covers_optimizer_rng_and_cursor(self):
        saved = {
            "state_dict": {"weight": torch.ones(3)},
            "optimizer": {0: {"momentum_buffer": torch.arange(3.0)}},
            "torch_rng": torch.get_rng_state(),
            "progress": {"samples": 64},
            "updates": 1,
        }
        clone = {**saved, "optimizer": {0: {"momentum_buffer": torch.arange(3.0)}}}
        self.assertTrue(transfer.tensor_tree_equal(saved, clone))
        clone["optimizer"][0]["momentum_buffer"][0] = 9
        self.assertFalse(transfer.tensor_tree_equal(saved, clone))
        self.assertFalse(
            transfer.tensor_tree_equal(saved, {**saved, "progress": {"samples": 128}})
        )


if __name__ == "__main__":
    unittest.main()
