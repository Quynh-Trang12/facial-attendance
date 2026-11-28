"""Resume ranks optimization state, not copied timestamps, and rejects wrong arms."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
health_spec = importlib.util.spec_from_file_location(
    "health_for_selection_fixture", SCRIPTS / "liveness_training_health.py"
)
health = importlib.util.module_from_spec(health_spec)
health_spec.loader.exec_module(health)
spec = importlib.util.spec_from_file_location(
    "colab_resume_fixture", SCRIPTS / "run_liveness_comparison.py"
)
workflow = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"liveness_training_health": health}):
    spec.loader.exec_module(workflow)


def saved(arm, updates):
    return {
        "run": {
            "arm": arm,
            "training_config": {"epochs": 15, "batch_size": 64},
            "data": {"train_samples_per_epoch": 444357},
        },
        "updates": 14 * 6944 + updates,
        "resume_epoch": 15,
        "epoch": 15,
        "progress": {"samples": updates * 64, "optimizer_updates": updates},
        "state_dict": {},
        "optimizer": {},
        "torch_rng": b"fixture",
        "cuda_rng": [],
    }


class ResumeSelectionTests(unittest.TestCase):
    def test_reviewed_membership_rejects_historical_or_other_selection_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inventory = directory / "cache-view.json"
            inventory.write_text('{"selection": "fixture"}')
            digest = workflow.sha(inventory)
            binding = {"reviewed_cache_view_sha256": digest}
            reviewed = {"data": {"train": binding, "validation": binding}}
            workflow.check_reviewed_run_binding(reviewed, directory)
            workflow.check_reviewed_run_binding({"data": {}}, None)
            for run, requested in (({"data": {}}, directory), (reviewed, None)):
                with self.assertRaisesRegex(ValueError, "selection changed"):
                    workflow.check_reviewed_run_binding(run, requested)
            inventory.write_text('{"selection": "different"}')
            with self.assertRaisesRegex(ValueError, "selection changed"):
                workflow.check_reviewed_run_binding(reviewed, directory)

    def test_runner_caps_workers_without_disabling_requested_benchmark(self):
        root = SCRIPTS.parent
        for requested, expected in ((None, 2), (12, 2), (1, 1)):
            args = SimpleNamespace(
                project_root=root,
                loader_workers=requested,
                benchmark_loader_workers=True,
            )
            state = SimpleNamespace(state={})
            with (
                patch.object(
                    workflow,
                    "runtime_loader_workers",
                    side_effect=lambda n: min(2, 12 if n is None else n),
                ),
                patch.object(workflow, "print_resume_plan"),
                patch("torch.cuda.is_available", return_value=False),
                self.assertRaisesRegex(RuntimeError, "CUDA is required"),
            ):
                workflow.run(args, state)
            self.assertEqual(args.loader_workers, expected)
            self.assertTrue(args.benchmark_loader_workers)

    def test_runner_uses_external_config_and_project_relative_overrides(self):
        root = SCRIPTS.parent
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "data with 'quotes'"
            config = Path(directory) / "project.toml"
            original = (root / "configs/project.toml").read_text()
            config.write_text(
                original.replace(
                    'liveness_data = "data/raw/liveness/celeba-spoof-full"',
                    "liveness_data = " + json.dumps(str(external)),
                )
            )
            args = SimpleNamespace(project_root=root, config=config)
            self.assertEqual(
                workflow.configured_paths(args).liveness_data, external.resolve()
            )
            args.data_root = Path("alternate/dataset")
            args.trials_root = Path(directory) / "camera"
            args.detector_model = Path("assets/detector.onnx")
            paths = workflow.configured_paths(args)
            self.assertEqual(paths.liveness_data, (root / args.data_root).resolve())
            self.assertEqual(paths.camera_trials, args.trials_root.resolve())
            self.assertEqual(
                paths.detector_model, (root / args.detector_model).resolve()
            )

    def test_evaluation_commands_use_configured_camera_and_capped_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "arm"
            output.mkdir()
            (output / "liveness-inference.pt").write_bytes(b"synthetic checkpoint")
            args = SimpleNamespace(
                project_root=SCRIPTS.parent,
                loader_workers=12,
                trials_root=Path(directory) / "recordings",
                detector_model=Path(directory) / "detector.onnx",
            )
            publisher = SimpleNamespace(
                state={"job_id": "fixture"},
                phase=lambda *a, **k: None,
                finish=lambda: None,
            )
            with (
                patch.object(workflow, "run_cli") as command,
                patch.object(workflow, "runtime_loader_workers", return_value=2),
                patch.dict(
                    sys.modules,
                    {
                        "liveness_camera_inputs": SimpleNamespace(
                            recordings=lambda p: {}
                        )
                    },
                ),
            ):
                workflow.evaluate_models(
                    args, publisher, Path(directory), {"arm": output}, "arm"
                )
            dataset, camera = [call.args[0] for call in command.call_args_list]
            self.assertEqual(dataset[dataset.index("--workers") + 1], "2")
            self.assertEqual(
                camera[camera.index("--trials-root") + 1], str(args.trials_root)
            )
            self.assertEqual(
                camera[camera.index("--detector-model") + 1], str(args.detector_model)
            )

    def test_default_plan_matches_the_four_configuration_experiment(self):
        from facial_attendance.liveness.experiments.aenet_experiment import (
            SUPERVISION_VARIANTS,
        )

        expected = [
            "binary-classification",
            "binary-semantic",
            "binary-geometric",
            "aenet-full",
        ]
        root = SCRIPTS.parent
        release = json.loads((root / "configs/liveness-release.json").read_text())
        notebook = json.loads(
            (root / "notebooks/liveness-training-and-evaluation.ipynb").read_text()
        )
        import ast

        configuration = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "ARMS = " in "".join(cell["source"])
        )
        assignment = next(
            node
            for node in ast.parse(configuration).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ARMS"
                for target in node.targets
            )
        )
        namespace = {"experiment": release["experiment"]}
        exec(
            compile(
                ast.Module(body=[assignment], type_ignores=[]), "selected-arms", "exec"
            ),
            namespace,
        )
        self.assertEqual(namespace["ARMS"], ["binary-semantic", "aenet-full"])
        self.assertEqual(
            release["experiment"]["arms"], ["binary-semantic", "aenet-full"]
        )
        from facial_attendance.liveness.experiments.aenet_experiment import (
            FullTrainingConfig,
        )

        recipe = FullTrainingConfig()
        self.assertEqual(release["experiment"]["backbone"], "ResNet-18")
        self.assertEqual(release["experiment"]["input_size"], [224, 224])
        self.assertEqual(release["experiment"]["optimizer"], "SGD")
        self.assertEqual(release["experiment"]["learning_rate"], recipe.learning_rate)
        self.assertEqual(
            release["experiment"]["color_distortion"],
            {
                name: getattr(recipe, name)
                for name in ("brightness", "contrast", "saturation", "hue")
            },
        )
        self.assertEqual(list(workflow.DEFAULT_COMPARISON_ARMS), expected)
        self.assertEqual(
            set(SUPERVISION_VARIANTS["binary-geometric"]),
            {"depth_map", "reflection_map"},
        )
        self.assertEqual(
            set(SUPERVISION_VARIANTS["aenet-full"]),
            set(SUPERVISION_VARIANTS["binary-semantic"])
            | {"depth_map", "reflection_map"},
        )
        self.assertEqual(SUPERVISION_VARIANTS["binary-classification"], ())
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                run_root=Path(temporary),
                project_root=root,
                screening_epochs=15,
                arms=None,
                initialize_arms=None,
            )
            plan = workflow.inspect_resume_plan(args)
        self.assertEqual([row["arm"] for row in plan["arms"]], expected)
        self.assertEqual([row["action"] for row in plan["arms"]], ["new"] * 4)
        self.assertEqual(plan["target_epochs"], 15)

    def test_notebook_keeps_supervision_and_result_sections_in_execution_order(self):
        notebook = json.loads(
            (
                SCRIPTS.parent / "notebooks/liveness-training-and-evaluation.ipynb"
            ).read_text()
        )
        positions = {cell["id"]: index for index, cell in enumerate(notebook["cells"])}
        self.assertLess(positions["comparison-design"], positions["semantic-targets"])
        self.assertLess(
            positions["semantic-targets-code"], positions["geometric-targets"]
        )
        self.assertLess(positions["geometric-targets-code"], positions["liveness-24"])
        self.assertLess(positions["liveness-24"], positions["validation-metrics"])
        self.assertLess(
            positions["validation-metrics"], positions["selection-inspection-code"]
        )

    def test_display_uses_relative_directories_without_changing_saved_plan(self):
        root = Path(tempfile.gettempdir()) / "private-account/coursework/runs"
        plan = {"arms": [{"directory": str(root / "control")}], "target_epochs": 50}
        stream = io.StringIO()
        with (
            patch.object(workflow, "inspect_resume_plan", return_value=plan),
            redirect_stdout(stream),
        ):
            workflow.print_resume_plan(SimpleNamespace(run_root=root))
        self.assertEqual(
            json.loads(stream.getvalue())["arms"][0]["directory"], "control"
        )
        self.assertNotIn("private-account", stream.getvalue())

    def test_inspection_requires_explicit_permission_for_uncheckpointed_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "binary-classification"
            path.mkdir()
            (path / "training-config.json").write_text(
                json.dumps({"arm": "binary-classification"})
            )
            args = SimpleNamespace(
                run_root=root,
                project_root=root,
                screening_epochs=15,
                arms=["binary-classification"],
                initialize_arms=None,
            )
            with self.assertRaisesRegex(RuntimeError, "refusing fresh training"):
                workflow.inspect_resume_plan(args)
            args.initialize_arms = ["binary-classification"]
            plan = workflow.inspect_resume_plan(args)
            self.assertEqual(plan["arms"][0]["action"], "new")
            self.assertEqual(
                plan["arms"][0]["reason"],
                "Explicitly authorized uncheckpointed attempt",
            )
            (path / "liveness-inference.pt").write_bytes(b"previous weights")
            with self.assertRaisesRegex(RuntimeError, "cannot be discarded"):
                workflow.inspect_resume_plan(args)

    def test_prior_output_without_config_never_becomes_a_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "binary-semantic"
            path.mkdir()
            (path / "last-training.pt").write_bytes(b"prior state")
            with self.assertRaisesRegex(RuntimeError, "refusing fresh training"):
                workflow.training_attempts(root, "binary-semantic")
            (path / "training-config.json").write_text(
                json.dumps({"arm": "binary-semantic"})
            )
            self.assertEqual(
                workflow.training_attempts(root, "binary-semantic"), [path]
            )

    def test_orphaned_epoch_archives_never_become_a_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(workflow.training_attempts(root, "binary-semantic"), [])
            archive = root / "binary-semantic-epoch-checkpoints"
            archive.mkdir()
            (archive / "epoch-001-training.pt").write_bytes(b"prior epoch")
            with self.assertRaisesRegex(RuntimeError, "refusing fresh training"):
                workflow.training_attempts(root, "binary-semantic")

    def test_epoch_extension_is_explicit_and_rejects_historical_fifty_epoch_runs(self):
        value = saved("binary-semantic", 0)
        value.update(epoch=5, resume_epoch=6, progress=None, updates=5 * 6944)
        value["run"].update(phase="screening")
        value["run"]["training_config"]["epochs"] = 5
        torch = SimpleNamespace(load=lambda *args, **kwargs: value)
        path = Path("/fixture/checkpoint")
        with self.assertRaises(ValueError):
            workflow.choose_checkpoint(torch, [path], "binary-semantic", epochs=15)
        with self.assertRaises(ValueError):
            workflow.choose_checkpoint(
                torch, [path], "binary-semantic", epochs=15, allow_extension=True
            )
        value.update(epoch=15, resume_epoch=16, updates=15 * 6944)
        value["run"]["training_config"]["epochs"] = 15
        self.assertEqual(
            workflow.choose_checkpoint(
                torch, [path], "binary-semantic", epochs=25, allow_extension=True
            ),
            path,
        )
        value["run"]["training_config"]["epochs"] = 50
        with self.assertRaises(ValueError):
            workflow.choose_checkpoint(
                torch, [path], "binary-semantic", epochs=25, allow_extension=True
            )

    def test_completed_run_without_rolling_checkpoint_is_verified_and_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "binary-semantic"
            path.mkdir()
            checkpoint = path / "liveness-inference.pt"
            checkpoint.write_bytes(b"selected checkpoint fixture")
            result = {
                "completed_epochs": 5,
                "screening_completed": True,
                "checkpoint_sha256": workflow.sha(checkpoint),
                "epoch_checkpoint_directory": "/old/device/binary-semantic-epoch-checkpoints",
            }
            (path / "training-results.json").write_text(json.dumps(result))
            with patch.object(workflow, "verify_epoch_checkpoints") as verify:
                self.assertEqual(
                    workflow.completed_training(
                        [path],
                        root,
                        "binary-semantic",
                        5,
                        screening=True,
                    ),
                    (path, result),
                )
            verify.assert_called_once_with(
                root / "binary-semantic-epoch-checkpoints", "binary-semantic", 5
            )
            (path / "training-config.json").write_text(
                json.dumps({"arm": "binary-semantic"})
            )
            abandoned = root / "binary-semantic-resume-abandoned"
            abandoned.mkdir()
            (abandoned / "console.log").write_text(
                "Failed before configuration or checkpoint"
            )
            args = SimpleNamespace(
                run_root=root,
                project_root=root,
                screening_epochs=5,
                arms=["binary-semantic"],
            )
            with patch.object(workflow, "verify_epoch_checkpoints"):
                plan = workflow.inspect_resume_plan(args)
            self.assertEqual(plan["arms"][0]["action"], "skip")
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(
                ValueError, "Completed selected checkpoint changed"
            ):
                workflow.completed_training(
                    [path], root, "binary-semantic", 5, screening=True
                )

    def test_highest_update_is_selected_even_when_input_order_is_reversed(self):
        with tempfile.TemporaryDirectory() as temporary:
            low, high = (
                Path(temporary) / "copied-recently",
                Path(temporary) / "older-mtime",
            )
            values = {
                low: saved("binary-classification", 100),
                high: saved("binary-classification", 150),
            }
            torch = SimpleNamespace(load=lambda path, **kwargs: values[path.parent])
            self.assertEqual(
                workflow.choose_checkpoint(torch, [high, low], "binary-classification"),
                high,
            )
            self.assertEqual(
                workflow.choose_checkpoint(torch, [], "binary-classification"), None
            )

    def test_changed_batch_uses_actual_completed_history_without_rewriting_updates(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            value = saved("binary-classification", 3)
            value["epoch"] = value["resume_epoch"] = 20
            value["run"]["training_config"].update(
                epochs=25, large_batch_size=256, large_batch_start_epoch=19
            )
            value["progress"]["samples"] = 3 * 256
            value["updates"] = 18 * 6944 + 1736 + 3
            rows = [
                {"epoch": epoch, "optimizer_updates": 6944 if epoch < 19 else 1736}
                for epoch in range(1, 20)
            ]
            (path / "training-history.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows)
            )
            torch = SimpleNamespace(load=lambda *args, **kwargs: value)
            self.assertEqual(
                workflow.choose_checkpoint(
                    torch, [path], "binary-classification", epochs=25
                ),
                path,
            )
            value["updates"] -= 1
            with self.assertRaisesRegex(ValueError, "Global optimizer"):
                workflow.choose_checkpoint(
                    torch, [path], "binary-classification", epochs=25
                )

    def test_inconsistent_progress_is_rejected(self):
        value = saved("binary-classification", 100)
        value["updates"] += 1
        torch = SimpleNamespace(load=lambda *args, **kwargs: value)
        with self.assertRaisesRegex(ValueError, "Global optimizer"):
            workflow.choose_checkpoint(
                torch, [Path("/fixture/checkpoint")], "binary-classification"
            )

    def test_other_arm_and_missing_optimizer_are_rejected(self):
        path = Path("/fixture/checkpoint")
        for value in (
            saved("aenet-full", 100),
            {
                key: item
                for key, item in saved("binary-classification", 100).items()
                if key != "optimizer"
            },
        ):
            torch = SimpleNamespace(load=lambda *args, **kwargs: value)
            with self.assertRaises(ValueError):
                workflow.choose_checkpoint(torch, [path], "binary-classification")
