"""Run matched liveness configurations with durable checkpoint and progress records.

Process locking requires a POSIX runtime.

This orchestration does not change the trainer, model, data, loss or precision.
The ML imports are delayed until the configured project is available on sys.path.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from liveness_training_health import (
    DurableTrainingOutput,
    TrainingHealth,
    check_producer_available,
    runtime_loader_workers,
    verify_epoch_checkpoints,
)
from liveness_training_health import run_logged_command as run_cli

# Locking is only needed when launching the Linux/Colab training workflow.
if os.name == "posix":
    import fcntl

# Compare the two supervision groups and their combination against binary classification.
DEFAULT_COMPARISON_ARMS = (
    "binary-classification",
    "binary-semantic",
    "binary-geometric",
    "aenet-full",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configured_paths(args):
    """Resolve CLI overrides and shared storage paths from the project root."""
    # The runner can be loaded before a staged project package is available.
    sys.path.insert(0, str(args.project_root / "src"))
    from facial_attendance.config import load_paths

    config = getattr(args, "config", None) or Path("configs/project.toml")
    return load_paths(
        args.project_root / config,
        root=args.project_root,
        overrides={
            "liveness_data": getattr(args, "data_root", None),
            "camera_trials": getattr(args, "trials_root", None),
            "detector_model": getattr(args, "detector_model", None),
        },
    )


def check_reviewed_run_binding(run, directory):
    """Never reuse completed results or optimization state from another selection."""
    expected = sha(Path(directory) / "cache-view.json") if directory else None
    for split in ("train", "validation"):
        observed = run["data"].get(split, {}).get("reviewed_cache_view_sha256")
        if observed != expected:
            raise ValueError("Training selection changed; use a separate output run")


def batch_size_for_epoch(recipe, epoch):
    larger = recipe.get("large_batch_size")
    if larger is not None and epoch >= recipe["large_batch_start_epoch"]:
        return larger
    return recipe["batch_size"]


def choose_checkpoint(torch, candidates, arm, *, epochs=15, allow_extension=False):
    """Rank readable optimization state, never copied-file modification times."""
    valid = []
    for path in candidates:
        saved = torch.load(
            path / "last-training.pt", map_location="cpu", weights_only=True
        )
        saved_epochs = saved["run"]["training_config"]["epochs"]
        extension = (
            allow_extension
            and (saved_epochs, epochs) in ((15, 25),)
            and saved["run"]["phase"] == "screening"
        )
        if saved["run"]["arm"] != arm or (saved_epochs != epochs and not extension):
            raise ValueError(
                "Checkpoint belongs to another arm/epoch recipe: " + str(path)
            )
        for required in (
            "state_dict",
            "optimizer",
            "torch_rng",
            "cuda_rng",
            "updates",
            "resume_epoch",
            "epoch",
            "progress",
        ):
            if required not in saved:
                raise ValueError("Incomplete resume checkpoint: " + required)
        count = saved["run"]["data"]["train_samples_per_epoch"]
        recipe = saved["run"]["training_config"]
        batch = batch_size_for_epoch(recipe, saved["epoch"])
        per_epoch = (count + batch - 1) // batch
        progress = saved["progress"]
        if progress is None:
            expected = saved["epoch"] * per_epoch
            if saved["resume_epoch"] != saved["epoch"] + 1:
                raise ValueError("Invalid epoch-boundary resume cursor")
        else:
            samples = progress["samples"]
            if not 0 <= samples <= count or (samples < count and samples % batch):
                raise ValueError("Invalid mid-epoch sample cursor")
            steps = (samples + batch - 1) // batch
            if (
                progress["optimizer_updates"] != steps
                or saved["resume_epoch"] != saved["epoch"]
            ):
                raise ValueError("Resume cursor/update count differs")
            expected = (saved["epoch"] - 1) * per_epoch + steps
        completed_epochs = saved["epoch"] if progress is None else saved["epoch"] - 1
        history_file = path / "training-history.jsonl"
        if history_file.exists():
            rows = [
                json.loads(line)
                for line in history_file.read_text().splitlines()
                if line
            ]
            history = {
                row["epoch"]: row for row in rows if row["epoch"] <= completed_epochs
            }
            if set(history) != set(range(1, completed_epochs + 1)):
                raise ValueError(
                    "Incomplete epoch history for cumulative optimizer updates"
                )
            for number, row in history.items():
                size = batch_size_for_epoch(recipe, number)
                if row["optimizer_updates"] != (count + size - 1) // size:
                    raise ValueError("Epoch history disagrees with its batch schedule")
            expected = sum(row["optimizer_updates"] for row in history.values()) + (
                progress["optimizer_updates"] if progress else 0
            )
        elif recipe.get("large_batch_size") is not None and completed_epochs:
            raise ValueError("Batch schedule requires actual completed epoch history")
        if saved["updates"] != expected:
            raise ValueError("Global optimizer updates differ from the saved cursor")
        valid.append((saved["updates"], saved["resume_epoch"], str(path), path))
    return max(valid)[-1] if valid else None


def training_attempts(run_root, arm, *, allow_unconfigured=False):
    """Detect damaged prior outputs before considering a fresh configuration."""
    attempts = []
    for path in sorted(run_root.glob(arm + "*")):
        if not path.is_dir():
            continue
        config_file = path / "training-config.json"
        if config_file.exists():
            if json.loads(config_file.read_text()).get("arm") == arm:
                attempts.append(path)
        elif (
            not allow_unconfigured
            and (path.name == arm or path.name.startswith(arm + "-resume-"))
            and any(path.iterdir())
        ):
            raise RuntimeError(
                "Prior output lacks its configuration; refusing fresh training: "
                + str(path)
            )
    archives = run_root / (arm + "-epoch-checkpoints")
    if (
        not allow_unconfigured
        and not attempts
        and archives.exists()
        and any(archives.iterdir())
    ):
        raise RuntimeError(
            "Archived state exists without a run directory; refusing fresh training: "
            + arm
        )
    return attempts


def inspect_resume_plan(args):
    """Read actual CPU checkpoints without allocating CUDA or writing run outputs."""
    import torch

    target = args.screening_epochs or 15
    screening = args.screening_epochs is not None
    plan = []
    for arm in args.arms or DEFAULT_COMPARISON_ARMS:
        # A verified completed run takes precedence over abandoned older
        # folders. If completion cannot be proved, all prior folders matter.
        attempts = training_attempts(args.run_root, arm, allow_unconfigured=True)
        completed = completed_training(
            attempts, args.run_root, arm, target, screening=screening
        )
        if completed:
            path, result = completed
            plan.append(
                {
                    "arm": arm,
                    "action": "skip",
                    "completed_epochs": target,
                    "selected_checkpoint_sha256": result["checkpoint_sha256"],
                    "directory": str(path),
                }
            )
            continue
        attempts = training_attempts(args.run_root, arm)
        candidates = [path for path in attempts if (path / "last-training.pt").exists()]
        explicit = {
            "binary-classification": getattr(args, "resume_control", None),
            "aenet-full": getattr(args, "resume_full", None),
        }.get(arm)
        if explicit is not None:
            if candidates:
                raise ValueError(
                    "Do not combine an explicit resume with prior outputs in this run root"
                )
            candidates = [explicit]
        resume = choose_checkpoint(
            torch, candidates, arm, epochs=target, allow_extension=screening
        )
        if attempts and resume is None:
            if arm not in (getattr(args, "initialize_arms", None) or ()):
                raise RuntimeError(
                    "Prior output has no valid resume state; refusing fresh training: "
                    + arm
                )
            if any(
                list(path.rglob("*.pt")) or list(path.rglob("*.pth"))
                for path in attempts
            ):
                raise RuntimeError(
                    "Existing model weights cannot be discarded for initialization: "
                    + arm
                )
        if resume is None:
            reason = (
                "Explicitly authorized uncheckpointed attempt"
                if attempts
                else "No prior output or archived state"
            )
            plan.append({"arm": arm, "action": "new", "reason": reason})
            continue
        saved = torch.load(
            resume / "last-training.pt", map_location="cpu", weights_only=True
        )
        if (
            not saved["state_dict"]
            or not torch.is_tensor(saved["torch_rng"])
            or not isinstance(saved["cuda_rng"], list)
            or not saved["optimizer"].get("param_groups")
        ):
            raise ValueError(
                "Checkpoint has incomplete model/optimizer/RNG state: " + arm
            )
        epochs = saved["epoch"] if saved["progress"] is None else saved["epoch"] - 1
        if screening:
            verify_epoch_checkpoints(
                args.run_root / (arm + "-epoch-checkpoints"), arm, epochs
            )
        if saved["run"]["training_config"]["epochs"] != target:
            from release.package_liveness_release import screening_extension_sources

            screening_extension_sources(args.project_root, resume, saved)
        plan.append(
            {
                "arm": arm,
                "action": "resume",
                "directory": str(resume),
                "checkpoint_sha256": sha(resume / "last-training.pt"),
                "completed_epochs": epochs,
                "resume_epoch": saved["resume_epoch"],
                "updates": saved["updates"],
                "progress": saved["progress"],
                "saved_target_epochs": saved["run"]["training_config"]["epochs"],
            }
        )
    return {"schema": "liveness-resume-plan-v1", "target_epochs": target, "arms": plan}


def print_resume_plan(args):
    """Display run-relative directories without exposing local account paths."""
    plan = copy.deepcopy(inspect_resume_plan(args))
    for arm in plan["arms"]:
        if "directory" in arm:
            directory = Path(arm["directory"])
            try:
                arm["directory"] = directory.relative_to(args.run_root).as_posix()
            except ValueError:
                arm["directory"] = directory.name
    print(json.dumps(plan, indent=2), flush=True)


def prepare_decoded_cache(args, health, root):
    """Reuse prepared mmap arrays; build only when explicitly given source images."""
    if args.decoded_cache is None:
        return
    from facial_attendance.liveness.datasets.decoded_cache import build_decoded_cache
    from facial_attendance.liveness.datasets.source_dataset import ReproductionImages

    health.phase("preparing-decoded-cache")
    for split, annotations in (
        ("train", "reviewed_project_train.json"),
        ("validation", "project_validation.json"),
    ):
        source = ReproductionImages(
            root,
            root / "audits/current-schema-2026-11-12/inventory.csv",
            root / annotations,
        )
        last_progress = time.monotonic()

        def progress(count, total):
            nonlocal last_progress
            if time.monotonic() - last_progress >= 30 or count == total:
                health.state.update(
                    cached_split=split,
                    cached_samples=count,
                    cache_total_samples=total,
                    last_progress_at=datetime.now(UTC).isoformat(),
                )
                health.publish()
                print(
                    json.dumps(
                        {
                            "phase": "preparing-decoded-cache",
                            "split": split,
                            "samples": count,
                            "total_samples": total,
                        }
                    ),
                    flush=True,
                )
                last_progress = time.monotonic()

        cache_workers = max(1, runtime_loader_workers(args.loader_workers))
        cached = build_decoded_cache(
            source, args.decoded_cache / split, workers=cache_workers, progress=progress
        )
        shutil.copyfile(
            args.decoded_cache / split / "decoded-faces.json",
            args.run_root
            / ("decoded-cache-" + split + "-" + health.state["job_id"] + ".json"),
        )
        del cached, source


def completed_training(attempts, run_root, arm, epochs, *, screening):
    """Accept completed results independently of an optional rolling checkpoint."""
    completed = []
    for path in attempts:
        result_file = path / "training-results.json"
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text())
        completion_field = (
            "screening_completed" if screening else "full_training_completed"
        )
        if result.get("completed_epochs") != epochs or not result.get(completion_field):
            continue
        if sha(path / "liveness-inference.pt") != result["checkpoint_sha256"]:
            raise ValueError("Completed selected checkpoint changed: " + str(path))
        if result.get("epoch_checkpoint_directory") is not None:
            verify_epoch_checkpoints(
                run_root / (arm + "-epoch-checkpoints"), arm, epochs
            )
        elif screening:
            raise ValueError("Screening result lacks per-epoch checkpoints")
        completed.append((path, result))
    return completed[-1] if completed else None


def add_reference_results(args, results, directories):
    """Include historical controls only when their data and recipe match exactly."""
    if args.reference_runs_root is None:
        return
    reference = json.loads((args.reference_runs_root / "model-choice.json").read_text())
    for arm, result in reference["arms"].items():
        if arm in results:
            raise ValueError("Reference arm duplicates a newly trained arm")
        directory = args.reference_runs_root / Path(reference["directories"][arm]).name
        if sha(directory / "liveness-inference.pt") != result["checkpoint_sha256"]:
            raise ValueError("Reference checkpoint changed: " + arm)
        candidate_data = json.loads(
            (next(iter(directories.values())) / "training-config.json").read_text()
        )
        reference_data = json.loads((directory / "training-config.json").read_text())
        for key in ("training_config", "initialization_sha256"):
            if reference_data[key] != candidate_data[key]:
                raise ValueError("Reference recipe differs: " + key)
        for key in ("inventory_sha256", "train", "validation"):
            if reference_data["data"][key] != candidate_data["data"][key]:
                raise ValueError("Reference data differs: " + key)
        results[arm], directories[arm] = result, directory


def evaluate_models(args, health, root, directories, chosen):
    """Assess frozen checkpoints after validation selection; never tune on test."""
    paths = configured_paths(args)
    for arm, output in directories.items():
        checkpoint = output / "liveness-inference.pt"
        health.phase("evaluation", arm=arm, output=output)
        evaluation = output / "test-evaluation"
        if evaluation.exists() and not (evaluation / "evaluation.json").exists():
            evaluation = output / ("test-evaluation-" + health.state["job_id"])
        if (evaluation / "evaluation.json").exists():
            prior = json.loads((evaluation / "evaluation.json").read_text())
            if prior["checkpoint_sha256"] != sha(checkpoint):
                raise ValueError("Stale test evaluation for " + arm)
        else:
            command = [
                sys.executable,
                str(
                    args.project_root
                    / "scripts/evaluation/score_liveness_image_split.py"
                ),
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(root),
                "--inventory",
                str(root / "audits/current-schema-2026-11-12/inventory.csv"),
                "--annotations",
                str(root / "metas/intra_test/test_label.json"),
                "--split",
                "test",
                "--output-dir",
                str(evaluation),
                "--workers",
                str(runtime_loader_workers(args.loader_workers)),
            ]
            run_cli(command)
        trials = output / "presentation-evaluation"
        if trials.exists() and not (trials / "physical-evaluation.json").exists():
            trials = output / ("presentation-evaluation-" + health.state["job_id"])
        if (trials / "physical-evaluation.json").exists() and json.loads(
            (trials / "physical-evaluation.json").read_text()
        )["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("Stale physical evaluation for " + arm)
        if not (trials / "physical-evaluation.json").exists():
            run_cli(
                [
                    sys.executable,
                    str(
                        args.project_root
                        / "scripts/evaluation/score_liveness_trial_inventory.py"
                    ),
                    "--checkpoint",
                    str(checkpoint),
                    "--trials-root",
                    str(paths.camera_trials),
                    "--detector-model",
                    str(paths.detector_model),
                    "--output-dir",
                    str(trials),
                ]
            )
    health.state["evaluations_complete"] = True
    health.state["chosen_arm"] = chosen
    # Camera dependencies are loaded only after the selected project is configured.
    from liveness_camera_inputs import recordings

    health.state["physical_print_trial_available"] = any(
        kind == "print" for _, _, _, kind in recordings(paths.camera_trials).values()
    )
    health.finish()


def run(args, health):
    sys.path.insert(0, str(args.project_root / "src"))
    import torch

    from facial_attendance.liveness.dataset_files import compatible_sources
    from facial_attendance.liveness.experiments.aenet_experiment import (
        SUPERVISION_VARIANTS,
        FullTrainingConfig,
        run_full_training,
    )

    paths = configured_paths(args)
    requested_workers = args.loader_workers
    args.loader_workers = runtime_loader_workers(requested_workers)
    if args.benchmark_loader_workers and args.loader_workers == 0:
        raise ValueError("Loader benchmarking requires at least one worker")
    health.state.update(
        requested_loader_workers=requested_workers,
        loader_workers=args.loader_workers,
        data_root=str(paths.liveness_data),
        camera_trials=str(paths.camera_trials),
        detector_model=str(paths.detector_model),
    )

    print_resume_plan(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    if str(torch.__version__) not in ("2.13.0+cu126", "2.13.0+cu130"):
        raise RuntimeError("Use the declared Python 3.13/CUDA project environment")

    root = paths.liveness_data
    prepare_decoded_cache(args, health, root)
    screening = args.screening_epochs is not None
    target_epochs = args.screening_epochs or 15
    if screening and (
        target_epochs not in (15, 25)
        or not args.validation_only
        or args.reference_runs_root is not None
    ):
        raise ValueError(
            "Screening requires 15 or 25 epochs, validation-only and matched controls"
        )

    config = FullTrainingConfig(epochs=target_epochs)
    results, directories = {}, {}
    arms = args.arms or DEFAULT_COMPARISON_ARMS
    if len(set(arms)) != len(arms) or set(arms) - set(SUPERVISION_VARIANTS):
        raise ValueError("Unknown or duplicated supervision variants")
    health.state.update(
        requested_arms=list(arms),
        validation_only=args.validation_only,
        target_epochs=target_epochs,
        screening=screening,
    )
    health.publish(force=True)
    worker_receipt = (
        args.run_root.parent
        / "runtime-performance"
        / args.run_root.name
        / ("loader-benchmark-" + health.state["job_id"] + ".json")
    )
    if args.benchmark_loader_workers and worker_receipt.exists():
        args.loader_workers = runtime_loader_workers(
            json.loads(worker_receipt.read_text())["selected_workers"]
        )

    for arm in arms:
        explicit_resume = {
            "binary-classification": args.resume_control,
            "aenet-full": args.resume_full,
        }.get(arm)
        attempts = training_attempts(args.run_root, arm, allow_unconfigured=True)
        candidates = [path for path in attempts if (path / "last-training.pt").exists()]
        completed = completed_training(
            attempts, args.run_root, arm, target_epochs, screening=screening
        )
        if completed:
            directories[arm], results[arm] = completed
            saved_config = json.loads(
                (directories[arm] / "training-config.json").read_text()
            )
            check_reviewed_run_binding(
                saved_config, getattr(args, "reviewed_cache_view", None)
            )
            config = FullTrainingConfig(**saved_config["training_config"])
        else:
            attempts = training_attempts(args.run_root, arm)
            resume = explicit_resume or choose_checkpoint(
                torch, candidates, arm, epochs=target_epochs, allow_extension=screening
            )
            if explicit_resume is not None and candidates:
                raise ValueError(
                    "Do not combine an explicit resume with prior outputs in this run root"
                )
            saved = (
                torch.load(
                    resume / "last-training.pt", map_location="cpu", weights_only=True
                )
                if resume
                else None
            )
            if saved:
                check_reviewed_run_binding(
                    saved["run"], getattr(args, "reviewed_cache_view", None)
                )
            if saved and saved["run"]["training_config"]["epochs"] != target_epochs:
                from release.package_liveness_release import extend_screening

                extension = (
                    args.project_root
                    / "transfer"
                    / ("extension-" + health.state["job_id"] + "-" + arm)
                )
                resume = extend_screening(
                    args.project_root, resume, extension, target_epochs
                )
                saved = torch.load(
                    resume / "last-training.pt", map_location="cpu", weights_only=True
                )
            prior_workers = (
                saved["run"]["training_config"]["workers"] if saved else config.workers
            )
            selected_workers = (
                args.loader_workers
                if args.loader_workers is not None
                else prior_workers
            )
            prior_recipe = saved["run"]["training_config"] if saved else {}
            if screening and saved:
                completed_epochs = (
                    saved["epoch"] if saved["progress"] is None else saved["epoch"] - 1
                )
                verify_epoch_checkpoints(
                    args.run_root / (arm + "-epoch-checkpoints"), arm, completed_epochs
                )
            config = FullTrainingConfig(
                epochs=target_epochs,
                workers=selected_workers,
                large_batch_size=args.training_batch_size
                if args.training_batch_size is not None
                else prior_recipe.get("large_batch_size"),
                large_batch_start_epoch=args.batch_start_epoch
                if args.training_batch_size is not None
                else prior_recipe.get("large_batch_start_epoch", 1),
                gpu_augmentation=prior_recipe.get("gpu_augmentation", False)
                if args.gpu_augmentation is None
                else args.gpu_augmentation,
            )
            augmentation_changed = config.gpu_augmentation != prior_recipe.get(
                "gpu_augmentation", False
            )
            schedule_changed = config.large_batch_size != prior_recipe.get(
                "large_batch_size"
            ) or config.large_batch_start_epoch != prior_recipe.get(
                "large_batch_start_epoch", 1
            )
            current_sources = {
                path.name: sha(path)
                for path in (args.project_root / "src/facial_attendance/liveness").glob(
                    "*.py"
                )
            }
            source_changed = bool(
                saved
                and not compatible_sources(
                    saved["run"]["source_sha256"], current_sources, args.project_root
                )
            )
            if resume and (
                selected_workers != prior_workers
                or source_changed
                or schedule_changed
                or augmentation_changed
            ):
                if selected_workers != prior_workers and args.loader_benchmark is None:
                    raise ValueError(
                        "Changing loader workers requires a measured input-parity benchmark"
                    )
                health.phase("verifying-resume", arm=arm)
                rebound = (
                    args.project_root
                    / "transfer"
                    / ("loader-" + health.state["job_id"] + "-" + arm)
                )
                command = [
                    sys.executable,
                    str(
                        args.project_root
                        / "scripts/release/package_liveness_release.py"
                    ),
                    "relocate",
                    "--repository",
                    str(args.project_root),
                    "--inventory",
                    str(args.project_root / "transfer/release-inventory.json"),
                    "--output",
                    str(rebound),
                    "--resume",
                    str(resume),
                ]
                if selected_workers != prior_workers:
                    command += [
                        "--loader-workers",
                        str(selected_workers),
                        "--loader-benchmark",
                        str(args.loader_benchmark),
                    ]
                if args.pipeline_benchmark is not None:
                    command += ["--pipeline-benchmark", str(args.pipeline_benchmark)]
                if schedule_changed:
                    if args.batch_benchmark is None:
                        raise ValueError(
                            "New batch schedule requires its actual GPU benchmark"
                        )
                    command += [
                        "--training-batch-size",
                        str(config.large_batch_size),
                        "--batch-start-epoch",
                        str(config.large_batch_start_epoch),
                        "--batch-benchmark",
                        str(args.batch_benchmark),
                    ]
                if augmentation_changed:
                    command += [
                        "--gpu-augmentation"
                        if config.gpu_augmentation
                        else "--no-gpu-augmentation"
                    ]
                run_cli(command)
                resume = rebound / arm
            output = args.run_root / (
                arm if not attempts else arm + "-resume-" + health.state["job_id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            if resume and (resume / "epoch-extension.json").exists():
                shutil.copyfile(
                    resume / "epoch-extension.json", output / "epoch-extension.json"
                )
            if resume and (resume.parent / "migration.json").exists():
                shutil.copyfile(
                    resume.parent / "migration.json", output / "resume-migration.json"
                )
            health.phase("training", arm=arm, output=output)
            results[arm] = run_full_training(
                root,
                root / "audits/current-schema-2026-11-12/inventory.csv",
                root / "reviewed_project_train.json",
                root / "project_validation.json",
                args.geometry_cache,
                args.project_root
                / "artifacts/pretrained/resnet18-imagenet1k-initialization.pth",
                output,
                config,
                arm=arm,
                resume=resume,
                checkpoint_seconds=300,
                decoded_cache=args.decoded_cache,
                reviewed_cache_view=getattr(args, "reviewed_cache_view", None),
                screening=screening,
                retain_epoch_checkpoints=True,
                benchmark_receipt=worker_receipt
                if args.benchmark_loader_workers
                and not worker_receipt.exists()
                and resume is None
                else None,
            )
            if args.benchmark_loader_workers and worker_receipt.exists():
                args.loader_workers = runtime_loader_workers(
                    json.loads(worker_receipt.read_text())["selected_workers"]
                )
            verify_epoch_checkpoints(
                args.run_root / (arm + "-epoch-checkpoints"), arm, target_epochs
            )
            directories[arm] = output
        health.state["completed_arms"][arm] = results[arm]["completed_epochs"]
        health.publish(force=True)
        torch.cuda.empty_cache()

    add_reference_results(args, results, directories)
    chosen = min(results, key=lambda arm: tuple(results[arm]["selection_key"]))
    choice = {
        "chosen": chosen,
        "basis": "validation selection keys",
        "test_or_physical_used_for_choice": False,
        "arms": results,
        "directories": {arm: str(path) for arm, path in directories.items()},
    }
    choice_file = args.run_root / "model-choice.json"
    if choice_file.exists() and json.loads(choice_file.read_text()) != choice:
        previous_bytes = choice_file.read_bytes()
        previous = json.loads(previous_bytes)
        budgets = {value["completed_epochs"] for value in previous["arms"].values()}
        if (
            set(previous["arms"]) != set(results)
            or len(budgets) != 1
            or next(iter(budgets)) >= target_epochs
        ):
            raise ValueError("Existing model choice belongs to another result")
        archive = args.run_root / f"model-choice-epochs-{next(iter(budgets))}.json"
        if archive.exists() and archive.read_bytes() != previous_bytes:
            raise ValueError("Prior model-selection archive differs")
        if not archive.exists():
            archive.write_bytes(previous_bytes)

    pending_choice = choice_file.with_suffix(".json.writing")
    pending_choice.write_text(json.dumps(choice, indent=2) + "\n")
    pending_choice.replace(choice_file)

    if args.validation_only:
        health.state.update(
            chosen_arm=chosen,
            selection_complete=True,
            evaluation_scope="Validation-only ablation; final test and physical evaluation deferred",
        )
        health.finish()
        return
    evaluate_models(args, health, root, directories, chosen)


def main():
    if os.name != "posix":
        raise RuntimeError(
            "Training orchestration requires Linux/Colab; inspect and CPU inference use run_liveness_notebook.py"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Project-relative or absolute TOML")
    parser.add_argument(
        "--data-root", type=Path, help="Override configured liveness data"
    )
    parser.add_argument(
        "--trials-root", type=Path, help="Override configured camera trials"
    )
    parser.add_argument(
        "--detector-model", type=Path, help="Override configured detector"
    )
    parser.add_argument("--resume-control", type=Path)
    parser.add_argument("--resume-full", type=Path)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--screening-epochs", type=int, choices=(15, 25))
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Verify actual checkpoints on CPU and print skip/resume/new decisions",
    )
    parser.add_argument(
        "--initialize-arms",
        nargs="+",
        help="Explicitly authorize a new attempt only where no checkpoint or saved weights exist",
    )
    parser.add_argument("--reference-runs-root", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--loader-workers", type=int)
    parser.add_argument("--benchmark-loader-workers", action="store_true")
    parser.add_argument("--loader-benchmark", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--reviewed-cache-view", type=Path)
    parser.add_argument("--pipeline-benchmark", type=Path)
    parser.add_argument("--training-batch-size", type=int)
    parser.add_argument("--batch-start-epoch", type=int)
    parser.add_argument("--batch-benchmark", type=Path)
    parser.add_argument(
        "--gpu-augmentation", action=argparse.BooleanOptionalAction, default=None
    )

    args = parser.parse_args()
    args.project_root, args.run_root = (
        args.project_root.resolve(),
        args.run_root.resolve(),
    )
    if args.inspect_only:
        print_resume_plan(args)
        return

    lock_path = args.project_root / "transfer/training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = check_producer_available(args.run_root, allow_handoff=True)
        job_id = (
            prior["job_id"]
            if prior
            else datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
        )
        health = TrainingHealth(args.run_root, job_id)
        health.state["hostname"] = socket.gethostname()
        launch = args.run_root / "launch-source" / job_id
        launch.mkdir(parents=True, exist_ok=False)
        for name in (
            "run_liveness_comparison.py",
            "liveness_training_health.py",
            "release/package_liveness_release.py",
        ):
            shutil.copyfile(Path(__file__).with_name(name), launch / name)
        (launch / "source-sha256.json").write_text(
            json.dumps({path.name: sha(path) for path in launch.glob("*.py")}, indent=2)
        )
        native_out, native_err = sys.stdout, sys.stderr
        log = args.run_root / (job_id + "-console.log")
        sys.stdout = DurableTrainingOutput(native_out, log, health, progress=True)
        sys.stderr = DurableTrainingOutput(native_err, log, health)
        try:
            run(args, health)
        except BaseException as error:
            # Preserve the primary exception even when the Google Drive writer fails.
            sys.stdout, sys.stderr = native_out, native_err
            try:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
                health.finish(error=error)
            except OSError as persistence_error:
                print(
                    "Could not persist failure notification:",
                    repr(persistence_error),
                    file=native_err,
                )
            traceback.print_exc()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = native_out, native_err


if __name__ == "__main__":
    main()
