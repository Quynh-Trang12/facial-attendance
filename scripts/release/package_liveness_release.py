"""Export a hashed transfer allowlist and explicitly relocate bound run metadata.

No dataset images, model tensors, optimizer state, epoch cursor or RNG state are
rewritten. Original cache/checkpoints remain separate from relocated copies.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch

from facial_attendance.liveness.dataset_files import (
    compatible_sources,
    source_transitions,
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def binding_relative(path, root):
    """Read recorded paths using the producing OS's path syntax."""
    kind = PureWindowsPath if PureWindowsPath(root).drive else PurePosixPath
    return Path(*kind(path).relative_to(kind(root)).parts)


def relocate_binding(binding, root):
    root = Path(root).resolve()

    result = copy.deepcopy(binding)
    for field in ("inventory", "annotations"):
        relative = binding_relative(binding[field], binding["root"])
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or sha(target) != binding[field + "_sha256"]:
            raise ValueError("Relocated data integrity mismatch: " + field)
        result[field] = str(target)
    result["root"] = str(root)
    return result


def tensor_tree_equal(first, second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            tensor_tree_equal(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            tensor_tree_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def screening_extension_sources(repository, original, saved):
    """Permit only the reviewed epoch guard/report edit, never numerical edits."""
    source = Path(repository) / "src/facial_attendance/liveness"
    current = {
        path.relative_to(source).as_posix(): sha(path) for path in source.rglob("*.py")
    }
    previous = saved["run"]["source_sha256"]
    baseline = current
    if previous.keys() != current.keys():
        candidates = [
            scope["before"]
            for transition in source_transitions(repository)
            for scope in transition["scopes"]
            if scope["after"] == current and scope["before"].keys() == previous.keys()
        ]
        baseline = next(
            (candidate for candidate in candidates if candidate == previous),
            candidates[0] if candidates else current,
        )
    if previous.keys() != baseline.keys():
        raise ValueError("Screening extension cannot add or remove numerical modules")
    for name, digest in previous.items():
        old = Path(original) / "source" / name
        if sha(old) != digest:
            raise ValueError("Producing source snapshot changed: " + name)
        if digest == baseline[name]:
            continue
        # These two adapters are not imported by the training/data graph.
        # Their physical-inference changes remain recorded in the migration.
        if name in (
            "inference/liveness_adapter.py",
            "evaluation/presentation_evaluation.py",
        ):
            continue
        if name not in (
            "full_training.py",
            "source_protocol_training.py",
            "aenet_training.py",
            "experiments/aenet.py",
        ):
            raise ValueError("Unreviewed numerical source change: " + name)
        raise ValueError("Unreviewed source change beyond its guard/report: " + name)
    return current


def extend_screening(repository, original, output, epochs):
    """Extend an explicitly supported epoch budget without changing optimization state."""
    original, output = Path(original), Path(output)
    checkpoint = original / "last-training.pt"
    original_sha = sha(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run = saved["run"]
    previous_epochs = run["training_config"]["epochs"]
    if (previous_epochs, epochs) not in ((15, 25),) or run["phase"] != "screening":
        raise ValueError(
            "Only 15-to-25 continuation is supported; never shorten an existing run"
        )
    required = (
        "state_dict",
        "optimizer",
        "torch_rng",
        "cuda_rng",
        "updates",
        "epoch",
        "resume_epoch",
        "progress",
        "best_epoch",
        "best_key",
    )
    if any(key not in saved for key in required):
        raise ValueError("Cannot extend incomplete optimization state")
    sources = screening_extension_sources(repository, original, saved)
    migrated = dict(saved)
    migrated["run"] = copy.deepcopy(run)
    migrated["run"]["training_config"]["epochs"] = epochs
    migrated["run"]["protocol"]["epochs"] = epochs
    migrated["run"]["protocol"]["stopping"] = (
        f"Exactly {epochs} epochs; validation-only screening"
    )
    migrated["run"]["source_sha256"] = sources
    output.mkdir(parents=True, exist_ok=False)
    torch.save(migrated, output / "last-training.pt")
    reloaded = torch.load(
        output / "last-training.pt", map_location="cpu", weights_only=True
    )
    if not tensor_tree_equal(migrated, reloaded) or sha(checkpoint) != original_sha:
        raise ValueError(
            "Epoch extension changed saved state or its original checkpoint"
        )
    for name in (
        "liveness-inference.pt",
        "multitask-training.pt",
        "training-history.jsonl",
    ):
        shutil.copyfile(original / name, output / name)
        if sha(original / name) != sha(output / name):
            raise ValueError("Extension changed selected weights/history: " + name)
    shutil.copytree(
        Path(repository) / "src/facial_attendance/liveness",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    write_json(output / "training-config.json", migrated["run"])
    write_json(
        output / "epoch-extension.json",
        {
            "schema": "liveness-screening-extension-v1",
            "arm": run["arm"],
            "from_epochs": previous_epochs,
            "to_epochs": epochs,
            "original_checkpoint_sha256": original_sha,
            "extended_checkpoint_sha256": sha(output / "last-training.pt"),
            "preserved_optimization_state": list(required),
            "state_bitwise_equal_after_reload": True,
            "updates": saved["updates"],
            "resume_epoch": saved["resume_epoch"],
            "producing_source_sha256": run["source_sha256"],
            "extended_source_sha256": sources,
        },
    )
    return output


def export(repository, cache, resume, benchmark, output):
    repository, cache, resume = map(Path.resolve, (repository, cache, resume))
    receipt = json.loads((cache / "geometric-cache.json").read_text())
    if (
        not receipt["completed"]
        or sha(cache / "geometric-targets.npy") != receipt["array_sha256"]
    ):
        raise ValueError("Completed, unchanged FP32 cache required")
    saved = torch.load(
        resume / "last-training.pt", map_location="cpu", weights_only=True
    )
    root = Path(receipt["bindings"]["data"]["root"])
    mapping = {}

    def add(source, target):
        source = Path(source).resolve()
        if not source.is_file() or target in mapping:
            raise ValueError("Missing or duplicate transfer input: " + str(source))
        mapping[target] = {
            "source": str(source),
            "sha256": sha(source),
            "bytes": source.stat().st_size,
        }

    for folder in ("src", "scripts", "tests", "configs"):
        for source in sorted((repository / folder).rglob("*")):
            if (
                source.is_file()
                and source.suffix in (".py", ".toml")
                and "__pycache__" not in source.parts
            ):
                add(source, source.relative_to(repository).as_posix())
    for name in (
        ".python-version",
        "requirements.txt",
        "requirements-cuda.txt",
        "requirements.lock",
    ):
        add(repository / name, name)
    for binding in (saved["run"]["data"]["train"], saved["run"]["data"]["validation"]):
        for field in ("inventory", "annotations"):
            source = Path(binding[field])
            target = (
                "data/raw/liveness/celeba-spoof-full/"
                + source.relative_to(root).as_posix()
            )
            if target not in mapping:
                add(source, target)
    test = Path(saved["run"]["data"]["test_annotations"])
    add(
        test, "data/raw/liveness/celeba-spoof-full/" + test.relative_to(root).as_posix()
    )
    for name in (
        "depth-teacher.pt",
        "reflection-teacher.pt",
        "resnet18-imagenet1k-initialization.pth",
    ):
        add(repository / "artifacts/pretrained" / name, "artifacts/pretrained/" + name)
    for name in ("face_ind.txt", "triangles.txt"):
        add(
            repository / "artifacts/pretrained/prnet-tensorflow" / name,
            "artifacts/pretrained/prnet-tensorflow/" + name,
        )
    add(
        repository / "artifacts/detectors/face_detection_yunet_2023mar.onnx",
        "artifacts/detectors/face_detection_yunet_2023mar.onnx",
    )
    for source in (repository / "data/liveness/camera-trials").glob("*"):
        if source.is_file() and source.suffix.lower() == ".mp4":
            add(source, source.relative_to(repository).as_posix())
    for name in ("geometric-targets.npy", "geometric-cache.json"):
        add(cache / name, "transfer/original-cache/" + name)
    for name in (
        "last-training.pt",
        "liveness-inference.pt",
        "multitask-training.pt",
        "training-history.jsonl",
        "training-config.json",
    ):
        add(resume / name, "transfer/original-control/" + name)
    for source in sorted((resume / "source").rglob("*.py")):
        add(
            source,
            "transfer/original-control/source/"
            + source.relative_to(resume / "source").as_posix(),
        )
    add(benchmark, "transfer/optimization-benchmark.json")
    report = {
        "schema": "liveness-colab-transfer-v1",
        "files": mapping,
        "source_transitions": source_transitions(repository),
        "bytes": sum(row["bytes"] for row in mapping.values()),
        "dataset_images": "Existing Drive archive; extracted bytes must match the transferred inventory",
        "checkpoint_epoch": saved["epoch"],
        "resume_epoch": saved.get("resume_epoch"),
        "completed_samples_in_epoch": (saved.get("progress") or {}).get("samples", 0),
        "optimizer_updates": saved["updates"],
    }
    write_json(output, report)

    print(json.dumps({key: value for key, value in report.items() if key != "files"}))


def verified_loader_change(saved, workers, benchmark_path):
    """Allow execution-only worker changes after real-input parity measurement."""
    if workers is None:
        return None
    if workers < 1 or benchmark_path is None:
        raise ValueError(
            "Worker migration requires a positive count and loader benchmark"
        )
    report = json.loads(Path(benchmark_path).read_text())
    config, data = saved["run"]["training_config"], saved["run"]["data"]
    rows = report["rows"]
    required = {config["workers"], workers}
    if (
        not required.issubset({row["workers"] for row in rows})
        or any(
            not row["inputs_bitwise_equal"] or row["batch_size"] != config["batch_size"]
            for row in rows
        )
        or len({row["input_target_sha256"] for row in rows}) != 1
        or report["seed"] != config["seed"]
        or report["precision"] != "FP32"
        or report["training_annotations_sha256"] != data["train"]["annotations_sha256"]
        or report["inventory_sha256"] != data["inventory_sha256"]
        or not compatible_sources(
            report["source_sha256"], saved["run"]["source_sha256"]
        )
    ):
        raise ValueError("Loader benchmark does not prove unchanged inputs and recipe")
    return {
        "old_workers": config["workers"],
        "new_workers": workers,
        "benchmark_sha256": sha(benchmark_path),
        "input_tensors_bitwise_equal": True,
    }


def verified_pipeline_change(saved, sources, changed, benchmark_path):
    permitted = {
        "decoded_cache.py",
        "full_training.py",
        "source_protocol_training.py",
        "aenet_training.py",
        "device_augmentation.py",
        "image_augmentation.py",
    }
    if saved["run"]["source_sha256"] != sources and compatible_sources(
        saved["run"]["source_sha256"], sources
    ):
        return None
    if not changed:
        return None
    if benchmark_path is None or not changed.issubset(permitted):
        raise ValueError("Unverified input pipeline source changes: " + repr(changed))
    report = json.loads(Path(benchmark_path).read_text())
    previous = saved["run"]["source_sha256"]
    snapshots = [report["original_source_sha256"]]
    parent = Path(benchmark_path).resolve().parent
    for ancestor in report.get("source_ancestors", []):
        path = (parent / ancestor["file"]).resolve()
        if (
            not path.is_relative_to(parent)
            or ancestor["field"] not in ("source_sha256", "original_source_sha256")
            or sha(path) != ancestor["sha256"]
        ):
            raise ValueError("Invalid source ancestry evidence")
        snapshots.append(json.loads(path.read_text())[ancestor["field"]])
    if (
        not compatible_sources(report["source_sha256"], sources)
        or not any(
            all(snapshot.get(name) == previous.get(name) for name in changed)
            for snapshot in snapshots
        )
        or not report["training_input_target_bitwise_equal"]
        or not report["validation_input_bitwise_equal"]
        or report["optimizer_state_bitwise_equal_by_arm"]
        != {"binary-classification": True, "aenet-full": True}
        or report["precision"] != "FP32"
        or report.get("base_batch_size", report.get("batch_size"))
        != saved["run"]["training_config"]["batch_size"]
        or report["inventory_sha256"] != saved["run"]["data"]["inventory_sha256"]
    ):
        raise ValueError(
            "Pipeline benchmark does not prove unchanged training and validation"
        )
    return {
        "benchmark_sha256": sha(benchmark_path),
        "verified_source_changes": sorted(changed),
    }


def bind_geometric_cache(repository, output):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    original = repository / "transfer/geometric-cache"
    if not original.is_dir():
        # Retain read compatibility for independently stored execution snapshots.
        original = repository / "transfer/original-cache"
    receipt = json.loads((original / "geometric-cache.json").read_text())
    if (
        not receipt["completed"]
        or sha(original / "geometric-targets.npy") != receipt["array_sha256"]
    ):
        raise ValueError("Completed unchanged geometric cache required")
    receipt["bindings"]["data"] = relocate_binding(
        receipt["bindings"]["data"], repository / "data/raw/liveness/celeba-spoof-full"
    )
    output.mkdir(parents=True, exist_ok=False)
    os.link(original / "geometric-targets.npy", output / "geometric-targets.npy")
    write_json(output / "geometric-cache.json", receipt)
    print(
        json.dumps(
            {
                "result": "PASS",
                "geometry_cache": output.name,
                "checkpoint_migrated": False,
            }
        )
    )


def relocate(
    repository,
    inventory_path,
    output,
    resume=None,
    loader_workers=None,
    loader_benchmark=None,
    pipeline_benchmark=None,
    training_batch_size=None,
    batch_start_epoch=None,
    batch_benchmark=None,
    gpu_augmentation=None,
):
    repository, output = repository.resolve(), output.resolve()
    inventory = json.loads(inventory_path.read_text())
    for name, row in inventory["files"].items():
        path = (repository / name).resolve()
        if not path.is_relative_to(repository) or sha(path) != row["sha256"]:
            raise ValueError("Transferred file mismatch: " + name)
    original = (
        Path(resume).resolve()
        if resume is not None
        else repository / "transfer/original-control"
    )
    saved = torch.load(
        original / "last-training.pt", map_location="cpu", weights_only=True
    )
    loader_change = verified_loader_change(saved, loader_workers, loader_benchmark)
    receipt = json.loads(
        (repository / "transfer/original-cache/geometric-cache.json").read_text()
    )
    benchmark = json.loads(
        (repository / "transfer/optimization-benchmark.json").read_text()
    )
    runs = benchmark.get("training_step_runs", [])
    if len(runs) != 12 or not all(
        row.get("state_bitwise_equal")
        and row.get("optimizer_bitwise_equal")
        and row.get("binary_loss_equal")
        for row in runs
    ):
        raise ValueError("Actual old/new optimizer-loop equivalence benchmark required")
    source = repository / "src/facial_attendance/liveness"
    sources = {
        path.relative_to(source).as_posix(): sha(path) for path in source.rglob("*.py")
    }
    old_sources = saved["run"]["source_sha256"]
    changed = {
        name
        for name in old_sources.keys() | sources.keys()
        if old_sources.get(name) != sources.get(name)
    }
    pipeline_change = verified_pipeline_change(
        saved,
        sources,
        changed
        - {"multitask_model.py", "semantic_experiment.py", "multitask_training.py"},
        pipeline_benchmark,
    )
    for name, digest in old_sources.items():
        if sha(original / "source" / name) != digest:
            raise ValueError("Original executed source changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    data_root = repository / "data/raw/liveness/celeba-spoof-full"
    rebound_cache = copy.deepcopy(receipt)
    rebound_cache["bindings"]["data"] = relocate_binding(
        receipt["bindings"]["data"], data_root
    )
    cache = output / "geometric-cache"
    cache.mkdir()
    # Both paths are read-only inputs after preparation; link avoids duplicating
    # the 1.3 GiB array. The runtime workflow copies into its own writable output.
    os.link(
        repository / "transfer/original-cache/geometric-targets.npy",
        cache / "geometric-targets.npy",
    )
    write_json(cache / "geometric-cache.json", rebound_cache)
    migrated = dict(saved)
    run = copy.deepcopy(saved["run"])
    run["data"]["root"] = str(data_root)
    run["data"]["inventory"] = str(
        data_root
        / binding_relative(
            saved["run"]["data"]["inventory"], saved["run"]["data"]["root"]
        )
    )
    for split in ("train", "validation"):
        run["data"][split] = relocate_binding(saved["run"]["data"][split], data_root)
    test = data_root / "metas/intra_test/test_label.json"
    if sha(test) != run["data"]["test_annotations_sha256"]:
        raise ValueError("Test split assignment changed")
    run["data"]["test_annotations"] = str(test)
    if loader_change is not None:
        run["training_config"]["workers"] = loader_workers
    run["training_config"].setdefault("gpu_augmentation", False)
    if gpu_augmentation is not None:
        if pipeline_benchmark is None:
            raise ValueError(
                "Device augmentation requires its real input/optimizer parity evidence"
            )
        report = json.loads(Path(pipeline_benchmark).read_text())
        if (
            not compatible_sources(report["source_sha256"], sources)
            or not report.get("device_augmentation_bitwise_equal")
            or report["optimizer_state_bitwise_equal_by_arm"]
            != {"binary-classification": True, "aenet-full": True}
        ):
            raise ValueError(
                "Device augmentation source or optimizer parity is unverified"
            )
        run["training_config"]["gpu_augmentation"] = gpu_augmentation
    run["training_config"].setdefault("large_batch_size", None)
    run["training_config"].setdefault("large_batch_start_epoch", 1)
    if training_batch_size is not None:
        if batch_benchmark is None or batch_start_epoch is None:
            raise ValueError(
                "Larger batch requires the actual GPU benchmark and an epoch boundary"
            )
        report = json.loads(Path(batch_benchmark).read_text())
        rows = [
            row for row in report["rows"] if row["batch_size"] == training_batch_size
        ]
        if (
            report["precision"] != "FP32"
            or {row["arm"] for row in rows} != {"binary-classification", "aenet-full"}
            or not all(row["finite_loss_and_gradients"] for row in rows)
            or training_batch_size < run["training_config"]["batch_size"]
            or batch_start_epoch <= saved["epoch"]
            or batch_start_epoch > run["training_config"]["epochs"]
        ):
            raise ValueError(
                "Batch schedule is not verified or would rewrite completed training"
            )
        run["training_config"].update(
            large_batch_size=training_batch_size,
            large_batch_start_epoch=batch_start_epoch,
        )
    run["source_sha256"] = sources
    run["geometry_source_bindings"] = rebound_cache["bindings"]
    run["geometry_cache_manifest_sha256"] = sha(cache / "geometric-cache.json")
    migrated["run"] = run
    arm = saved["run"]["arm"]
    if arm not in ("binary-classification", "aenet-full"):
        raise ValueError("Unsupported training arm in resume checkpoint")
    checkpoint_directory = output / arm
    checkpoint_directory.mkdir()
    torch.save(migrated, checkpoint_directory / "last-training.pt")
    reloaded = torch.load(
        checkpoint_directory / "last-training.pt", map_location="cpu", weights_only=True
    )
    for name in saved.keys() - {"run"}:
        if not tensor_tree_equal(saved[name], reloaded[name]):
            raise ValueError("Migration changed training state: " + name)
    for name in (
        "liveness-inference.pt",
        "multitask-training.pt",
        "training-history.jsonl",
    ):
        shutil.copyfile(original / name, checkpoint_directory / name)
    proof = {
        "result": "PASS",
        "old_checkpoint_sha256": sha(original / "last-training.pt"),
        "new_checkpoint_sha256": sha(checkpoint_directory / "last-training.pt"),
        "transfer_manifest_sha256": sha(inventory_path),
        "optimization_benchmark_sha256": sha(
            repository / "transfer/optimization-benchmark.json"
        ),
        "cache_array_sha256": receipt["array_sha256"],
        "verified_source_changes": sorted(changed),
        "weights_optimizer_rng_cursor_unchanged": True,
        "original_run": saved["run"],
        "scope": "Explicit path/source-metadata migration. Cross-GPU bitwise training equivalence is not claimed.",
    }
    if loader_change is not None:
        proof["loader_runtime_change"] = loader_change
    if pipeline_change is not None:
        proof["input_pipeline_change"] = pipeline_change
    if training_batch_size is not None:
        proof["batch_schedule_change"] = {
            "base_batch_size": run["training_config"]["batch_size"],
            "large_batch_size": training_batch_size,
            "first_large_batch_epoch": batch_start_epoch,
            "benchmark_sha256": sha(batch_benchmark),
            "quality_equivalence_claimed": False,
        }
    write_json(output / "migration.json", proof)
    print(
        json.dumps(
            {
                "result": "PASS",
                "resume": str(checkpoint_directory),
                "geometry_cache": str(cache),
                "resume_epoch": reloaded.get("resume_epoch"),
                "updates": reloaded["updates"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "relocate", "cache"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--loader-workers", type=int)
    parser.add_argument("--loader-benchmark", type=Path)
    parser.add_argument("--pipeline-benchmark", type=Path)
    parser.add_argument("--training-batch-size", type=int)
    parser.add_argument("--batch-start-epoch", type=int)
    parser.add_argument("--batch-benchmark", type=Path)
    parser.add_argument(
        "--gpu-augmentation", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--inventory", "--manifest", dest="inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "cache":
        bind_geometric_cache(args.repository, args.output)
        return
    if args.mode == "export":
        if not all((args.cache, args.resume, args.benchmark)):
            parser.error("export needs --cache, --resume and --benchmark")
        export(args.repository, args.cache, args.resume, args.benchmark, args.output)
    else:
        if not args.inventory:
            parser.error("relocate needs --inventory")
        relocate(
            args.repository,
            args.inventory,
            args.output,
            resume=args.resume,
            loader_workers=args.loader_workers,
            loader_benchmark=args.loader_benchmark,
            pipeline_benchmark=args.pipeline_benchmark,
            training_batch_size=args.training_batch_size,
            batch_start_epoch=args.batch_start_epoch,
            batch_benchmark=args.batch_benchmark,
            gpu_augmentation=args.gpu_augmentation,
        )


if __name__ == "__main__":
    main()
