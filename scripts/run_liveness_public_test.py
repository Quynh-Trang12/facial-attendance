"""Prepare the portable cache and score the public test with the validation winner."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from liveness_camera_inputs import selected_checkpoint
from liveness_training_health import TrainingHealth, runtime_loader_workers
from supervise_liveness_training_process import launch, monitor

# Only runtime staging uses POSIX process locks. Cache construction stays portable.
if os.name == "posix":
    import fcntl

    from stage_liveness_training_runtime import (
        copy_resumable,
        prepare_environment,
        rclone_cache_download,
        stage_verified_inputs,
    )


DATA = Path("data/raw/liveness/celeba-spoof-full")
INVENTORY = DATA / "audits/current-schema-2026-11-12/inventory.csv"
ANNOTATIONS = DATA / "metas/intra_test/test_label.json"
CACHE = Path("data/liveness/face-cache/faces/test")
BINDING_FIELDS = ("inventory_sha256", "annotations_sha256", "samples", "crop_recipe")
CAMERA_SCORER = "src/facial_attendance/liveness/evaluation/presentation_evaluation.py"


class CacheTransferProgress:
    """Update one notebook region with measured test-cache transfer bytes."""

    def __init__(self, total, status, *, initial=0):
        self.total, self.status, self.initial = total, status, initial
        self.done = initial
        self.started = self.last = time.monotonic()

    def put(self, count):
        self.done = min(self.total, self.done + count)
        now = time.monotonic()
        if now - self.last >= 5 or self.done == self.total:
            rate = (self.done - self.initial) / max(now - self.started, 0.001)
            eta = round((self.total - self.done) / rate) if rate else None
            remaining = (
                f"~{eta // 60}m {eta % 60:02d}s" if eta is not None else "pending"
            )
            self.status(
                f"Test cache: {self.done / 1e9:.2f}/{self.total / 1e9:.2f} GB "
                f"({100 * self.done / max(self.total, 1):.1f}%) | "
                f"copy ETA {remaining}; SHA-256 follows"
            )
            self.last = now


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_test_cache(data_root, receipt_path):
    """Stream one NPY to stdout, avoiding a second 10.1 GB local allocation."""
    # This CPU-only boundary uses the project's existing ML environment.
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import Resize

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from facial_attendance.liveness.datasets.source_dataset import ReproductionImages
    from facial_attendance.liveness.models.binary_model import (
        RESNET_ARCHITECTURE,
        face_transform,
    )

    torch.set_num_threads(1)
    source = ReproductionImages(
        data_root,
        data_root / "audits/current-schema-2026-11-12/inventory.csv",
        data_root / "metas/intra_test/test_label.json",
    )
    shape = (len(source), 224, 224, 3)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "|u1", "fortran_order": False, "shape": shape}
    )
    checksum = hashlib.sha256()
    content = header.getvalue()
    checksum.update(content)
    sys.stdout.buffer.write(content)
    size = len(content)
    resize = Resize((224, 224))
    transform = face_transform(architecture=RESNET_ARCHITECTURE)
    checked = 0
    for index in range(len(source)):
        image, _ = source[index]  # Verifies each original image and bounding-box SHA.
        pixels = np.asarray(resize(image), dtype=np.uint8)
        if index % max(1, len(source) // 512) == 0 or index == len(source) - 1:
            if not torch.equal(transform(image), transform(Image.fromarray(pixels))):
                raise ValueError("Original-image and cached-image preprocessing differ")
            checked += 1
        content = pixels.tobytes(order="C")
        checksum.update(content)
        size += len(content)
        sys.stdout.buffer.write(content)
        if (index + 1) % 2000 == 0:
            print(
                f"Test cache: {index + 1:,}/{len(source):,} "
                "images verified and streamed",
                file=sys.stderr,
            )
    sys.stdout.buffer.flush()
    binding = source.bindings()
    receipt = {
        "schema": "liveness-decoded-faces-v1",
        "data": {key: binding[key] for key in BINDING_FIELDS},
        "shape": list(shape),
        "dtype": "uint8",
        "sha256": checksum.hexdigest(),
        "bytes": size,
        "verification": {
            "original_image_and_box_sha256": "all rows",
            "tensor_parity_rows": checked,
            "tensor_parity": "exact",
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"Test cache complete: {len(source):,} images; receipt saved.", file=sys.stderr
    )


def stage_test_cache(source, project, status):
    """Copy the existing cache once; rebind paths only after checksum verification."""
    remote, local = (source / CACHE).resolve(), (project / CACHE).resolve()
    if remote == local:
        raise ValueError("Source and runtime test-cache directories must differ")
    local_receipt = local / "decoded-faces.json"
    if local_receipt.exists() and local_receipt.samefile(remote / "decoded-faces.json"):
        raise ValueError("Runtime receipt must not alias the persistent source receipt")
    receipt = json.loads((remote / "decoded-faces.json").read_text())
    if receipt["schema"] != "liveness-decoded-faces-v1" or receipt["dtype"] != "uint8":
        raise ValueError("Unsupported public-test cache")

    annotations = project / ANNOTATIONS
    count = len(json.loads(annotations.read_text()))
    expected = {
        "inventory_sha256": digest(project / INVENTORY),
        "annotations_sha256": digest(annotations),
        "samples": count,
        "crop_recipe": "celeba-spoof-client-integer-coordinates-ratio-first-v1",
    }
    if receipt["data"] != expected or receipt["shape"] != [count, 224, 224, 3]:
        raise ValueError(
            "Public-test cache differs from the frozen inventory, split or crop recipe"
        )

    release = json.loads((source / "configs/liveness-release.json").read_text())
    anchor = release["evaluation"]["test_cache_receipt"]
    if digest(remote / "decoded-faces.json") != anchor["sha256"]:
        raise ValueError("Public-test receipt differs from the published release")

    target = local / "decoded-faces.npy"
    local.mkdir(parents=True, exist_ok=True)
    verified = target.is_file() and digest(target) == receipt["sha256"]
    if not verified:
        partial = target.with_name(target.name + ".copying")
        prefix_bytes = partial.stat().st_size if partial.exists() else 0
        if (
            shutil.disk_usage(local).free
            < max(0, receipt["bytes"] - prefix_bytes) + 2**30
        ):
            raise ValueError(
                "Insufficient runtime SSD capacity for the public-test cache"
            )
        status("Loading only the public-test NPY cache; rclone uses up to 16 streams…")
        downloader = rclone_cache_download(remote)
        verified = (
            downloader.copy(
                remote / target.name,
                target,
                receipt["sha256"],
                CacheTransferProgress(receipt["bytes"], status),
            )
            if downloader
            else False
        )
        if not verified:
            status(
                "Using the Google Drive mount for a partial copy or rclone fallback; "
                "checksum verification remains required."
            )
            partial = target.with_name(target.name + ".copying")
            initial = partial.stat().st_size if partial.exists() else 0
            copy_resumable(
                remote / target.name,
                target,
                receipt["bytes"],
                CacheTransferProgress(receipt["bytes"], status, initial=initial),
            )
            if digest(target) != receipt["sha256"]:
                raise ValueError("Staged public-test cache checksum mismatch")

    receipt["data"] = {
        **expected,
        "root": str((project / DATA).resolve()),
        "inventory": str((project / INVENTORY).resolve()),
        "annotations": str(annotations.resolve()),
    }
    info = target.stat()
    receipt["array_identity"] = [
        str(target.resolve()),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]
    (local / "decoded-faces.json").write_text(json.dumps(receipt, indent=2) + "\n")
    status(f"Public-test cache verified: {count:,} images. No training cache loaded.")
    return local


def prepare(source, project, run_root, status):
    """Restore inference source and test inputs, excluding all training caches."""
    if os.name != "posix":
        raise RuntimeError(
            "Use Google Colab or Linux for runtime staging; "
            "standalone evaluators remain separate"
        )
    if source.resolve() == project.resolve():
        raise ValueError(
            "Runtime staging must not overwrite the persistent source project"
        )
    if (
        not (source / CACHE / "decoded-faces.json").is_file()
        or not (source / CACHE / "decoded-faces.npy").is_file()
    ):
        raise FileNotFoundError(
            "Build and upload data/liveness/face-cache/faces/test before starting evaluation"
        )

    release = json.loads((source / "configs/liveness-release.json").read_text())
    exact = {
        str(INVENTORY),
        str(ANNOTATIONS),
        ".python-version",
        "requirements.txt",
        "requirements-cuda.txt",
        "requirements.lock",
        "scripts/evaluation/score_liveness_image_split.py",
        "scripts/stage_liveness_training_runtime.py",
        "scripts/liveness_training_health.py",
        "scripts/supervise_liveness_training_process.py",
    }
    release["files"] = {
        name: dict(record)
        for name, record in release["files"].items()
        if name != CAMERA_SCORER and (name in exact or name.startswith("src/"))
    }
    for name, record in release["files"].items():
        if (
            not (source / record["delivery_path"]).is_file()
            and (source / name).is_file()
        ):
            record["delivery_path"] = name
    private = project / "evaluation-work"
    private.mkdir(parents=True, exist_ok=True)
    inventory = private / "evaluation-input-inventory.json"
    inventory.write_text(json.dumps(release, indent=2) + "\n")
    health = TrainingHealth(private, "evaluation-preparation")
    args = SimpleNamespace(
        project_root=project,
        run_root=private,
        source_root=source,
        release_inventory=inventory,
        prepare_only=False,
    )
    status("Verifying evaluation source and public-test metadata…")
    stage_verified_inputs(args, health)
    status("Checking the pinned Python 3.13 evaluation environment…")
    python = prepare_environment(args, health)
    cache = stage_test_cache(source, project, status)
    arm, checkpoint, checksum = selected_checkpoint(run_root)
    local = private / arm / checksum[:16] / "liveness-inference.pt"
    copy_resumable(checkpoint, local, checkpoint.stat().st_size)
    if digest(local) != checksum:
        raise ValueError("Copied inference checkpoint checksum mismatch")
    return python, cache, arm, local, checksum


def evaluation_input_signatures(source):
    """Verify public inputs and derive the signature used by older combined indexes."""

    release = json.loads((source / "configs/liveness-release.json").read_text())
    records = {
        name: record["sha256"]
        for name, record in release["files"].items()
        if name != CAMERA_SCORER
        and (
            name.startswith("src/")
            or name
            in (
                str(INVENTORY),
                str(ANNOTATIONS),
                "scripts/evaluation/score_liveness_image_split.py",
            )
        )
    }
    for name, expected in records.items():
        record = release["files"][name]
        path = source / record["delivery_path"]
        if not path.is_file():
            path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Evaluation input path escapes its project")
        if digest(path) != expected:
            raise ValueError(
                "Evaluation source or declared inputs changed: " + Path(name).name
            )
    # A renamed entry point does not change the bytes of its saved evaluation input.
    if "scripts/evaluation/score_liveness_image_split.py" in records:
        records["scripts/evaluate_liveness.py"] = records.pop(
            "scripts/evaluation/score_liveness_image_split.py"
        )
    records["test-cache-receipt"] = release["evaluation"]["test_cache_receipt"][
        "sha256"
    ]
    if digest(source / CACHE / "decoded-faces.json") != records["test-cache-receipt"]:
        raise ValueError("Public-test cache receipt changed")
    # Earlier public-test indexes included the unused camera scorer.
    camera_versions = (
        release["evaluation"].get("camera_evaluator", {}).get("compatible_results", [])
    )
    previous = [
        {
            **records,
            CAMERA_SCORER: version["source"].get(
                Path(CAMERA_SCORER).name, release["files"][CAMERA_SCORER]["sha256"]
            ),
        }
        for version in camera_versions
        if CAMERA_SCORER in release["files"]
    ] or [records]
    legacy_only = {
        # Preserve the historical fingerprint namespace after moving camera inputs.
        name.replace(
            "data/liveness/camera-trials/", "data/liveness-trials/", 1
        ): record["sha256"]
        for name, record in release["files"].items()
        if name.startswith(("data/liveness-trials/", "data/liveness/camera-trials/"))
        or name
        in (
            "scripts/evaluation/score_liveness_trial_inventory.py",
            "artifacts/detectors/face_detection_yunet_2023mar.onnx",
        )
    }
    legacy_only.update(
        release["evaluation"].get("legacy_camera_evaluation_bindings", {})
    )
    return tuple(
        hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        for binding in (
            records,
            *previous,
            *({**bound, **legacy_only} for bound in previous),
        )
    )


def observe_active_inference(output_root, status):
    """Follow an inference process that outlived its notebook cell."""
    if not Path("/proc").is_dir():
        return
    owners = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().decode().split("\0")
            programs = {Path(value).name for value in arguments[:3]}
            if "evaluation/score_liveness_image_split.py" not in programs:
                continue
            directory = Path(arguments[arguments.index("--output-dir") + 1])
            parent = int(
                next(
                    line.split()[1]
                    for line in (process / "status").read_text().splitlines()
                    if line.startswith("PPid:")
                )
            )
            if directory.resolve().is_relative_to(output_root.resolve()):
                owners[int(process.name)] = (parent, arguments, directory)
        except (OSError, UnicodeError, ValueError, StopIteration):
            continue
    roots = [(pid, row) for pid, row in owners.items() if row[0] not in owners]
    if len(roots) > 1:
        raise ValueError(
            "Multiple evaluation producers exist; no additional inference started"
        )
    for pid, (_, arguments, directory) in roots:
        status(
            "Following the existing evaluation process; no duplicate inference started."
        )
        log = directory.parent / (directory.name + ".log")
        try:
            monitor(arguments, log, pid, on_status=status)
        except FileNotFoundError:
            pass  # The producer completed between discovery and attachment.


def evaluate(source, project, run_root, status, *, output_root=None):
    """Serialize evaluation launches while allowing existing jobs to finish."""
    if os.name != "posix":
        raise ValueError(
            "Use Google Colab or Linux for orchestration; "
            "standalone evaluators remain available"
        )
    if source.resolve() == project.resolve():
        raise ValueError("Use a separate runtime working directory")
    private = project / "evaluation-work"
    private.mkdir(parents=True, exist_ok=True)
    with (private / "evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Evaluation is already running; no second orchestration started"
            ) from None
        return evaluate_frozen(
            source, project, run_root, status, output_root=output_root
        )


def evaluate_frozen(source, project, run_root, status, *, output_root=None):
    signature, *previous_signatures = evaluation_input_signatures(source)
    arm, _, checksum = selected_checkpoint(run_root)
    destination = run_root if output_root is None else Path(output_root)
    output = destination / "evaluation" / arm / checksum[:16]
    observe_active_inference(output, status)
    index_path = output / "evaluation-results.json"
    saved = {"checkpoint_sha256": checksum, "input_signature": signature, "results": {}}
    if index_path.is_file():
        saved = json.loads(index_path.read_text())
        if saved["checkpoint_sha256"] != checksum:
            raise ValueError("Evaluation result index checkpoint differs")
        if saved.get("input_signature") not in (signature, *previous_signatures):
            raise ValueError(
                "Saved public-test inputs or code differ; "
                "use a separate evaluation output"
            )

    directory = output / saved["results"].get("public-test", "public-test")
    if not directory.resolve().is_relative_to(output.resolve()):
        raise ValueError("Public-test result path escapes its output")
    result = directory / "evaluation.json"
    if result.is_file():
        if json.loads(result.read_text())["checkpoint_sha256"] != checksum:
            raise ValueError("Saved public-test result belongs to another checkpoint")
        if "public-test" not in saved["results"]:
            # Recover a producer that finished before its index was published.
            saved["results"]["public-test"] = str(directory.relative_to(output))
            index_path.write_text(json.dumps(saved, indent=2) + "\n")
        status("Saved public-test results reused; no cache download or GPU inference.")
        return output

    python, cache, prepared_arm, checkpoint, prepared_checksum = prepare(
        source, project, run_root, status
    )
    if (prepared_arm, prepared_checksum) != (arm, checksum):
        raise ValueError("Validation selection changed during public-test preparation")
    output.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = directory.with_name(directory.name + "-" + attempt)

    status("Running the public test with the frozen validation threshold.")
    log = output / (directory.name + ".log")
    launch(
        [
            str(python),
            "-u",
            str(project / "scripts/evaluation/score_liveness_image_split.py"),
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(project / DATA),
            "--inventory",
            str(project / INVENTORY),
            "--annotations",
            str(project / ANNOTATIONS),
            "--decoded-cache",
            str(cache),
            "--split",
            "test",
            "--workers",
            str(runtime_loader_workers()),
            "--batch-size",
            "256",
            "--output-dir",
            str(directory),
        ],
        log,
        on_status=status,
    )
    if (
        json.loads((directory / "evaluation.json").read_text())["checkpoint_sha256"]
        != checksum
    ):
        raise ValueError("New public-test result belongs to another checkpoint")

    saved["input_signature"] = signature
    saved["results"]["public-test"] = str(directory.relative_to(output))
    index_path.write_text(json.dumps(saved, indent=2) + "\n")
    status("Public-test predictions and metrics saved on Google Drive.")
    return output


def run_notebook(source, project, run_root, *, output_root=None):
    """Keep detailed diagnostics in files and publish them after the operation."""
    from IPython.display import Markdown, display

    status = display(Markdown("Checking frozen evaluation inputs…"), display_id=True)
    log = project / "evaluation-work/evaluation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            log.open("a", buffering=1) as stream,
            redirect_stdout(stream),
            redirect_stderr(stream),
        ):
            try:
                return evaluate(
                    source,
                    project,
                    run_root,
                    lambda message: status.update(Markdown(message)),
                    output_root=output_root,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(repr(error), file=stream)
                status.update(
                    Markdown(
                        "Evaluation stopped; existing models and results retained. "
                        "See evaluation.log."
                    )
                )
                raise RuntimeError(
                    "Evaluation failed. See evaluation-work/evaluation.log."
                ) from None
    finally:
        destination = (output_root or run_root) / "evaluation/input-preparation.log"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log, destination)
        except OSError:
            # Diagnostic publication must not replace the primary model/data failure.
            print(
                "Diagnostic publication failed; the local evaluation.log is retained.",
                file=sys.__stderr__,
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "build-test-cache writes binary NPY to stdout; "
            "redirect it to a file or transfer pipeline."
        ),
    )
    parser.add_argument("action", choices=("build-test-cache", "evaluate"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Writable results directory; --run-root remains the checkpoint input",
    )

    args = parser.parse_args()
    if args.action == "build-test-cache":
        if args.data_root is None or args.receipt is None:
            parser.error("build-test-cache requires --data-root and --receipt")
        if sys.stdout.isatty():
            parser.error("Redirect binary NPY stdout to a file or transfer pipeline")
        try:
            build_test_cache(args.data_root.resolve(), args.receipt.resolve())
        except (OSError, ValueError) as error:
            print(
                "Test-cache build failed: "
                + type(error).__name__
                + "; no complete cache was published.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    else:
        if (
            args.source_root is None
            or args.project_root is None
            or args.run_root is None
        ):
            parser.error(
                "evaluate requires --source-root, --project-root and --run-root"
            )
        project = args.project_root.resolve()
        log = project / "evaluation-work/evaluation.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                log.open("a", buffering=1) as stream,
                redirect_stdout(stream),
                redirect_stderr(stream),
            ):
                evaluate(
                    args.source_root.resolve(),
                    project,
                    args.run_root.resolve(),
                    lambda message: print(message, file=sys.__stdout__, flush=True),
                    output_root=args.output_root.resolve()
                    if args.output_root
                    else None,
                )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with log.open("a") as stream:
                print(repr(error), file=stream)
            raise SystemExit(
                "Evaluation stopped; inspect evaluation-work/evaluation.log"
            ) from None


if __name__ == "__main__":
    main()
