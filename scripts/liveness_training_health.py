"""Publish liveness workflow progress without a terminal or background watcher."""

import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def runtime_loader_workers(
    requested=None, *, quota_file=Path("/sys/fs/cgroup/cpu.max")
):
    """Cap the requested loader count by process affinity and the visible CPU quota.

    Twelve is the measured default ceiling, not an assumed CPU allocation.
    Zero explicitly keeps DataLoader work in the main process.
    """
    if requested is not None and (type(requested) is not int or requested < 0):
        raise ValueError("Loader workers must be a nonnegative integer")
    available = os.process_cpu_count() or 1
    if quota_file.is_file():
        quota, period = quota_file.read_text().split()
        if quota != "max":
            available = min(available, max(1, math.ceil(int(quota) / int(period))))
    return min(12 if requested is None else requested, available)


def utc_text(seconds):
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def run_logged_command(command):
    """Stream child progress through the durable writer without requiring fileno."""
    with subprocess.Popen(
        [str(value) for value in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def check_producer_available(run_root, *, clock=time.time, allow_handoff=False):
    path = Path(run_root) / "training-health.json"
    if not path.exists():
        return None
    prior = json.loads(path.read_text())
    updated = datetime.fromisoformat(prior["updated_at"]).timestamp()
    handoff = (
        allow_handoff
        and prior.get("pid") == os.getpid()
        and prior.get("phase") == "ready-for-training"
    )
    if prior["state"] == "running" and clock() - updated < 300 and not handoff:
        raise RuntimeError(
            "A recent producer is already recorded; inspect it before relaunching"
        )
    return prior if handoff else None


def checkpoint_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_epoch_checkpoints(directory, arm, epochs):
    """Check actual saved optimization state before declaring retention complete."""
    # Keep bootstrap health reporting independent of the optional ML runtime.
    import torch

    directory = Path(directory)
    for epoch in range(1, epochs + 1):
        receipt = json.loads((directory / f"epoch-{epoch:03d}.json").read_text())
        name = f"epoch-{epoch:03d}-training.pt"
        checkpoint = directory / name
        if (
            receipt["epoch"] != epoch
            or receipt["checkpoint"] != name
            or checkpoint.stat().st_size != receipt["bytes"]
            or checkpoint_sha256(checkpoint) != receipt["sha256"]
        ):
            raise ValueError(f"Archived epoch checksum/receipt mismatch: {epoch}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required = (
            "state_dict",
            "optimizer",
            "torch_rng",
            "cuda_rng",
            "run",
            "updates",
            "resume_epoch",
            "progress",
            "epoch",
        )
        if (
            any(key not in saved for key in required)
            or saved["epoch"] != epoch
            or saved["resume_epoch"] != epoch + 1
            or saved["progress"] is not None
            or saved["run"]["arm"] != arm
            or receipt["inference_metadata"]["epoch"] != epoch
        ):
            raise ValueError(f"Incomplete archived optimization state: {epoch}")
        predictions = receipt.get("validation_predictions")
        if predictions is not None:
            name = f"epoch-{epoch:03d}-validation.csv"
            path = directory / name
            if (
                predictions["file"] != name
                or path.stat().st_size != predictions["bytes"]
                or checkpoint_sha256(path) != predictions["sha256"]
            ):
                raise ValueError(
                    f"Archived validation checksum/receipt mismatch: {epoch}"
                )


class TrainingHealth:
    def __init__(self, run_root, job_id, *, clock=time.time, publish_seconds=60):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock, self.publish_seconds = clock, publish_seconds
        self.owner_pid = os.getpid()
        self.last_published = float("-inf")
        self.checkpoint = None
        self.state = {
            "schema": "liveness-training-health-v1",
            "job_id": job_id,
            "pid": self.owner_pid,
            "state": "running",
            "phase": "preparing",
            "started_at": utc_text(clock()),
            "completed_arms": {},
            "evaluations_complete": False,
            "checkpoint_interval_seconds": 300,
        }
        self.publish(force=True)

    def phase(self, phase, *, arm=None, output=None):
        if arm is not None and arm != self.state.get("arm"):
            for name in (
                "epoch",
                "samples",
                "total_samples",
                "last_progress_at",
                "_progress_key",
                "completed_epoch",
                "total_updates",
                "epoch_optimizer_updates",
            ):
                self.state.pop(name, None)
        self.state.update(phase=phase, phase_started_at=utc_text(self.clock()))
        if arm is not None:
            self.state["arm"] = arm
        if output is not None:
            self.state.pop("checkpoint_modified_at", None)
            self.checkpoint = Path(output) / "last-training.pt"
        self.publish(force=True)
        # Notebook observers consume phase changes without reading personal paths.
        print(
            json.dumps(
                {"phase": "workflow-phase", "step": phase, "arm": self.state.get("arm")}
            ),
            flush=True,
        )

    def record(self, line):
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            if line.startswith("Full extraction: "):
                try:
                    self.state["restored_files"] = int(line.split()[2])
                except (ValueError, IndexError):
                    return
                self.state["last_progress_at"] = utc_text(self.clock())
                self.publish()
            return
        if not isinstance(record, dict):
            return
        if record.get("phase") == "training":
            key = (self.state.get("arm"), record["epoch"], record["samples"])
            if key != self.state.get("_progress_key"):
                self.state.update(
                    _progress_key=key,
                    last_progress_at=utc_text(self.clock()),
                    epoch=record["epoch"],
                    samples=record["samples"],
                    total_samples=record["total_samples"],
                    epoch_optimizer_updates=record["optimizer_updates"],
                )
        elif "arm" in record and "total_updates" in record:
            self.state.update(
                arm=record["arm"],
                epoch=record["epoch"],
                completed_epoch=record["epoch"],
                total_updates=record["total_updates"],
                last_progress_at=utc_text(self.clock()),
            )
            if "train_samples" in record:
                self.state.update(
                    samples=record["train_samples"],
                    total_samples=record["train_samples"],
                )
            self.state.pop("epoch_optimizer_updates", None)
            # Epoch completion must not leave a throttled, partial-batch snapshot on Drive.
            self.publish(force=True)
            return
        elif "materialized_rows" in record or "packed_rows" in record:
            count = record.get("materialized_rows", record.get("packed_rows"))
            self.state.update(
                cached_samples=count,
                cache_total_samples=record["total_rows"],
                last_progress_at=utc_text(self.clock()),
            )
        else:
            return
        self.publish()

    def publish(self, *, force=False):
        if os.getpid() != self.owner_pid:
            return
        now = self.clock()
        if not force and now - self.last_published < self.publish_seconds:
            return
        if self.checkpoint is not None:
            self.state["checkpoint_relative_path"] = self.checkpoint.relative_to(
                self.root
            ).as_posix()
            if self.checkpoint.exists():
                self.state["checkpoint_modified_at"] = utc_text(
                    self.checkpoint.stat().st_mtime
                )
        payload = {
            key: value for key, value in self.state.items() if not key.startswith("_")
        }
        payload["updated_at"] = utc_text(now)
        # Close each write so Google Drive can commit it. Atomic replacement changes
        # Google Drive file IDs; the external monitor addresses this file by name.
        with (self.root / "training-health.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        self.last_published = now

    def finish(self, *, error=None):
        self.state.update(
            state="failed" if error else "complete",
            phase="failed" if error else "complete",
            finished_at=utc_text(self.clock()),
        )
        if error:
            self.state["error"] = f"{type(error).__name__}: {error}"
        self.publish(force=True)


class DurableTrainingOutput:
    """Tee stdout/stderr to a closed-per-line Google Drive log and the native local FD."""

    def __init__(self, original, log_path, health, *, progress=False):
        self.original, self.log_path, self.health = original, Path(log_path), health
        self.progress = progress
        self.pending = ""
        self.owner_pid = os.getpid()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, text):
        result = self.original.write(text)
        if os.getpid() != self.owner_pid:
            return result
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if self.progress:
                self.health.record(line)
        return result

    def flush(self):
        self.original.flush()
        if self.pending and os.getpid() == self.owner_pid:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(self.pending)
            self.pending = ""


class InferenceProgress:
    """Report counted inference work without touching tensors or predictions."""

    def __init__(self, total, batch_size, *, unit="images"):
        self.total, self.batch_size, self.unit = (
            total,
            batch_size,
            unit,
        )
        self.samples = 0
        self.started = self.last = time.monotonic()

    def __call__(self):
        self.samples = min(self.total, self.samples + self.batch_size)
        now = time.monotonic()
        if now - self.last >= 5 or self.samples == self.total:
            print(
                json.dumps(
                    {
                        "phase": "evaluation",
                        "samples": self.samples,
                        "total_samples": self.total,
                        "unit": self.unit,
                        "elapsed_seconds": now - self.started,
                    }
                ),
                flush=True,
            )
            self.last = now
