"""Keep the notebook execution attached while computation writes to a local file.

The training process never writes to a CLI websocket pipe. This does not keep
an idle runtime alive or prevent Google Colab from reclaiming its VM.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


class CacheStagingProgress:
    """Summarize staging for one notebook display, independently of the producer."""

    def __init__(self, command):
        source = Path(command[command.index("--prepared-cache-root") + 1])
        destination = Path(command[command.index("--decoded-cache") + 1])
        self.files = {}
        for split in ("train", "validation"):
            receipt = json.loads((source / split / "decoded-faces.json").read_text())
            names = [
                "decoded-faces.npy",
                *receipt.get("corrections", {}).get("files", {}),
            ]
            for name in names:
                if Path(name).name != name:
                    raise ValueError("Cache filename escapes its folder")
                self.files[(split, name)] = (
                    destination / split / name,
                    (source / split / name).stat().st_size,
                )
        self.downloaded = {}
        self.check = None
        self.previous_time = time.monotonic()
        self.previous_bytes = self.copied_bytes()

    def copied_bytes(self):
        total = 0
        for key, (target, expected) in self.files.items():
            sequential = target.with_name(target.name + ".copying")
            # Random-write rclone files may already have their full logical size.
            # Count reported transfer bytes, never their sparse file length.
            local = target if target.exists() else sequential
            try:
                count = local.stat().st_size
            except FileNotFoundError:
                # The producer may atomically rename a completed prefix now.
                count = (
                    target.stat().st_size
                    if target.exists()
                    else self.downloaded.get(key, 0)
                )
            total += min(expected, count)
        return total

    def record(self, event):
        key = (event.get("split"), event.get("file"))
        if event.get("cache_transfer") == "rclone-progress":
            self.downloaded[key] = event["copied_bytes"]
        if event.get("phase") == "cache-sha256":
            self.check = event
        elif event.get("phase") == "cache-copy" or event.get("cache_transfer") in (
            "rclone-start",
            "rclone-progress",
        ):
            self.check = None

    def message(self):
        now = time.monotonic()
        copied = self.copied_bytes()
        rate = max(0, copied - self.previous_bytes) / max(
            now - self.previous_time, 0.001
        )
        self.previous_bytes, self.previous_time = copied, now
        total = sum(size for _, size in self.files.values())
        prefix = f"Cache copied: {copied / 1e9:.2f}/{total / 1e9:.2f} GB ({100 * copied / max(total, 1):.1f}%)"
        if self.check:
            check = self.check
            done, size = check["checked_bytes"], check["total_bytes"]
            return (
                f"{prefix} | SHA-256: {check['split']}/{check['file']} "
                f"{100 * done / max(size, 1):.1f}% | check ETA {self.eta(size - done, check['bytes_per_second'])}"
            )
        if copied >= total:
            return (
                prefix + " | Finishing integrity checks and receipts; ETA unavailable"
            )
        return f"{prefix} | {rate / 1e6:.1f} MB/s | copy ETA {self.eta(total - copied, rate)}; checks follow"

    @staticmethod
    def eta(remaining, rate):
        if remaining <= 0:
            return "0s"
        if rate <= 0:
            return "unavailable (waiting for data)"
        minutes, seconds = divmod(round(remaining / rate), 60)
        return f"~{minutes}m {seconds:02d}s"


class WorkflowProgress:
    """Render observed workflow progress without changing training execution."""

    def __init__(self, command):
        evaluator = any(
            Path(str(value)).name
            in ("score_liveness_image_split.py", "score_liveness_trial_inventory.py")
            for value in command[:3]
        )
        self.mode = (
            "evaluation"
            if evaluator
            else "environment"
            if "--runtime-only" in command
            else "cache"
            if "--prepare-only" in command
            else "training"
        )
        self.validation_only = "--validation-only" in command
        self.step = "Verifying source and inputs"
        self.started = self.changed = time.monotonic()
        self.source = None
        self.arm = None
        self.epoch = None
        self.sample = None
        self.previous_sample = None
        self.rate = 0
        self.completed = {}
        self.last_validation = None
        self.last_epoch_at = {}
        self.epoch_seconds = {}
        self.target = (
            int(command[command.index("--screening-epochs") + 1])
            if "--screening-epochs" in command
            else 15
        )
        self.arms = []
        if "--arms" in command:
            for value in command[command.index("--arms") + 1 :]:
                if str(value).startswith("--"):
                    break
                self.arms.append(value)
        self.cache = (
            CacheStagingProgress(command)
            if self.mode != "environment"
            and "--prepared-cache-root" in command
            and "--decoded-cache" in command
            else None
        )

    def record(self, event):
        phase = event.get("phase")
        now = time.monotonic()
        if self.cache is not None:
            self.cache.record(event)
        if phase == "workflow-phase":
            self.step = event["step"]
            self.changed = now
            self.source = None
            if event.get("arm") != self.arm:
                self.sample = self.previous_sample = None
                self.rate = 0
            self.arm = event.get("arm")
        elif phase == "source-verification":
            self.source = event
        elif event.get("schema") == "liveness-resume-plan-v1":
            for item in event["arms"]:
                self.completed[item["arm"]] = item.get("completed_epochs", 0)
        elif phase == "resume-plan":
            self.completed[event["arm"]] = event.get("completed_epochs", 0)
        elif phase == "evaluation":
            self.step = "evaluation"
            self.sample = event
            self.changed = now
        elif phase == "training":
            self.step = "training"
            previous = self.previous_sample
            if previous and event["epoch"] == previous["epoch"]:
                elapsed = event["elapsed_seconds"] - previous["elapsed_seconds"]
                if elapsed > 0:
                    self.rate = max(0, event["samples"] - previous["samples"]) / elapsed
            else:
                self.rate = 0
            self.sample = self.previous_sample = event
            self.epoch = event["epoch"]
            self.changed = now
        elif "arm" in event and "total_updates" in event:
            arm = event["arm"]
            self.arm = arm
            if (
                arm in self.last_epoch_at
                and event["epoch"] == self.completed.get(arm, 0) + 1
            ):
                self.epoch_seconds[arm] = now - self.last_epoch_at[arm]
            self.last_epoch_at[arm] = now
            self.completed[arm] = event["epoch"]
            self.last_validation = event
            self.sample = self.previous_sample = None
            self.epoch = event["epoch"]
            self.step = "Epoch saved; waiting for the next training update"
            self.changed = now

    def message(self):
        elapsed = round(time.monotonic() - self.started)
        age = round(time.monotonic() - self.changed)
        if self.source:
            row = self.source
            return f"Verifying source and inputs: {row['completed']}/{row['total']} files | elapsed {elapsed}s | ETA unavailable (file sizes vary)"
        if self.cache is not None and self.step in (
            "staging-prepared-face-cache",
            "publishing-prepared-face-cache",
        ):
            return self.cache.message()
        if self.mode == "evaluation" and self.sample:
            row = self.sample
            done, total = row["samples"], row["total_samples"]
            rate = done / max(row["elapsed_seconds"], 0.001)
            return (
                f"Evaluation: {done:,}/{total:,} {row.get('unit', 'images')} "
                f"({100 * done / max(total, 1):.1f}%) | inference ETA "
                f"{CacheStagingProgress.eta(total - done, rate)} | last progress {age}s ago"
            )
        if self.sample:
            row = self.sample
            count, total = row["samples"], row["total_samples"]
            text = (
                f"**{self.arm or 'Training'} · epoch {row['epoch']}/{self.target}** | "
                f"images {count:,}/{total:,} ({100 * count / max(total, 1):.1f}%) | "
                f"training ETA {CacheStagingProgress.eta(total - count, self.rate)} (excludes validation/saving)"
            )
        else:
            text = f"**{self.step.replace('-', ' ').capitalize()}**"
            if self.arm:
                text += f" | {self.arm}"
            text += f" | elapsed {elapsed}s | phase ETA unavailable"
        if self.mode == "training" and self.arms:
            done = sum(
                min(self.target, self.completed.get(arm, 0)) for arm in self.arms
            )
            text += (
                f"\n\nComparison: {done}/{len(self.arms) * self.target} epochs saved"
            )
            seconds = self.epoch_seconds.get(self.arm)
            if seconds:
                remaining = self.target - self.completed.get(self.arm, 0)
                text += f" | configuration ETA {CacheStagingProgress.eta(remaining * seconds, 1)} (recent complete epoch)"
            else:
                text += " | configuration ETA pending a complete epoch timing"
        text += f" | last progress {age}s ago"
        if self.last_validation:
            row = self.last_validation
            text += f"\n\nLatest saved validation: {row['arm']}, epoch {row['epoch']} | ACER {100 * row['validation_ACER']:.2f}%"
        return text

    def finished(self, failed):
        if failed:
            return f"{self.mode.capitalize()} failed. See the diagnostic log; existing cache and checkpoints are retained."
        if self.mode == "evaluation":
            return "Evaluation task complete: predictions and metrics saved."
        if self.mode == "environment":
            return "Environment check passed: source/input verification and CUDA optimizer smoke completed."
        if self.mode == "cache":
            return "Cache staging complete: files and SHA-256 checks passed. Ready to launch and monitor training."
        return "Training workflow complete: checkpoint selection saved. " + (
            "Public-test and physical evaluation remain deferred."
            if self.validation_only
            else "Configured evaluation steps completed."
        )


def launch(command, log_path, *, on_status=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_reported = False
    progress = WorkflowProgress(command) if on_status is not None else None
    next_status = 0

    def display(line):
        nonlocal fallback_reported
        # Notebook output stays bounded; full progress remains in the log file.
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        if progress is not None:
            progress.record(event)
        if (
            isinstance(event, dict)
            and event.get("cache_transfer") == "DriveFS-fallback"
            and not fallback_reported
        ):
            print(
                "Cache transfer fallback: Google Drive mount; downloads may be slower. See the diagnostic log.",
                flush=True,
            )
            fallback_reported = True
        if (
            isinstance(event, dict)
            and event.get("phase") == "resume-plan"
            and event.get("action") == "resume"
        ):
            print(f"Resume {event['arm']}: epoch {event['resume_epoch']}", flush=True)

    with log_path.open("ab", buffering=0) as log:
        start_offset = log.tell()
        with subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            # Tail only the child-owned file; the producer never depends on the browser pipe.
            with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                reader.seek(start_offset)
                pending = ""
                while process.poll() is None:
                    text = reader.read()
                    if text:
                        lines = (pending + text).split("\n")
                        pending = lines.pop()
                        for line in lines:
                            display(line)
                    if progress is not None and time.monotonic() >= next_status:
                        on_status(progress.message())
                        next_status = time.monotonic() + 5
                    time.sleep(1)
                for line in (pending + reader.read()).splitlines():
                    display(line)
            result = process.returncode

    if result:
        if on_status is not None:
            on_status(progress.finished(True))
        print(
            "Workflow failed; inspect the saved diagnostic log:",
            log_path.name,
            flush=True,
        )
        raise subprocess.CalledProcessError(result, command)

    if on_status is not None:
        on_status(progress.finished(False))


def monitor(command, log_path, pid, *, on_status):
    """Observe an existing producer without starting, stopping or resuming it."""
    process = Path("/proc") / str(pid) / "cmdline"
    identity = process.read_bytes()
    if not identity:
        on_status("Producer already ended; read the saved results.")
        return
    progress = WorkflowProgress(command)
    with Path(log_path).open(encoding="utf-8", errors="replace") as reader:
        pending = ""
        replay = True
        while True:
            text = pending + reader.read()
            lines = text.split("\n")
            pending = lines.pop()
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    progress.record(event)
            if replay:
                # Historical lines are read together, not timed epoch completions.
                progress.epoch_seconds.clear()
                progress.last_epoch_at.clear()
                replay = False
            on_status(progress.message())
            try:
                alive = process.read_bytes() == identity
            except FileNotFoundError:
                alive = False
            if not alive:
                on_status(
                    f"{progress.mode.capitalize()} producer ended. Read saved results or the diagnostic log; no new computation started."
                )
                return
            time.sleep(5)


def run_notebook(command, log_path):
    """Show one status area; attach to a matching live trainer when present."""
    # IPython is needed only at the notebook presentation boundary.
    from IPython.display import Markdown, display

    log_path = Path(log_path)
    status = display(Markdown("Checking workflow state…"), display_id=True)

    def update(message):
        status.update(Markdown(message))

    if "--prepare-only" not in command and "--runtime-only" not in command:
        root = Path(command[command.index("--run-root") + 1])
        health = root / "training-health.json"
        saved = json.loads(health.read_text()) if health.is_file() else {}
        pid = saved.get("pid")
        try:
            arguments = (
                (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
            )
        except FileNotFoundError:
            arguments = []
        same_output = (
            "--run-root" in arguments
            and Path(arguments[arguments.index("--run-root") + 1]).resolve()
            == root.resolve()
        )
        if same_output and any(
            Path(value).name == "run_liveness_comparison.py" for value in arguments[:3]
        ):
            monitor(command, log_path, pid, on_status=update)
            return
    try:
        launch(command, log_path, on_status=update)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Workflow failed (exit {error.returncode}); inspect {log_path.name}"
        ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A Python workflow command is required after --")
    launch(command, args.log_path)


if __name__ == "__main__":
    main()
