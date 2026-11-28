"""Stage verified training inputs and continue into the comparison runner.

Prepared face caches are copied from the delivery layout to runtime storage.
Source inventory, split assignment and checksums are verified before training.
The process lock requires a POSIX runtime.
"""

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import SimpleQueue
from urllib.request import Request, urlopen

from liveness_training_health import (
    DurableTrainingOutput,
    TrainingHealth,
    check_producer_available,
)
from liveness_training_health import run_logged_command as execute

RCLONE_DOWNLOAD_OPTIONS = (
    "--multi-thread-streams",
    "16",
    "--multi-thread-cutoff",
    "32M",
    "--transfers",
    "1",
    "--buffer-size",
    "8M",
)


def sha(path, *, report_progress=False):
    if report_progress:
        size = path.stat().st_size
        checked = 0
        started = last_report = time.monotonic()
        digest = hashlib.sha256()

        def report():
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "phase": "cache-sha256",
                        "split": path.parent.name,
                        "file": path.name.removesuffix(".rclone-copying"),
                        "checked_bytes": checked,
                        "total_bytes": size,
                        "bytes_per_second": checked / max(elapsed, 0.001),
                    }
                ),
                flush=True,
            )

        report()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                checked += len(chunk)
                if time.monotonic() - last_report >= 5:
                    report()
                    last_report = time.monotonic()
        report()
        return digest.hexdigest()
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_resumable(source, target, expected_bytes, progress=None):
    if source.stat().st_size != expected_bytes:
        raise ValueError("Source size mismatch: " + source.name)
    if target.exists():
        if target.stat().st_size != expected_bytes:
            raise ValueError("Incomplete committed target: " + target.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        print(
            json.dumps(
                {
                    "phase": "cache-copy",
                    "split": target.parent.name,
                    "file": target.name,
                }
            ),
            flush=True,
        )
    temporary = target.with_name(target.name + ".copying")
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > expected_bytes:
            raise ValueError("Oversized partial transfer: " + str(temporary))
        try:
            with source.open("rb") as src, temporary.open("ab") as dst:
                src.seek(offset)
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
                    if progress is not None:
                        progress.put(len(chunk))
            if temporary.stat().st_size != expected_bytes:
                raise ValueError("Truncated source transfer: " + source.name)
            temporary.rename(target)
            return
        except OSError as error:
            if attempt == 2 or error.errno not in (
                errno.EIO,
                errno.ETIMEDOUT,
                errno.ECONNRESET,
                errno.ECONNABORTED,
                errno.EAGAIN,
            ):
                raise
            time.sleep(2**attempt)


class RcloneCacheDownload:
    """Download verified cache bytes directly; never resume sparse rclone output."""

    def __init__(self, binary, folder_id, credentials, *, source_root=None):
        self.binary, self.folder_id, self.credentials = binary, folder_id, credentials
        self.source_root = source_root

    def run_download(self, command, *, env, progress, split, file):
        """Expose bounded transfer statistics without printing raw credential diagnostics."""
        started = time.monotonic()
        transferred = 0
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as diagnostics:
            with subprocess.Popen(
                command, env=env, stdout=subprocess.DEVNULL, stderr=diagnostics
            ) as process:
                position = 0
                while process.poll() is None:
                    chunk = os.pread(diagnostics.fileno(), 1024 * 1024, position)
                    complete = chunk.rsplit(b"\n", 1)[0] if b"\n" in chunk else b""
                    lines = complete.decode("utf-8", errors="replace").splitlines()
                    position += len(complete) + (1 if complete else 0)
                    for line in lines:
                        try:
                            stats = json.loads(line).get("stats", {})
                        except ValueError:
                            continue
                        count = int(stats.get("bytes", transferred))
                        if count > transferred:
                            progress.put(count - transferred)
                            transferred = count
                            print(
                                json.dumps(
                                    {
                                        "cache_transfer": "rclone-progress",
                                        "split": split,
                                        "file": file,
                                        "copied_bytes": count,
                                        "bytes_per_second": stats.get("speed"),
                                    }
                                ),
                                flush=True,
                            )
                    if time.monotonic() - started > 1800:
                        process.kill()
                        process.wait()
                        raise subprocess.TimeoutExpired(command, 1800)
                    time.sleep(1)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
        return transferred

    def copy(self, source, target, expected_sha256, progress):
        # Preserve a valid sequential prefix left by an earlier DriveFS copy.
        if target.exists() or target.with_name(target.name + ".copying").exists():
            return False
        relative = (
            source.relative_to(self.source_root).parts
            if self.source_root
            else source.parts[-2:]
        )
        if not relative or any(part in ("", ".", "..") for part in relative):
            raise ValueError("Cache download escapes its source folder")
        split = relative[0] if len(relative) > 1 else source.parent.name
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + ".rclone-copying")
        # An interrupted multi-stream file may contain holes, even at full size.
        if pending.exists():
            pending.unlink()
        try:
            # Optional Colab dependency: this boundary is unused for local/DriveFS copies.
            from google.auth.transport.requests import Request

            if not self.credentials.valid:
                self.credentials.refresh(Request())
            expiry = self.credentials.expiry or datetime.now(UTC) + timedelta(
                minutes=30
            )
            token = {
                "access_token": self.credentials.token,
                "token_type": "Bearer",
                "expiry": expiry.isoformat() + "Z"
                if expiry.tzinfo is None
                else expiry.isoformat(),
            }
            environment = {
                **os.environ,
                "RCLONE_CONFIG_CACHE_TYPE": "drive",
                "RCLONE_CONFIG_CACHE_TOKEN": json.dumps(token),
            }
            command = [
                str(self.binary),
                "copyto",
                "cache:" + "/".join(relative),
                str(pending),
                "--config",
                os.devnull,
                "--drive-root-folder-id",
                self.folder_id,
                *RCLONE_DOWNLOAD_OPTIONS,
                "--retries",
                "1",
                "--low-level-retries",
                "3",
                "--contimeout",
                "10s",
                "--timeout",
                "60s",
                "--stats",
                "30s",
                "--stats-log-level",
                "NOTICE",
                "--use-json-log",
                "--log-level",
                "NOTICE",
            ]
            print(
                json.dumps(
                    {
                        "cache_transfer": "rclone-start",
                        "split": split,
                        "file": source.name,
                        "expected_bytes": source.stat().st_size,
                        "streams": 16,
                    }
                ),
                flush=True,
            )
            started = time.monotonic()
            # Capture diagnostics: OAuth context and account paths never enter console output.
            transferred = self.run_download(
                command,
                env=environment,
                progress=progress,
                split=split,
                file=source.name,
            )
            if (
                pending.stat().st_size != source.stat().st_size
                or sha(pending, report_progress=True) != expected_sha256
            ):
                raise ValueError("Downloaded cache checksum mismatch")
            size = pending.stat().st_size
            pending.replace(target)
            progress.put(max(0, size - (transferred or 0)))
            print(
                json.dumps(
                    {
                        "cache_transfer": "rclone",
                        "split": split,
                        "file": source.name,
                        "bytes": size,
                        "sha256_verified": True,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            return True
        except (ImportError, OSError, ValueError, subprocess.SubprocessError) as error:
            print(
                json.dumps(
                    {
                        "cache_transfer": "DriveFS-fallback",
                        "split": split,
                        "file": source.name,
                        "reason": type(error).__name__,
                        "limit": "Google Drive quotas and Colab CU still apply",
                    }
                ),
                flush=True,
            )
            return False
        finally:
            if pending.exists():
                pending.unlink()


def prepare_rclone():
    """Install the pinned standalone transfer utility in the disposable runtime."""
    binary = shutil.which("rclone")
    if binary:
        return binary
    target = Path("/content/rclone")
    if target.is_file() and os.access(target, os.X_OK):
        return str(target)
    if not Path("/content/drive/MyDrive").is_dir():
        return None

    version = "v1.60.1"
    archive_name = f"rclone-{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/{version}/"
    # The release host rejects urllib's default User-Agent (HTTP 403/1010).
    # Identify this client for both downloads; keep release checksum validation.
    headers = {"User-Agent": "facial-attendance-cache-setup/1.0"}
    try:
        with urlopen(
            Request(base_url + "SHA256SUMS", headers=headers), timeout=30
        ) as response:
            checksums = response.read(1024 * 1024).decode("utf-8")
        # Published lists include a PGP envelope and blank lines.
        entries = (line.split() for line in checksums.splitlines())
        expected = next(
            fields[0]
            for fields in entries
            if len(fields) == 2 and fields[1].lstrip("*") == archive_name
        )
        # Extract only the named executable; no downloaded installer is executed.
        with tempfile.TemporaryDirectory(
            prefix="liveness-rclone-", dir="/content"
        ) as temporary:
            archive_path = Path(temporary) / archive_name
            with (
                urlopen(
                    Request(base_url + archive_name, headers=headers), timeout=60
                ) as response,
                archive_path.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if sha(archive_path) != expected:
                raise ValueError("Rclone release checksum mismatch")
            executable = Path(temporary) / "rclone"
            with zipfile.ZipFile(archive_path) as archive:
                with (
                    archive.open(f"rclone-{version}-linux-amd64/rclone") as source,
                    executable.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output)
            executable.chmod(0o700)
            executable.replace(target)
        print(
            json.dumps(
                {
                    "cache_transfer_utility": "rclone",
                    "version": version,
                    "release_sha256_verified": True,
                }
            ),
            flush=True,
        )
        return str(target)
    except (OSError, ValueError, StopIteration, KeyError, zipfile.BadZipFile) as error:
        print(
            json.dumps(
                {
                    "cache_transfer": "DriveFS-fallback",
                    "stage": "rclone-setup",
                    "reason": type(error).__name__,
                }
            ),
            flush=True,
        )
        return None


def rclone_cache_download(source):
    """Resolve the mounted source by Google Drive IDs without an owner-specific path."""
    binary = prepare_rclone()
    mount = Path("/content/drive/MyDrive")
    if binary is None or not source.is_relative_to(mount):
        return None
    try:
        # Colab supplies these optional libraries after Google Drive authorization.
        import google.auth
        from googleapiclient.discovery import build

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        api = build("drive", "v3", credentials=credentials, cache_discovery=False)
        folder_id = api.files().get(fileId="root", fields="id").execute()["id"]
        for name in source.relative_to(mount).parts:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            query = f"'{folder_id}' in parents and name='{escaped}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            matches = (
                api.files()
                .list(q=query, fields="files(id)", pageSize=2)
                .execute()["files"]
            )
            if len(matches) != 1:
                raise ValueError("Cache folder cannot be resolved unambiguously")
            folder_id = matches[0]["id"]
        return RcloneCacheDownload(binary, folder_id, credentials, source_root=source)
    except Exception as error:
        # Acquisition is optional; integrity errors in the chosen copy still fail below.
        print(
            json.dumps(
                {
                    "cache_transfer": "DriveFS-fallback",
                    "reason": type(error).__name__,
                    "stage": "rclone-authorization",
                }
            ),
            flush=True,
        )
        return None


def ram_cache_fits(required_bytes, shm_free, memory_available, memory_total):
    """Leave room for model setup, loader processes, pinned batches and the OS."""
    reserve = max(16 * 2**30, memory_total // 5)
    return (
        shm_free >= required_bytes + 2 * 2**30
        and memory_available >= required_bytes + reserve
    )


def choose_cache_directory(prepared, ssd, *, shm=Path("/dev/shm/liveness-faces")):
    """Use measured capacity, not a GPU name or the High-RAM setting."""
    # Preserve existing SSD progress instead of allocating a second full cache.
    if ssd.exists() and any(
        path.stat().st_size for path in ssd.rglob("*.npy*") if path.is_file()
    ):
        return ssd, "SSD: reusing existing cache or partial transfer"
    required = 0
    for split in ("train", "validation"):
        folder = prepared / split
        receipt = json.loads((folder / "decoded-faces.json").read_text())
        required += (folder / "decoded-faces.npy").stat().st_size
        required += sum(
            (folder / name).stat().st_size
            for name in receipt.get("corrections", {}).get("files", {})
        )
    if not shm.parent.is_dir():
        return ssd, "SSD: RAM disk unavailable"
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))
    }
    available, total = memory["MemAvailable"], memory["MemTotal"]
    # A container can have a smaller allocation than host /proc/meminfo reports.
    quota = Path("/sys/fs/cgroup/memory.max")
    if quota.is_file() and quota.read_text().strip() != "max":
        limit = int(quota.read_text())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        total, available = min(total, limit), min(available, max(0, limit - used))
    if ram_cache_fits(required, shutil.disk_usage(shm.parent).free, available, total):
        return shm, "RAM disk: capacity and training headroom checked"
    return ssd, "SSD: RAM disk capacity or training headroom insufficient"


def prepare_face_caches(args, python, data, health):
    """Prepare on CPU; the trainer continues to use its original mmap loader."""
    if args.decoded_cache is None:
        raise ValueError(
            "Face transport requires an explicit --decoded-cache directory"
        )
    ensure_cache_storage(args.face_cache_transport, args.decoded_cache, transport=True)
    health.phase("materializing-face-cache")
    for split, annotations in (
        ("train", "reviewed_project_train.json"),
        ("validation", "project_validation.json"),
    ):
        execute(
            [
                python,
                args.project_root / "scripts/release/transport_liveness_face_cache.py",
                "materialize",
                "--data-root",
                data,
                "--inventory",
                data / "audits/current-schema-2026-11-12/inventory.csv",
                "--annotations",
                data / annotations,
                "--transport",
                args.face_cache_transport / split,
                "--output-dir",
                args.decoded_cache / split,
            ]
        )


def ensure_cache_storage(source, destination, *, transport=False):
    """Check peak capacity for both arrays before writing the first split."""
    destination.mkdir(parents=True, exist_ok=True)
    remaining, largest_shard = 0, 0
    for split in ("train", "validation"):
        name = "face-cache-shards.json" if transport else "decoded-faces.json"
        record = json.loads((source / split / name).read_text())
        if transport:
            if not record["completed"]:
                raise ValueError(
                    "Complete face transports are required before preparation"
                )
            largest_shard = max(
                [largest_shard, *(row["bytes"] for row in record["shards"])]
            )
        # Include file/header allocation overhead. A partially written mmap
        # can report its final length while still containing sparse holes.
        target = destination / split / "decoded-faces.npy"
        allocated = target.stat().st_blocks * 512 if target.exists() else 0
        remaining += max(0, math.prod(record["shape"]) + 4096 - allocated)
        for name in record.get("corrections", {}).get("files", {}):
            if Path(name).name != name:
                raise ValueError("Correction filename escapes its cache")
            remaining += (source / split / name).stat().st_size
    required = remaining + largest_shard + 2 * 2**30
    if shutil.disk_usage(destination).free < required:
        raise ValueError(
            f"Insufficient storage for both face arrays and staging headroom: {required} bytes required"
        )


def copy_decoded_face_caches(source, destination, data, health, *, verify_destination):
    """Persist or stage complete arrays; verify staged bytes before GPU training."""
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination cache directories must differ")
    ensure_cache_storage(source, destination)
    downloader = rclone_cache_download(source) if verify_destination else None
    total = 0
    last_progress = time.monotonic()

    class Progress:
        def put(self, count):
            nonlocal total, last_progress
            total += count
            if time.monotonic() - last_progress >= 30:
                health.state.update(
                    cache_bytes_copied_this_session=total,
                    last_progress_at=datetime.now(UTC).isoformat(),
                )
                health.publish()
                print(
                    json.dumps(
                        {"phase": "cache-transfer", "copied_bytes_this_session": total}
                    ),
                    flush=True,
                )
                last_progress = time.monotonic()

    for split, annotations in (
        ("train", "reviewed_project_train.json"),
        ("validation", "project_validation.json"),
    ):
        receipt = json.loads((source / split / "decoded-faces.json").read_text())
        binding = receipt["data"]
        if not binding.get("crop_recipe"):
            raise ValueError(
                "Prepared cache lacks its crop recipe; verify preprocessing before staging"
            )
        inventory = data / "audits/current-schema-2026-11-12/inventory.csv"
        annotation_file = data / annotations
        if (
            receipt["schema"] != "liveness-decoded-faces-v1"
            or receipt["dtype"] != "uint8"
            or receipt["shape"][1:] != [224, 224, 3]
            or receipt["shape"][0] != binding["samples"]
            or binding["inventory_sha256"] != sha(inventory)
            or binding["annotations_sha256"] != sha(annotation_file)
        ):
            raise ValueError("Prepared face cache does not match " + split)
        array = source / split / "decoded-faces.npy"
        target = destination / split / "decoded-faces.npy"
        if not verify_destination and sha(array) != receipt["sha256"]:
            raise ValueError("Source decoded array checksum mismatch: " + split)
        previous_receipt = target.parent / "decoded-faces.json"
        verified = False
        if target.is_file() and previous_receipt.is_file():
            prior = json.loads(previous_receipt.read_text())
            info = target.stat()
            identity = [
                str(target.resolve()),
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ]
            verified = (
                prior.get("sha256") == receipt["sha256"]
                and prior.get("array_identity") == identity
            )
        if not verified and downloader:
            verified = downloader.copy(array, target, receipt["sha256"], Progress())
        if not verified:
            copy_resumable(array, target, array.stat().st_size, Progress())
        # Cloud publication is checked again on consumption; avoid rereading
        # the entire remote array just to copy its already verified receipt.
        if (
            verify_destination
            and not verified
            and sha(target, report_progress=True) != receipt["sha256"]
        ):
            raise ValueError("Staged decoded array checksum mismatch: " + split)
        for name, digest in receipt.get("corrections", {}).get("files", {}).items():
            if Path(name).name != name:
                raise ValueError("Correction filename escapes its cache")
            replacement = source / split / name
            copied = target.parent / name
            verified = (
                downloader.copy(replacement, copied, digest, Progress())
                if downloader
                else False
            )
            if not verified:
                copy_resumable(
                    replacement, copied, replacement.stat().st_size, Progress()
                )
            if (
                not verified
                and sha(copied, report_progress=verify_destination) != digest
            ):
                raise ValueError("Staged face correction checksum mismatch: " + split)
        receipt["data"] = {
            "root": str(data.resolve()),
            "inventory": str(inventory.resolve()),
            "inventory_sha256": binding["inventory_sha256"],
            "annotations": str(annotation_file.resolve()),
            "annotations_sha256": binding["annotations_sha256"],
            "samples": binding["samples"],
            "crop_recipe": binding["crop_recipe"],
        }
        receipt.pop("array_identity", None)
        if verify_destination:
            info = target.stat()
            receipt["array_identity"] = [
                str(target.resolve()),
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ]
        temporary = target.parent / "decoded-faces.json.partial"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target.parent / "decoded-faces.json")


def stage_verified_inputs(args, health):
    """Restore only inventory-bound inputs inside the selected working directory."""
    root = args.project_root
    inventory_path = (
        args.release_inventory or args.run_root / "reproduction/release-inventory.json"
    )
    source_root = (args.source_root or args.run_root).resolve()
    inventory = json.loads(inventory_path.read_text())
    geometry_downloader = None
    # Colab autosaves cell outputs into the notebook while this workflow runs.
    # It is an interactive entry point, not an input consumed by training.
    runtime_files = {
        name: record
        for name, record in inventory["files"].items()
        if name != "notebooks/liveness-training-and-evaluation.ipynb"
    }
    for index, (name, record) in enumerate(runtime_files.items(), 1):
        source, target = (
            (source_root / record["delivery_path"]).resolve(),
            (root / name).resolve(),
        )
        if not source.is_relative_to(source_root) or not target.is_relative_to(root):
            raise ValueError("Inventory target escapes its declared root")
        if target.exists():
            if sha(target) != record["sha256"]:
                raise ValueError(
                    "Existing staged input differs; use a fresh project root: " + name
                )
        else:
            verified = False
            if name == "transfer/geometric-cache/geometric-targets.npy":
                geometry_downloader = rclone_cache_download(source.parent)
                if geometry_downloader is not None:
                    verified = geometry_downloader.copy(
                        source, target, record["sha256"], SimpleQueue()
                    )
            if not verified:
                copy_resumable(source, target, record["bytes"])
                if sha(target) != record["sha256"]:
                    raise ValueError("Transferred content mismatch: " + name)
        health.state.update(
            verified_input_files=index,
            total_input_files=len(runtime_files),
            last_progress_at=datetime.now(UTC).isoformat(),
        )
        health.publish()
        print(
            json.dumps(
                {
                    "phase": "source-verification",
                    "completed": index,
                    "total": len(runtime_files),
                }
            ),
            flush=True,
        )
    transfer = root / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inventory_path, transfer / "release-inventory.json")


def prepare_environment(args, health):
    """Validate dependencies separately from CUDA access; never reinstall for a GPU failure."""
    root = args.project_root
    health.phase("environment")
    python = root / ".venv/bin/python"
    version_probe = (
        "import sys,torch,numpy,PIL; assert sys.version_info[:2]==(3,13); "
        "assert str(torch.__version__) in ('2.13.0+cu126','2.13.0+cu130'); "
        "assert numpy.__version__=='2.2.6'; assert PIL.__version__=='11.2.1'"
    )
    probe = (
        subprocess.run(
            [str(python), "-c", version_probe],
            capture_output=True,
            text=True,
        )
        if python.exists()
        else None
    )
    if probe is None or probe.returncode:
        if sys.version_info[:2] != (3, 13) or not shutil.which("uv"):
            raise RuntimeError("Use the project's Python 3.13 runtime with uv")
        execute(
            [
                "uv",
                "venv",
                "--allow-existing",
                "--python",
                sys.executable,
                root / ".venv",
            ]
        )
        for requirements in ("requirements-cuda.txt", "requirements.txt"):
            execute(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--no-cache",
                    "-r",
                    root / requirements,
                ]
            )
        execute(["uv", "pip", "check", "--python", python])
    if not args.prepare_only:
        execute(
            [
                python,
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "m=torch.nn.Conv2d(3,4,3).cuda(); x=torch.rand(2,3,16,16,device='cuda'); "
                "o=torch.optim.SGD(m.parameters(),lr=.005); o.zero_grad(); "
                "loss=m(x).square().mean(); loss.backward(); "
                "assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters()); "
                "o.step(); torch.cuda.synchronize(); print('CUDA optimizer smoke PASS:',torch.cuda.get_device_name())",
            ]
        )
    return python


def restore_original_images(args, health, data):
    """Historical source replay only; the screening route never calls this."""
    root = args.project_root
    health.phase("restoring-data")
    spec = importlib.util.spec_from_file_location(
        "source_extractor", root / "scripts/preparation/prepare_liveness_data.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    sources = [
        args.parts_root / f"CelebA_Spoof.zip.{index:03}" for index in range(1, 75)
    ]

    expected = [1048576000] * 73 + [453369048]
    for source, size in zip(sources, expected, strict=True):
        if source.stat().st_size != size:
            raise ValueError("Archive part size mismatch: " + source.name)
    parts, extracted = (
        root.parent / (root.name + "-archive-parts"),
        root.parent / (root.name + "-source-extraction"),
    )
    parts.mkdir(exist_ok=True)
    with (
        preparation.ArchiveParts(sources) as stream,
        zipfile.ZipFile(stream) as archive,
    ):
        footprint = sum(
            ((entry.file_size + 4095) // 4096) * 4096 if not entry.is_dir() else 4096
            for entry in archive.infolist()
        )
    del archive, stream

    missing = sum(
        max(
            0,
            size
            - (
                (parts / (source.name + ".copying")).stat().st_size
                if (parts / (source.name + ".copying")).exists()
                else 0
            ),
        )
        for source, size in zip(sources, expected, strict=True)
        if not (parts / source.name).exists()
    )
    remaining = 0 if (extracted / "extraction.json").exists() else footprint
    if shutil.disk_usage(root).free < missing + remaining + 2 * 2**30:
        raise ValueError(
            "Insufficient peak storage for source archives and extracted images"
        )

    progress = SimpleQueue()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            pool.submit(copy_resumable, source, parts / source.name, size, progress)
            for source, size in zip(sources, expected, strict=True)
        }
        transferred = 0
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            previous = transferred
            while not progress.empty():
                transferred += progress.get()
            if transferred > previous:
                health.state.update(
                    restored_bytes_this_session=transferred,
                    last_progress_at=datetime.now(UTC).isoformat(),
                )
                health.publish()
            for task in done:
                task.result()

    if not (extracted / "extraction.json").exists():
        preparation.extract_full(parts, extracted, workers=4, resume=extracted.exists())
    if (
        json.loads((extracted / "extraction.json").read_text())["crc_all_files"]
        != "PASS"
    ):
        raise ValueError("Complete archive CRC verification required")
    (extracted / "Data").rename(data / "Data")
    shutil.copyfile(
        extracted / "extraction.json",
        args.run_root / ("recovery-extraction-" + health.state["job_id"] + ".json"),
    )


def restore(args, health):
    root = args.project_root
    stage_verified_inputs(args, health)
    python = prepare_environment(args, health)
    if getattr(args, "runtime_only", False):
        health.finish()
        return
    sys.path.insert(0, str(root / "src"))
    from facial_attendance.config import load_paths

    data = load_paths(
        root / (getattr(args, "config", None) or "configs/project.toml"),
        root=root,
        overrides={"liveness_data": getattr(args, "data_root", None)},
    ).liveness_data
    if args.prepared_cache_root is not None:
        if args.decoded_cache is None:
            raise ValueError("Staging prepared arrays requires --decoded-cache")
        health.phase("staging-prepared-face-cache")
        copy_decoded_face_caches(
            args.prepared_cache_root,
            args.decoded_cache,
            data,
            health,
            verify_destination=True,
        )
    elif args.face_cache_transport is not None:
        prepare_face_caches(args, python, data, health)
    elif not (data / "Data").is_dir():
        if args.screening_epochs is not None or args.parts_root is None:
            raise ValueError(
                "Screening recovery requires --prepared-cache-root or --face-cache-transport; no ZIP fallback to the 74-part CelebA-Spoof dataset archive is allowed"
            )
        restore_original_images(args, health, data)

    health.phase("verifying-resume")
    rebound = root / "transfer" / ("recovery-" + health.state["job_id"])
    cache_only = args.arms is not None or all(
        any(
            (path / "training-results.json").exists()
            and json.loads((path / "training-results.json").read_text()).get(
                "full_training_completed"
            )
            for path in args.run_root.glob(arm + "*")
            if path.is_dir()
        )
        for arm in ("binary-classification", "aenet-full")
    )
    if cache_only:
        execute(
            [
                python,
                root / "scripts/release/package_liveness_release.py",
                "cache",
                "--repository",
                root,
                "--output",
                rebound / "geometric-cache",
            ]
        )
    else:
        relocation = [
            python,
            root / "scripts/release/package_liveness_release.py",
            "relocate",
            "--repository",
            root,
            "--inventory",
            root / "transfer/release-inventory.json",
            "--output",
            rebound,
        ]
        if args.pipeline_benchmark is not None:
            relocation.extend(["--pipeline-benchmark", args.pipeline_benchmark])
        if args.training_batch_size is not None:
            relocation.extend(
                [
                    "--training-batch-size",
                    args.training_batch_size,
                    "--batch-start-epoch",
                    args.batch_start_epoch,
                    "--batch-benchmark",
                    args.batch_benchmark,
                ]
            )
        if args.gpu_augmentation is not None:
            relocation.extend(
                [
                    "--gpu-augmentation"
                    if args.gpu_augmentation
                    else "--no-gpu-augmentation"
                ]
            )
        execute(relocation)

    if args.publish_cache_root is not None:
        if not args.prepare_only or args.decoded_cache is None:
            raise ValueError(
                "Cache publication requires preparation-only mode and --decoded-cache"
            )
        health.phase("publishing-prepared-face-cache")
        copy_decoded_face_caches(
            args.decoded_cache,
            args.publish_cache_root,
            data,
            health,
            verify_destination=False,
        )

    if args.prepare_only:
        health.state["preparation_complete"] = True
        health.finish()
        return

    health.phase("ready-for-training")
    sys.stdout.flush()
    sys.stderr.flush()

    # The same PID continues as the trainer. No local terminal pipe, detached
    # watchdog, new worker environment, or activity-spoofing loop is involved.
    command = [
        str(python),
        "-u",
        str(root / "scripts/run_liveness_comparison.py"),
        "--project-root",
        str(root),
        "--run-root",
        str(args.training_output_root or args.run_root),
        "--geometry-cache",
        str(rebound / "geometric-cache"),
    ]
    for option in ("config", "data_root", "trials_root", "detector_model"):
        value = getattr(args, option, None)
        if value is not None:
            command.extend(["--" + option.replace("_", "-"), str(value)])
    if args.loader_workers is not None:
        command.extend(["--loader-workers", str(args.loader_workers)])
    if args.benchmark_loader_workers:
        command.append("--benchmark-loader-workers")
    if args.loader_benchmark is not None:
        command.extend(["--loader-benchmark", str(args.loader_benchmark)])
    if args.decoded_cache is not None:
        command.extend(["--decoded-cache", str(args.decoded_cache)])
    if getattr(args, "reviewed_cache_view", None) is not None:
        command.extend(["--reviewed-cache-view", str(args.reviewed_cache_view)])
    if args.pipeline_benchmark is not None:
        command.extend(["--pipeline-benchmark", str(args.pipeline_benchmark)])
    if args.training_batch_size is not None:
        command.extend(
            [
                "--training-batch-size",
                str(args.training_batch_size),
                "--batch-start-epoch",
                str(args.batch_start_epoch),
                "--batch-benchmark",
                str(args.batch_benchmark),
            ]
        )
    if args.gpu_augmentation is not None:
        command.extend(
            ["--gpu-augmentation" if args.gpu_augmentation else "--no-gpu-augmentation"]
        )
    if args.screening_epochs is not None:
        command.extend(["--screening-epochs", str(args.screening_epochs)])
    if args.initialize_arms:
        command.extend(["--initialize-arms", *args.initialize_arms])
    if args.arms is not None:
        command.extend(["--arms", *args.arms])
    if args.reference_runs_root is not None:
        command.extend(["--reference-runs-root", str(args.reference_runs_root)])
    if args.validation_only:
        command.append("--validation-only")

    os.execv(str(python), command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trials-root", type=Path)
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--parts-root", type=Path)
    parser.add_argument("--face-cache-transport", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Stage source and validate the GPU environment without copying face caches or training",
    )
    parser.add_argument("--prepared-cache-root", type=Path)
    parser.add_argument("--publish-cache-root", type=Path)
    parser.add_argument(
        "--release-inventory",
        "--release-manifest",
        dest="release_inventory",
        type=Path,
        help="Current release file inventory; the old option remains readable for saved commands",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--training-output-root", type=Path)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--screening-epochs", type=int, choices=(15, 25))
    parser.add_argument("--initialize-arms", nargs="+")
    parser.add_argument("--reference-runs-root", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--benchmark-loader-workers", action="store_true")
    parser.add_argument("--loader-workers", type=int)
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
    # Current releases create their own output; legacy recovery still requires its run record.
    if (
        args.release_inventory is None
        and not (args.run_root / "manifest.json").exists()
    ):
        raise FileNotFoundError(
            "Supply the release inventory or an existing training run"
        )
    if args.release_inventory is not None and not args.release_inventory.is_file():
        raise FileNotFoundError("Release inventory is unavailable")
    args.project_root.mkdir(parents=True, exist_ok=True)
    lock_file = args.project_root / "transfer/training.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health_root = (
            (args.run_root / "face-cache-preparation")
            if args.prepare_only
            else (args.training_output_root or args.run_root)
        )
        check_producer_available(health_root)
        job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
        health = TrainingHealth(health_root, job_id)
        native_out, native_err = sys.stdout, sys.stderr
        log = health_root / (job_id + "-console.log")
        sys.stdout = DurableTrainingOutput(native_out, log, health, progress=True)
        sys.stderr = DurableTrainingOutput(native_err, log, health)
        try:
            restore(args, health)
        except BaseException as error:
            sys.stdout, sys.stderr = native_out, native_err
            try:
                health.finish(error=error)
            except OSError as persistence_error:
                print(
                    "Could not persist failure:",
                    repr(persistence_error),
                    file=native_err,
                )
            raise


if __name__ == "__main__":
    main()
