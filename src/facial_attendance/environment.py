"""Check dependency compatibility, then one explicit CPU or CUDA training step."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from .config import load_paths


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_pins(path: Path) -> dict[str, str]:
    """Read the project's exact version pins, excluding comments and blank lines."""
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"Expected an exact dependency pin in {path.name}: {line}")
        name, expected = line.split("==")
        if not name or not expected or name in pins:
            raise ValueError(f"Invalid or duplicate dependency pin: {line}")
        pins[name] = expected
    return pins


def version_matches(expected: str, installed: str) -> bool:
    # CPU and CUDA wheels share the public version, e.g. 2.13.0+cpu/+cu126.
    # An explicitly pinned build suffix still requires an exact build match.
    return (
        installed == expected
        if "+" in expected
        else installed.split("+", 1)[0] == expected
    )


def validate_python(
    actual: tuple[int, int, int], *, owner_patch: str | None = None
) -> None:
    # The team uses one Python minor version. Patch releases may differ between
    # machines/Colab; only an explicit owner snapshot replay requires its patch.
    require(actual[:2] == (3, 13), "Project runtime requires Python 3.13.x")
    if owner_patch is not None:
        require(
            ".".join(map(str, actual)) == owner_patch,
            "Owner snapshot requires its exact Python patch",
        )


def select_device(requested: str, *, cuda_available: bool) -> str:
    """Honor the requested device; unavailable CUDA never becomes CPU implicitly."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        require(
            cuda_available,
            "CUDA is inaccessible. Stop, or explicitly choose --device cpu.",
        )
        return "cuda:0"
    raise ValueError(f"Unsupported device: {requested}")


def validate_platform(
    system: str,
    machine: str,
    *,
    macos_version: str = "",
    device: str = "cpu",
    strict_owner: bool = False,
) -> None:
    """Reject targets without wheels for the pinned stack before ML imports."""
    x64 = machine.lower() in {"amd64", "x86_64"}
    if strict_owner:
        require(
            system == "Linux" and x64,
            "--strict-owner is the Linux x86_64 owner snapshot; omit it on Windows/macOS.",
        )
    if system == "Darwin":
        # Intel and Rosetta Python cannot install the pinned arm64 Torch wheels.
        require(
            machine.lower() == "arm64",
            "The pinned Torch stack requires native Apple Silicon Python on macOS. "
            "macOS Intel/Rosetta is unsupported; see docs/setup.md.",
        )
        try:
            major = int(macos_version.split(".")[0])
        except ValueError:
            raise RuntimeError(
                "Could not determine macOS version; macOS 14+ is required."
            ) from None
        require(major >= 14, "The pinned Torch wheels require macOS 14 or newer.")
        require(
            device == "cpu",
            "macOS does not support this CUDA profile. Select --device cpu explicitly; "
            "MPS execution has not been implemented or verified.",
        )
    elif system in {"Windows", "Linux"}:
        require(x64, f"This project setup targets {system} x86_64, not {machine}.")
    else:
        raise RuntimeError(
            f"Unsupported operating system: {system}. See docs/setup.md."
        )


def source_hashes(root: Path) -> dict[str, str]:
    """Identify the code/config used by this run, without hashing private datasets."""
    files = [
        root / name
        for name in (
            "pyproject.toml",
            "requirements.txt",
            "requirements-cpu.txt",
            "requirements-cuda.txt",
            "requirements.lock",
            ".python-version",
        )
    ]
    for folder, pattern in [
        ("src", "*.py"),
        ("scripts", "*.py"),
        ("tests", "*.py"),
        ("configs", "*.toml"),
    ]:
        files.extend((root / folder).rglob(pattern))
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def check_versions(root: Path, *, strict_owner: bool) -> dict:
    """Separate portable direct pins from the optional Linux/CUDA owner snapshot."""
    owner_patch = (
        (root / ".python-version").read_text().strip() if strict_owner else None
    )
    validate_python(sys.version_info[:3], owner_patch=owner_patch)
    expected = read_pins(root / "requirements.txt")
    installed = {name: version(name) for name in expected}
    mismatches = {
        name: installed[name]
        for name, wanted in expected.items()
        if not version_matches(wanted, installed[name])
    }
    require(
        not mismatches, f"Project version mismatch: {mismatches}; expected {expected}"
    )

    # A CPU wheel or Colab session must not be required to install the owner's
    # NVIDIA transitive packages simply to pass a portable application check.
    locked = read_pins(root / "requirements.lock") if strict_owner else {}
    differences = {
        name: version(name)
        for name, wanted in locked.items()
        if version(name) != wanted
    }
    require(not differences, f"Owner snapshot differs: {differences}")
    return {
        "versions": installed,
        "version_policy": "owner_snapshot" if strict_owner else "portable_pins",
        "locked_distributions_checked": len(locked),
    }


def check_environment(
    root: Path, *, device: str = "cuda", strict_owner: bool = False
) -> dict:
    """Validate setup in stages; dataset availability is reported, not fabricated."""
    validate_platform(
        platform.system(),
        platform.machine(),
        macos_version=platform.mac_ver()[0],
        device=device,
        strict_owner=strict_owner,
    )
    versions = check_versions(root, strict_owner=strict_owner)
    paths = load_paths(root / "configs/project.toml", root=root)
    # A single lazy boundary keeps --help/policy tests lightweight and lets main()
    # record missing-package/import failures in JSON instead of crashing at startup.
    from .runtime_checks import check_runtime

    computation = check_runtime(device)
    package_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    require(
        package_check.returncode == 0,
        f"Dependency conflicts: {package_check.stdout}\n{package_check.stderr}",
    )
    return {
        **versions,
        **computation,
        "checks": {
            "platform": True,
            "imports": True,
            "direct_pins": True,
            "config": True,
            "model_and_data_on_requested_device": True,
            "finite_loss": True,
            "finite_gradients": True,
            "parameters_updated": True,
            "inference": True,
            "image_and_metric_apis": True,
            "pip_check": True,
        },
        "pip_check": package_check.stdout.strip(),
        "face_dataset_present": paths.face_data.is_dir(),
        "limits": "One synthetic step; no dataset accuracy, webcam or GUI evaluation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New evidence JSON; never overwritten",
    )
    parser.add_argument("--operator", required=True, help="Self-reported run label")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--strict-owner",
        action="store_true",
        help="Also enforce the owner's exact Python patch and Linux/CUDA lock",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists. Choose a new path to preserve earlier evidence.")
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator,
        "status": "FAIL",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "macos_version": platform.mac_ver()[0],
    }
    exit_code = 1
    try:
        result["source_sha256"] = source_hashes(root)
        result.update(
            check_environment(root, device=args.device, strict_owner=args.strict_owner)
        )
        result["status"] = "PASS"
        exit_code = 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
