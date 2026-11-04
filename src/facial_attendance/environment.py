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
    # This is the supported project envelope, not a demand to replace Colab's Python.
    # The owner-specific patch pin is checked only for an explicit snapshot replay.
    require(
        (3, 12) <= actual[:2] < (3, 14), "Project runtime requires Python 3.12 or 3.13"
    )
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
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
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


def check_training_step(device) -> tuple[dict, object]:
    """Prove that autograd and the optimizer work, not merely that a model imports."""
    import torch

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    batch = torch.rand(4, 3, 32, 32, device=device)
    targets = torch.tensor([0, 1, 0, 1], device=device)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    require(
        all(p.device == batch.device == targets.device for p in model.parameters()),
        "Model, images and targets must share the requested device",
    )

    # Compare weights before/after: a finite loss alone cannot detect a no-op update.
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(batch), targets)
    require(torch.isfinite(loss).item(), "Training loss is not finite")
    loss.backward()
    require(
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        ),
        "Missing or nonfinite gradients",
    )
    optimizer.step()
    require(
        all(torch.isfinite(p).all().item() for p in model.parameters()),
        "Nonfinite parameters after optimization",
    )
    changed = sum(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )
    require(changed > 0, "No parameters changed after the optimizer step")

    # eval() changes layer behavior; inference_mode() independently disables autograd.
    model.eval()
    with torch.inference_mode():
        output = model(batch)
    require(not model.training and not output.requires_grad, "Invalid inference mode")
    require(
        list(output.shape) == [4, 2] and output.device == device,
        "Inference shape/device does not match the contract",
    )
    require(torch.isfinite(output).all().item(), "Nonfinite inference output")
    return {
        "input": {
            "source": "synthetic RGB tensor",
            "shape": list(batch.shape),
            "seed": 42,
        },
        "training": {
            "loss": loss.item(),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "updated_parameter_tensors": changed,
            "gradients_finite": True,
        },
        "inference": {
            "shape": list(output.shape),
            "finite": True,
            "device": str(output.device),
            "requires_grad": output.requires_grad,
        },
    }, output


def check_image_and_metric_apis(output) -> None:
    """Exercise array/image interoperability without reporting a fake project AUC."""
    import cv2
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    import torchvision

    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    converted = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    require(
        torchvision.transforms.functional.to_tensor(image).shape == (3, 8, 8)
        and converted.shape == (8, 8, 3),
        "Image conversion failed",
    )
    require(np.isfinite(output.cpu().numpy()).all(), "Tensor/NumPy conversion failed")
    require(roc_auc_score([0, 1], [0.1, 0.9]) == 1.0, "Metric API check failed")


def check_environment(
    root: Path, *, device: str = "cuda", strict_owner: bool = False
) -> dict:
    """Validate setup in stages; dataset availability is reported, not fabricated."""
    import cv2
    import matplotlib
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision
    import tqdm

    versions = check_versions(root, strict_owner=strict_owner)
    paths = load_paths(root / "configs/project.toml", root=root)
    # CPU is a deliberate mode. It can run with a CUDA-built wheel without using CUDA.
    selected = torch.device(
        select_device(device, cuda_available=torch.cuda.is_available())
    )
    computation, output = check_training_step(selected)
    check_image_and_metric_apis(output)
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
        "requested_device": device,
        "device": str(selected),
        "gpu": torch.cuda.get_device_name(selected)
        if selected.type == "cuda"
        else None,
        "torch_cuda_build": torch.version.cuda,
        "checks": {
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
