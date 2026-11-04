"""Check the pinned runtime with one CUDA optimizer step and inference pass."""

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
    """Read exact pins; index configuration belongs in installation instructions."""
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


def source_hashes(root: Path) -> dict[str, str]:
    files = [
        root / name
        for name in [
            "pyproject.toml",
            "requirements.txt",
            "requirements.lock",
            ".python-version",
        ]
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


def check_environment(root: Path) -> dict:
    """Return observations; any failed condition stops the check without CPU fallback."""
    import cv2
    import matplotlib
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    import torch
    import torchvision
    import tqdm

    require(
        platform.python_version() == (root / ".python-version").read_text().strip(),
        "Use the pinned Python interpreter; do not substitute system Python.",
    )
    expected = read_pins(root / "requirements.txt")
    installed = {name: version(name) for name in expected}
    require(
        installed == expected,
        f"Project versions differ: expected {expected}, found {installed}",
    )
    lock = read_pins(root / "requirements.lock")
    mismatches = {
        name: version(name) for name, wanted in lock.items() if version(name) != wanted
    }
    require(
        not mismatches,
        f"Installed packages differ from the Linux runtime lock: {mismatches}",
    )
    paths = load_paths(root / "configs/project.toml", root=root)
    require(
        torch.cuda.is_available(),
        "CUDA is inaccessible. Stop and report; CPU fallback is disabled.",
    )
    device = torch.device("cuda:0")
    torch.manual_seed(42)
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
        "Model parameters and both input/target tensors must share the CUDA device.",
    )
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    require(torch.isfinite(loss).item(), "Training loss is not finite.")
    loss.backward()
    require(
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        ),
        "Missing or nonfinite gradients.",
    )
    optimizer.step()
    require(
        all(torch.isfinite(p).all().item() for p in model.parameters()),
        "Nonfinite parameters after optimization.",
    )
    changed = sum(
        not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters())
    )
    require(changed > 0, "No parameters changed after the optimizer step.")

    model.eval()
    with torch.inference_mode():
        output = model(batch)
    require(
        not model.training and not output.requires_grad,
        "Inference mode was not applied.",
    )
    require(list(output.shape) == [4, 2], "Unexpected inference output shape.")
    require(
        output.is_cuda and torch.isfinite(output).all().item(),
        "Invalid GPU inference output.",
    )

    # This only exercises image/metric APIs; it is not verification-dataset evaluation.
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    converted = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    require(
        torchvision.transforms.functional.to_tensor(image).shape == (3, 8, 8)
        and converted.shape == (8, 8, 3),
        "Image conversion failed.",
    )
    require(np.isfinite(output.cpu().numpy()).all(), "Tensor/NumPy conversion failed.")
    require(roc_auc_score([0, 1], [0.1, 0.9]) == 1.0, "Metric API check failed.")
    package_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    require(
        package_check.returncode == 0,
        f"Dependency conflicts: {package_check.stdout}\n{package_check.stderr}",
    )
    return {
        "versions": installed,
        "locked_distributions_checked": len(lock),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
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
        "checks": {
            "imports": True,
            "direct_pins": True,
            "runtime_lock": True,
            "config": True,
            "model_and_data_on_gpu": True,
            "finite_loss": True,
            "finite_gradients": True,
            "parameters_updated": True,
            "inference": True,
            "image_and_metric_apis": True,
            "pip_check": True,
        },
        "pip_check": package_check.stdout.strip(),
        "face_dataset_present": paths.face_data.is_dir(),
        "limits": "One synthetic step only; no dataset, model accuracy, webcam or GUI evaluation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON evidence path; never overwritten",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Who ran this check (self-reported, not authentication)",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists. Choose a new path to preserve earlier evidence.")
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator,
        "status": "FAIL",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    exit_code = 1
    try:
        result["source_sha256"] = source_hashes(root)
        result.update(check_environment(root))
        result["status"] = "PASS"
        exit_code = 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
