"""ML runtime smoke checks; import this module only when executing those checks.

Environment/version policy and CLI parsing remain usable without ML packages.
The ML stack is imported once here; computation helpers do not import packages.
"""

# matplotlib/tqdm imports intentionally smoke-test the remaining direct dependencies.
import cv2
import matplotlib  # noqa: F401
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torchvision
import tqdm  # noqa: F401

from .environment import require, select_device


def check_training_step(device: torch.device) -> tuple[dict, torch.Tensor]:
    """Prove that autograd and the optimizer work, not merely that a model imports."""
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


def check_image_and_metric_apis(output: torch.Tensor) -> None:
    """Exercise array/image interoperability without reporting a fake project AUC."""
    # Distinct channels catch a missing/reversed BGR-to-RGB conversion; zeros cannot.
    bgr = np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = torchvision.transforms.functional.to_tensor(image)
    require(
        tensor.shape == (3, 8, 8)
        and np.array_equal(rgb[0, 0], [30, 20, 10])
        and torch.allclose(tensor[:, 0, 0], torch.tensor([30, 20, 10]) / 255),
        "Image conversion failed",
    )
    require(np.isfinite(output.cpu().numpy()).all(), "Tensor/NumPy conversion failed")
    require(roc_auc_score([0, 1], [0.1, 0.9]) == 1.0, "Metric API check failed")


def check_runtime(requested_device: str) -> dict:
    """Run computation on the explicit device, with no automatic CPU fallback."""
    selected = torch.device(
        select_device(requested_device, cuda_available=torch.cuda.is_available())
    )
    computation, output = check_training_step(selected)
    check_image_and_metric_apis(output)
    return {
        **computation,
        "requested_device": requested_device,
        "device": str(selected),
        "gpu": torch.cuda.get_device_name(selected)
        if selected.type == "cuda"
        else None,
        "torch_cuda_build": torch.version.cuda,
    }
