"""Face embedding network, shared preprocessing and the deployable checkpoint format.

Both face approaches use this network, so a comparison measures the training objective
rather than the architecture. The embedding is the linear layer before the identity
head; the head exists only to train the classification approach.
"""

from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

ARCHITECTURE = "resnet18-3x3-stem-no-maxpool"
INPUT_SIZE = 64
EMBEDDING_DIM = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
META_KEYS = (
    "model_id",
    "model_version",
    "architecture",
    "approach",
    "embedding_dim",
    "input_size",
    "mean",
    "std",
    "threshold",
    "threshold_source",
    "num_classes",
)


class FaceEmbeddingNet(nn.Module):
    """ResNet-18 adapted to 64-pixel faces, a linear embedding and an optional head."""

    def __init__(self, num_classes: int | None, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        # The ImageNet stem (7x7 stride 2, then max-pool) would shrink a 64-pixel face
        # to 16x16 before the first block; a 3x3 stride-1 stem keeps the spatial detail.
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()  # global average pooling output, 512 values
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes) if num_classes else None
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Raw embedding before any normalization; scoring decides how to normalize."""
        return self.embedding(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("This network has no identity head; call embed()")
        return self.head(self.embed(images))


def eval_transform(size: int = INPUT_SIZE):
    """The one inference path for dataset images and webcam crops (embedding schema)."""
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    return Compose([Resize((size, size)), ToTensor(), Normalize(MEAN, STD)])


def train_transform(size: int = INPUT_SIZE):
    """Flip plus 4-pixel pad and random crop, then the same scaling as inference."""
    from torchvision.transforms import (
        Compose,
        Normalize,
        RandomCrop,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
    )

    return Compose(
        [
            Resize((size, size)),
            RandomHorizontalFlip(),
            RandomCrop(size, padding=4),
            ToTensor(),
            Normalize(MEAN, STD),
        ]
    )


def model_version(state_dict: dict) -> str:
    """First 12 hex digits of SHA-256 over each tensor's name and bytes, in name order.

    It hashes the weights, not the file, so writing a threshold into meta later does not
    change the version a gallery was built with.
    """
    digest = sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:12]


def build_meta(model: FaceEmbeddingNet, *, model_id: str, approach: str) -> dict:
    state = model.state_dict()
    return {
        "model_id": model_id,
        "model_version": model_version(state),
        "architecture": ARCHITECTURE,
        "approach": approach,
        "embedding_dim": model.embedding_dim,
        "input_size": INPUT_SIZE,
        "mean": list(MEAN),
        "std": list(STD),
        # FAC-05 chooses the deployed threshold on validation pairs, never here.
        "threshold": None,
        "threshold_source": None,
        "num_classes": model.num_classes,
    }


def save_checkpoint(file: Path, model: FaceEmbeddingNet, meta: dict):
    """Write the deployable format: state_dict plus meta, nothing needed to resume."""
    missing = set(META_KEYS) - set(meta)
    if missing:
        raise ValueError(f"Checkpoint meta lacks {sorted(missing)}")
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "meta": meta}, temporary)
    # Replace atomically, so a Colab disconnect never leaves half a checkpoint.
    temporary.replace(file)


def load_checkpoint(file: Path, device: str | torch.device = "cpu"):
    """Rebuild the network from meta and refuse weights that do not match it."""
    content = torch.load(Path(file), map_location=device, weights_only=True)
    meta = content["meta"]
    if meta.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unsupported architecture {meta.get('architecture')!r}")
    model = FaceEmbeddingNet(meta["num_classes"], meta["embedding_dim"])
    model.load_state_dict(content["state_dict"])
    if model_version(model.state_dict()) != meta["model_version"]:
        raise ValueError("Checkpoint weights do not match meta model_version")
    return model.to(device).eval(), meta
