"""Trainable live/spoof baseline and the checkpoint consumed by integration."""

from hashlib import sha256
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

ARCHITECTURE = "mobilenet-v3-small-binary-v1"
INPUT_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
RESNET_ARCHITECTURE = "resnet18-binary-v1"
AENET_ARCHITECTURE = "resnet18-aenet-binary-v1"
RESNET_INITIALIZATION_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET_INITIALIZATION_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)


class ResNetBinaryClassifier(nn.Module):
    """Two-class CE model exported as a spoof logit for the existing adapter."""

    def __init__(self, imagenet_weights=None):
        super().__init__()
        self.backbone = resnet18(weights=None)
        if imagenet_weights is not None:
            state = torch.load(imagenet_weights, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def feature_map(self, images):
        """Expose the spatial embedding used by geometric supervision."""
        network = self.backbone
        features = network.maxpool(network.relu(network.bn1(network.conv1(images))))
        for layer in (network.layer1, network.layer2, network.layer3, network.layer4):
            features = layer(features)
        return features

    def encode(self, images):
        """Pool spatial features into one embedding per image for the binary and auxiliary heads."""
        return self.backbone.avgpool(self.feature_map(images)).flatten(1)

    def forward(self, images):
        """Return one spoof logit per image, shaped [batch, 1], from the two-class classifier."""
        logits = self.backbone.fc(self.encode(images))
        # sigmoid(spoof-live) equals softmax(two class logits)[spoof].
        return (logits[:, 1] - logits[:, 0]).unsqueeze(1)


def normalization(architecture):
    """Return the normalization recorded by the selected checkpoint architecture."""
    if architecture == ARCHITECTURE:
        return MEAN, STD
    if architecture == RESNET_ARCHITECTURE:
        return RESNET_MEAN, RESNET_STD
    if architecture == AENET_ARCHITECTURE:
        # The released AENet inference uses Resize + ToTensor, without Normalize.
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    raise ValueError(f"Unsupported liveness architecture: {architecture}")


def build_model(architecture=ARCHITECTURE, *, imagenet_weights=None):
    """Build locally; pretrained initialization requires an explicit weight file."""
    normalization(architecture)
    if architecture in (RESNET_ARCHITECTURE, AENET_ARCHITECTURE):
        return ResNetBinaryClassifier(imagenet_weights)
    if imagenet_weights is not None:
        raise ValueError("ImageNet initialization is supported only for ResNet-18")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def face_transform(training=False, *, architecture=ARCHITECTURE):
    """Resize an RGB face and apply architecture-specific tensor normalization.

    Random flipping is training-only; inference keeps the checkpoint recipe fixed.
    """
    mean, std = normalization(architecture)
    transforms = [Resize((INPUT_SIZE, INPUT_SIZE))]
    if training:
        transforms.append(RandomHorizontalFlip())
    return Compose([*transforms, ToTensor(), Normalize(mean, std)])


def state_version(state):
    """Hash tensor names, shapes, dtypes and bytes to identify the actual weights."""
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()[:16]


def save_checkpoint(path, state, meta, **resume):
    """Serialize model metadata and any supplied optimizer/RNG resume state."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": state, "meta": meta, **resume}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", *, require_threshold=True):
    """Load verified architecture/label metadata and optionally require a frozen threshold.

    Unknown architectures or inconsistent metadata stop loading before prediction.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = checkpoint["meta"]
    if meta.get("inference_precision", "float32") not in ("float32", "float16"):
        raise ValueError("Unsupported checkpoint inference precision")
    mean, std = normalization(meta.get("architecture"))
    if (
        meta.get("label_mapping") != {"live": 0, "spoof": 1}
        or meta.get("input_size") != INPUT_SIZE
        or meta.get("mean") != list(mean)
        or meta.get("std") != list(std)
    ):
        raise ValueError("Unsupported liveness checkpoint contract")
    if meta.get("model_version") != state_version(checkpoint["state_dict"]):
        raise ValueError("Checkpoint weights do not match model_version")
    threshold = meta.get("threshold")
    if require_threshold and (
        not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
        or meta.get("threshold_source") != "validation EER operating point"
    ):
        raise ValueError("Deployment checkpoint needs a validation-selected threshold")
    if meta.get("optimizer_updates", 0) < 1:
        raise ValueError("Untrained weights are not a deployable anti-spoofing model")
    model = build_model(meta["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), meta


class LivenessPredictor:
    """Consume one upstream face crop in PIL RGB; no capture ID is inferred."""

    def __init__(self, checkpoint, device="cpu"):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.device = torch.device(device)
        self.transform = face_transform(architecture=self.meta["architecture"])

    @torch.inference_mode()
    def predict_face(self, image):
        """Transform one RGB crop and compare its spoof probability with the saved threshold."""
        if image.mode != "RGB":
            raise ValueError("Integration input must be a PIL RGB face crop")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        # Match evaluation: autocast may produce half logits, but the score
        # and threshold comparison use float32, avoiding extra rounding.
        score = torch.sigmoid(self.model(tensor).reshape(-1).float())[0].item()
        if not 0 <= score <= 1:
            raise ValueError("Nonfinite spoof score")
        spoof = score >= self.meta["threshold"]
        return {
            "status": "spoof" if spoof else "live",
            "spoof_score": score,
            "live_score": 1 - score,
            "threshold": self.meta["threshold"],
            "allow_attendance": not spoof,
            "model_id": self.meta["model_id"],
            "model_version": self.meta["model_version"],
        }
