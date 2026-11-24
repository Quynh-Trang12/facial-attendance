"""CPU inference for Dung's pinned ResNet-18 emotion handover."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from .inference import EmotionCheckpointMismatch, EmotionPrediction, EmotionUnavailable, weights_version
from .model_contracts import RESNET18_INTERVAL, RESNET18_LABELS, RESNET18_MODEL_ID, RESNET18_SHA256


class ResNetEmotionInference:
    labels = RESNET18_LABELS
    prediction_interval = RESNET18_INTERVAL
    architecture = "ResNet-18"

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.model = self.transform = self.provenance = None

    def load(self):
        if self.model is not None:
            return self
        try:
            data = self.checkpoint.read_bytes()
        except OSError as error:
            raise EmotionUnavailable("ResNet-18 Emotion checkpoint is unavailable.") from error
        digest = sha256(data).hexdigest()
        if digest != RESNET18_SHA256:
            raise EmotionCheckpointMismatch("ResNet-18 Emotion checkpoint SHA-256 mismatch.")
        try:
            import torch
            from torch import nn
            from torchvision import models, transforms
        except (ImportError, RuntimeError) as error:
            raise EmotionUnavailable("Compatible torch and torchvision are required for ResNet-18 Emotion.") from error

        try:
            torch.set_num_threads(1)
            payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
            if payload["config"]["labels"] != ["sad", "happy", "neutral"] or payload["selected"]["epoch"] != 10:
                raise ValueError("Unexpected training label order or epoch.")
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 3))
            state = payload["state_dict"]
            if any(not torch.isfinite(value).all().item() for value in state.values()):
                raise ValueError("Nonfinite model tensor.")
            model.load_state_dict(state, strict=True)
            model.eval()
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            version = weights_version(state)
        except Exception as error:
            raise EmotionCheckpointMismatch("Cannot safely load compatible ResNet-18 Emotion weights.") from error

        self.model, self.transform = model, transform
        self.provenance = {
            "source": "model", "architecture": self.architecture, "model_id": RESNET18_MODEL_ID,
            "model_version": version, "checkpoint_sha256": digest,
            "preprocessing_version": "pillow-bgr-rgb-gray3-224-bilinear-imagenet-v1",
            "selected_epoch": 10,
        }
        return self

    def predict_bgr(self, crop):
        import numpy as np
        import torch
        from PIL import Image

        if (not isinstance(crop, np.ndarray) or crop.dtype != np.uint8
                or crop.ndim != 3 or crop.shape[2] != 3 or not crop.size):
            raise ValueError("Expected a nonempty uint8 BGR face crop.")
        self.load()
        image = Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
        with torch.inference_mode():
            logits = self.model(self.transform(image).unsqueeze(0))
            if logits.shape != (1, 3) or not torch.isfinite(logits).all().item():
                raise RuntimeError("Invalid ResNet-18 Emotion logits.")
            probabilities = logits.softmax(dim=1)[0]
            class_id = int(probabilities.argmax())
            return EmotionPrediction(class_id, self.labels[class_id], float(probabilities[class_id]),
                                     tuple(float(value) for value in probabilities))
