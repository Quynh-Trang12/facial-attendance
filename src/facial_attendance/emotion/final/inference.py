"""Portable adapter: already cropped face -> Sad / Happy / Neutral scores."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from .model import ResNet18
except ImportError:
    from model import ResNet18

LABELS = ("Sad", "Happy", "Neutral")
CHECKPOINT_SHA256 = "0dcecbdbc7195bfb8cfd4529e0aa287d9b40c43da323311ad3803d687fcbfd24"


class EmotionPredictor:
    def __init__(self, checkpoint=None, device="cpu"):
        checkpoint = (
            Path(checkpoint)
            if checkpoint
            else Path(__file__).resolve().parents[4]
            / "artifacts/models/emotion/best.pt"
        )
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint differs from the tested release.")
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload["config"]["labels"] != ["sad", "happy", "neutral"]
            or payload["selected"]["epoch"] != 10
        ):
            raise ValueError("Unexpected checkpoint labels or selected epoch.")
        self.model = ResNet18(1000)
        self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 3))
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_rgb(self, face):
        """Accept a PIL RGB/L image or uint8 HxWx3 RGB array of ONE face crop."""
        if isinstance(face, Image.Image):
            if face.mode not in ("RGB", "L"):
                raise ValueError("PIL input must be RGB or grayscale L.")
            image = face
        else:
            a = np.asarray(face)
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
                raise ValueError("Expected uint8 HxWx3 RGB image.")
            image = Image.fromarray(np.ascontiguousarray(a))
        if min(image.size) < 1:
            raise ValueError("Empty face crop.")
        with torch.inference_mode():
            p = (
                self.model(self.transform(image).unsqueeze(0).to(self.device))
                .softmax(1)[0]
                .cpu()
                .numpy()
            )
        if not np.isfinite(p).all():
            raise RuntimeError("Invalid prediction values.")
        class_id = int(p.argmax())
        return {
            "status": "ok",
            "class_id": class_id,
            "label": LABELS[class_id],
            "confidence": float(p[class_id]),
            "probabilities": {label: float(score) for label, score in zip(LABELS, p)},
        }

    def predict_bgr(self, face_bgr):
        """OpenCV arrays use BGR; convert explicitly before training preprocessing."""
        a = np.asarray(face_bgr)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("Expected uint8 HxWx3 BGR image.")
        return self.predict_rgb(np.ascontiguousarray(a[:, :, ::-1]))
