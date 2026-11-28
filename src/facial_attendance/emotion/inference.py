"""CPU inference for the supplied, hash-pinned EmotionCNN checkpoint.

Architecture and weight-version algorithm follow feature/emotion-eda at
284c48310e0335ed4b9ce4f641389c44b7c6841c. Training/optimizer state is not restored.
Torch is imported only when this optional model is used.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path
import re

CHECKPOINT_SHA256 = "323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd"
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
DISPLAY_LABELS = ("Negative", "Neutral", "Positive")
PREPROCESSING_VERSION = "pillow-bgr-rgb-gray48-bilinear-div255-v1"


class EmotionUnavailable(RuntimeError):
    """The optional runtime or checkpoint cannot be read safely."""


class EmotionCheckpointMismatch(ValueError):
    """File hash, checkpoint metadata or state tensors are incompatible."""


class EmotionInferenceError(RuntimeError):
    """The model did not return valid finite three-class logits."""


@dataclass(frozen=True)
class EmotionPrediction:
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, float, float]


def weights_version(state_dict):
    """Same name/dtype/shape/bytes digest as the owner's training.weights_hash."""
    digest = sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_payload(payload, model):
    import torch

    if not isinstance(payload, dict):
        raise EmotionCheckpointMismatch("Expected a checkpoint dictionary.")
    meta, config, state = (payload.get(key) for key in ("meta", "config", "state_dict"))
    if not all(isinstance(value, dict) for value in (meta, config, state)):
        raise EmotionCheckpointMismatch("Missing checkpoint metadata, config or state_dict.")
    expected = {"architecture": "EmotionCNN", "input_size": 48, "channels": 1, "scale": 255,
                "threshold": None, "threshold_source": "not_applicable_argmax"}
    if any(key not in meta or type(meta[key]) is not type(value) or meta[key] != value
           for key, value in expected.items()):
        raise EmotionCheckpointMismatch("Emotion architecture/preprocessing metadata differs.")
    if (meta.get("label_mapping") != LABEL_MAPPING
            or any(type(value) is not int for value in meta["label_mapping"].values())):
        raise EmotionCheckpointMismatch("Emotion label mapping differs from training.")
    if (not isinstance(meta.get("model_id"), str) or not meta["model_id"].strip()
            or not isinstance(meta.get("model_version"), str)
            or re.fullmatch(r"[0-9a-f]{64}", meta["model_version"]) is None):
        raise EmotionCheckpointMismatch("Missing model identity/version.")
    model_config = config.get("model")
    if (not isinstance(model_config, dict) or model_config.get("channels") != [32, 64, 128]
            or model_config.get("dropout") != 0.3):
        raise EmotionCheckpointMismatch("Expected channels [32,64,128] and dropout 0.3.")
    if type(payload.get("epoch")) is not int or payload["epoch"] != 9:
        raise EmotionCheckpointMismatch("This selected checkpoint must record epoch 9.")
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise EmotionCheckpointMismatch("Checkpoint state keys do not match EmotionCNN.")
    for name, tensor in state.items():
        expected_tensor = expected_state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected_tensor.dtype
                or tensor.shape != expected_tensor.shape or not torch.isfinite(tensor).all()):
            raise EmotionCheckpointMismatch(f"Invalid EmotionCNN tensor: {name}")
    if weights_version(state) != meta["model_version"]:
        raise EmotionCheckpointMismatch("Checkpoint weights differ from model_version.")
    return meta


def preprocess_bgr_crop(crop):
    """Copy BGR pixels; Pillow grayscale/bilinear matches the training loader."""
    import numpy as np
    import torch
    from PIL import Image

    if (not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or crop.ndim != 3
            or crop.shape[2] != 3 or min(crop.shape[:2]) < 1):
        raise ValueError("Expected a nonempty uint8 BGR face crop.")
    rgb = np.array(crop[..., ::-1], copy=True)
    image = Image.fromarray(rgb).convert("L").resize((48, 48), Image.Resampling.BILINEAR)
    pixels = np.array(image, dtype=np.float32, copy=True) / np.float32(255)
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).contiguous()


def prediction_from_logits(logits):
    import torch

    if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 3)
            or logits.dtype != torch.float32 or not torch.isfinite(logits).all()):
        raise EmotionInferenceError("Expected finite float32 logits of shape [1,3].")
    probabilities = torch.softmax(logits, dim=1).detach().cpu()[0]
    if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(), torch.tensor(1.0)):
        raise EmotionInferenceError("Invalid emotion softmax probabilities.")
    class_id = int(probabilities.argmax())
    return EmotionPrediction(class_id, DISPLAY_LABELS[class_id], float(probabilities[class_id]),
                             tuple(float(value) for value in probabilities))


class EmotionInference:
    """Load once on the adapter worker; inference is eval-mode, CPU and read-only.

    expected_sha256 can be injected by isolated test fixtures. The application
    CLI always uses the supplied checkpoint's pinned CHECKPOINT_SHA256.
    """

    def __init__(self, checkpoint, *, expected_sha256=CHECKPOINT_SHA256):
        self.checkpoint = Path(checkpoint)
        self.expected_sha256 = expected_sha256
        self.model = None
        self.provenance = None

    def load(self):
        if self.model is not None:
            return self
        try:
            data = self.checkpoint.read_bytes()
        except OSError as error:
            raise EmotionUnavailable("Emotion checkpoint is unavailable; provide a local file.") from error
        digest = sha256(data).hexdigest()
        if digest != self.expected_sha256:
            raise EmotionCheckpointMismatch("Emotion checkpoint SHA-256 mismatch.")
        try:
            import torch
            from .model import EmotionCNN
        except ImportError as error:
            raise EmotionUnavailable("PyTorch Emotion runtime is unavailable.") from error
        try:
            torch.set_num_threads(1)
            # Load exactly the bytes whose hash was checked; no path race or
            # arbitrary pickle fallback, no optimizer/RNG restoration.
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            model = EmotionCNN(channels=(32, 64, 128), dropout=0.3)
            meta = validate_payload(payload, model)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
        except EmotionCheckpointMismatch:
            raise
        except Exception as error:
            raise EmotionCheckpointMismatch("Cannot safely load compatible EmotionCNN weights.") from error
        self.provenance = {"source": "model", "architecture": "EmotionCNN", "model_id": meta["model_id"],
                           "model_version": meta["model_version"], "checkpoint_sha256": digest,
                           "preprocessing_version": PREPROCESSING_VERSION, "selected_epoch": payload["epoch"]}
        self.model = model
        return self

    def predict_bgr(self, crop):
        self.load()
        import torch
        tensor = preprocess_bgr_crop(crop)
        with torch.inference_mode():
            return prediction_from_logits(self.model(tensor))
