"""Selected FAC-04 CPU checkpoint and SYS-02 source-frame inference boundary."""

from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from .contracts import FaceInferenceContract

CHECKPOINT_SHA256 = "bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f"
PREPROCESSING_VERSION = "square1.2-clipped-bgr-rgb64-bilinear-imagenet-v1"


class FaceUnavailable(RuntimeError):
    pass


class FaceCheckpointMismatch(ValueError):
    pass


class FaceInferenceError(RuntimeError):
    pass


def crop_box(frame, bbox):
    from ..ui.webcam_prototype import box_in_frame, frame_size
    width, height = frame_size(frame)
    if not isinstance(frame, np.ndarray) or not box_in_frame(bbox, width, height):
        raise ValueError("Invalid original-frame face region.")
    x1, y1, x2, y2 = bbox
    if min(x2 - x1, y2 - y1) < 64:
        raise ValueError("Face box must be at least 64 pixels on its shorter side.")
    side = 1.2 * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
            min(width, math.ceil(cx+side/2)), min(height, math.ceil(cy+side/2)))


def normalized_vector(values):
    array = np.asarray(values)
    if array.shape != (512,) or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError("Expected a finite 512D numeric embedding.")
    norm = math.hypot(*(float(v) for v in array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero/invalid embedding.")
    return tuple(float(value) / norm for value in array)


def cosine(left, right):
    contract = FaceInferenceContract()
    contract.validate_normalized_embedding(left)
    contract.validate_normalized_embedding(right)
    score = math.fsum(float(a)*float(b) for a, b in zip(left, right))
    score /= math.hypot(*left) * math.hypot(*right)
    return max(-1.0, min(1.0, score))


class FaceInference:
    """Load once before Tk construction; all subsequent forwards run on workers."""

    def __init__(self, checkpoint, *, expected_sha256=CHECKPOINT_SHA256):
        self.checkpoint = Path(checkpoint)
        self.expected_sha256 = expected_sha256
        self.model = self.transform = self.provenance = None
        self.contract = FaceInferenceContract()

    def load(self):
        if self.model is not None:
            return self
        try:
            before = sha256(self.checkpoint.read_bytes()).hexdigest()
        except OSError as error:
            raise FaceUnavailable("Face checkpoint is unavailable; supply a local file.") from error
        if before != self.expected_sha256:
            raise FaceCheckpointMismatch("Face checkpoint file SHA-256 mismatch.")
        try:
            import torch
            from .model import load_checkpoint, model_version, eval_transform
        except (ImportError, RuntimeError) as error:
            raise FaceUnavailable("Compatible torch/torchvision Face runtime is unavailable.") from error
        try:
            torch.set_num_threads(1)
            # Reuse the team's weights-only CPU loader and explicit network.
            model, meta = load_checkpoint(self.checkpoint, device="cpu")
            self.contract.validate_checkpoint_metadata(meta)
            if meta.get("num_classes") is not None or model.head is not None:
                raise ValueError("Selected metric checkpoint must have no classification head.")
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("Face checkpoint has nonfinite tensors.")
            if model_version(model.state_dict()) != self.contract.model_version:
                raise ValueError("Face weights do not match selected FAC-05 version.")
            if sha256(self.checkpoint.read_bytes()).hexdigest() != before:
                raise ValueError("Face checkpoint changed during loading.")
            transform = eval_transform(meta["input_size"])
        except Exception as error:
            raise FaceCheckpointMismatch(f"Incompatible selected Face checkpoint: {error}") from error
        self.model, self.transform = model, transform
        self.provenance = {"source": "model", "model_id": meta["model_id"], "model_version": meta["model_version"],
                           "checkpoint_sha256": before, "architecture": meta["architecture"],
                           "preprocessing_version": PREPROCESSING_VERSION, "threshold": self.contract.cosine_threshold,
                           "checkpoint_epoch": None, "epoch_note": "Deployable checkpoint contains no epoch field; reported selection epoch 29 is not independently verified from this file."}
        return self

    def preprocess(self, frame, bbox):
        from PIL import Image
        self.load()
        x1, y1, x2, y2 = crop_box(frame, bbox)
        rgb = np.array(frame[y1:y2, x1:x2, ::-1], copy=True)
        return self.transform(Image.fromarray(rgb)).unsqueeze(0)

    def embed(self, frame, bbox):
        import torch
        tensor = self.preprocess(frame, bbox)
        with torch.inference_mode():
            raw = self.model.embed(tensor)
        if (not isinstance(raw, torch.Tensor) or raw.shape != (1, 512)
                or raw.dtype != torch.float32 or not torch.isfinite(raw).all()):
            raise FaceInferenceError("Expected finite float32 face embeddings [1,512].")
        try:
            return normalized_vector(raw.detach().cpu().numpy()[0])
        except ValueError as error:
            raise FaceInferenceError(str(error)) from error
