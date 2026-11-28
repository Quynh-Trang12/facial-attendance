"""Experimental scoring for the two supplied LIV-04 checkpoints.

The release's inference code is unavailable. This trial uses the documented
LIV-03 input transform and the spoof index of a two-logit softmax. It exposes
scores and threshold comparisons, with no live/uncertain authorization policy.
"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path

from ..recognition.data_loader import image_transform


CHECKPOINTS = {
    "f60a8c8d7d2234cbff37b302b1d19e7c375c844e377df837452d4fc9e05091d7":
        ("Full", "liveness-resnet18-aenet-full", "ee6f1341785d22ad"),
    "ce07182f54cfeecb4c7b2de01dd45593fb55328c77e7773185573500c3876a10":
        ("Baseline", "liveness-resnet18-binary-control", "97208d8b15f8ac34"),
}
SCORING_METHOD = "experimental-softmax-spoof-index-1"


@dataclass(frozen=True)
class TrialScore:
    variant: str
    model_version: str
    score: float
    threshold: float

    @property
    def threshold_met(self):
        return self.score >= self.threshold


class LivenessCheckpointTrial:
    def __init__(self, paths):
        import torch
        from torchvision.models import resnet18

        self._models = {}
        self._transform = image_transform(224)
        for path in paths:
            data = Path(path).read_bytes()
            digest = sha256(data).hexdigest()
            if digest not in CHECKPOINTS or digest in self._models:
                raise ValueError("Use each of the two supplied liveness checkpoints at most once.")
            variant, model_id, version = CHECKPOINTS[digest]
            payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
            meta, state = payload["meta"], payload["state_dict"]
            threshold = meta.get("threshold")
            if (meta["schema"] != "liveness-checkpoint-v1"
                    or meta["architecture"] != "resnet18-aenet-binary-v1"
                    or meta["model_id"] != model_id or meta["model_version"] != version
                    or meta["label_mapping"] != {"live": 0, "spoof": 1} or meta["input_size"] != 224
                    or meta["mean"] != [0., 0., 0.] or meta["std"] != [1., 1., 1.]
                    or type(threshold) is not float or not isfinite(threshold) or not 0 <= threshold <= 1
                    or any(not key.startswith("backbone.") or not torch.isfinite(value).all().item()
                           for key, value in state.items())):
                raise ValueError("Unsupported liveness checkpoint metadata or weights.")
            model = resnet18(weights=None, num_classes=2)
            model.load_state_dict({key.removeprefix("backbone."): value for key, value in state.items()}, strict=True)
            self._models[digest] = (variant, version, threshold, model.eval())
        if not self._models:
            raise ValueError("Select at least one liveness checkpoint for the trial.")

    def __call__(self, crop):
        import numpy as np
        from PIL import Image
        import torch

        if (not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or crop.ndim != 3
                or crop.shape[2] != 3 or not crop.size):
            raise ValueError("The trial requires a single, nonempty uint8 BGR face crop.")
        rgb = Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
        scores = []
        with torch.inference_mode():
            tensor = self._transform(rgb).unsqueeze(0)
            for variant, version, threshold, model in self._models.values():
                logits = model(tensor)
                if logits.shape != (1, 2) or not torch.isfinite(logits).all().item():
                    raise RuntimeError("Invalid two-class liveness logits.")
                score = float(logits.softmax(dim=1)[0, 1])
                scores.append(TrialScore(variant, version, score, threshold))
        return tuple(scores)
