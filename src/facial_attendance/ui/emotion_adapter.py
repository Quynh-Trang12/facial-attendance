"""Optional trained Emotion completion; recognition and liveness stay stubs."""

from hashlib import file_digest
from pathlib import Path

from ..emotion.inference import (
    EmotionCheckpointMismatch,
    EmotionInference,
    EmotionUnavailable,
)
from ..emotion.model_contracts import RESNET18_SHA256
from ..emotion.resnet_inference import ResNetEmotionInference
from .webcam_prototype import (
    CONTRACT_VERSION,
    Completion,
    Emotion,
    FaceRequest,
    OperationError,
    StubAdapters,
    box_in_frame,
    frame_size,
)


class TrainedEmotionAdapter:
    def __init__(self, checkpoint, *, inference=None):
        if inference is None:
            try:
                with Path(checkpoint).open("rb") as handle:
                    digest = file_digest(handle, "sha256").hexdigest()
            except OSError:
                digest = None  # The loader reports unavailable files through the existing typed error.
            loader = (
                ResNetEmotionInference
                if digest == RESNET18_SHA256
                else EmotionInference
            )
            inference = loader(checkpoint)
        self.inference = inference

    def __call__(self, request: FaceRequest) -> Completion:
        if not isinstance(request, FaceRequest):
            raise ValueError("Emotion requires a context-bound FaceRequest.")
        try:
            if request.context.contract_version != CONTRACT_VERSION:
                return Completion(
                    request.context,
                    OperationError(
                        "emotion",
                        "version_mismatch",
                        "Emotion request contract version differs.",
                        False,
                    ),
                )
            width, height = frame_size(request.frame)
            if not box_in_frame(request.bbox, width, height):
                raise ValueError("Invalid source-frame emotion crop.")
            prediction = self.inference.predict_bgr(request.face_crop)
            provenance = self.inference.provenance
            return Completion(
                request.context,
                Emotion(
                    prediction.class_id,
                    prediction.label,
                    prediction.confidence,
                    source="model",
                    model_id=provenance["model_id"],
                    model_version=provenance["model_version"],
                    checkpoint_sha256=provenance["checkpoint_sha256"],
                    preprocessing_version=provenance["preprocessing_version"],
                    selected_epoch=provenance["selected_epoch"],
                ),
            )
        except EmotionUnavailable as error:
            code, message = "model_unavailable", str(error)
        except EmotionCheckpointMismatch as error:
            code, message = "version_mismatch", str(error)
        except (ValueError, TypeError, AttributeError):
            code, message = (
                "invalid_input",
                "Invalid context-bound emotion source or crop.",
            )
        except Exception:
            code, message = "inference_error", "Trained emotion inference failed."
        return Completion(
            request.context, OperationError("emotion", code, message, False)
        )


class MixedEmotionAdapters(StubAdapters):
    """Explicit partial integration, with no trained liveness or recognition claim."""

    trained_emotion = True

    def __init__(self, scenario, checkpoint):
        super().__init__(scenario)
        self.emotion_adapter = TrainedEmotionAdapter(checkpoint)

    def emotion(self, request):
        return self.emotion_adapter(request)
