"""Opt-in trained Face/Emotion preview with simulated liveness and no genuine writes."""

from ..recognition.contracts import FaceInferenceContract
from ..recognition.gallery_matching import (
    ControlledTestEnrollment,
    GalleryMatcher,
    controlled_frame,
    empty_gallery,
)
from ..recognition.inference import (
    FaceCheckpointMismatch,
    FaceInference,
    FaceUnavailable,
)
from .emotion_adapter import TrainedEmotionAdapter
from .gallery import FaceGallery, GalleryError
from .webcam_prototype import (
    CONTRACT_VERSION,
    Completion,
    FaceRequest,
    OperationError,
    Recognition,
    StubAdapters,
)


class TrainedRecognitionAdapter:
    def __init__(self, inference, matcher):
        self.inference, self.matcher = inference, matcher

    def __call__(self, request):
        if not isinstance(request, FaceRequest):
            raise ValueError("Recognition requires an original-context FaceRequest.")
        try:
            if request.context.contract_version != CONTRACT_VERSION:
                raise FaceCheckpointMismatch(
                    "Recognition request contract version differs."
                )
            vector = self.inference.embed(request.frame, request.bbox)
            match = self.matcher.match(vector)
            provenance = self.inference.provenance
            return Completion(
                request.context,
                Recognition(
                    match.person_id,
                    match.name,
                    match.score,
                    match.accepted,
                    provenance["model_id"],
                    provenance["model_version"],
                    source="model",
                    checkpoint_sha256=provenance["checkpoint_sha256"],
                    preprocessing_version=provenance["preprocessing_version"],
                ),
            )
        except FaceUnavailable as error:
            code, message = "model_unavailable", str(error)
        except FaceCheckpointMismatch as error:
            code, message = "version_mismatch", str(error)
        except GalleryError as error:
            code, message = "gallery_incompatible", str(error)
        except (ValueError, TypeError, AttributeError):
            code, message = (
                "invalid_input",
                "Invalid face source, crop, embedding or gallery template.",
            )
        except Exception:
            code, message = "inference_error", "Trained Face inference failed."
        return Completion(
            request.context, OperationError("recognition", code, message, True)
        )


class TrainedPreviewAdapters(StubAdapters):
    trained_recognition = True

    def __init__(
        self,
        face_checkpoint,
        *,
        emotion_checkpoint=None,
        gallery_path=None,
        controlled_registration=False,
        scenario="Live match",
        crop_to_detector_box=False,
    ):
        super().__init__(scenario)
        if controlled_registration and gallery_path is not None:
            raise ValueError("Controlled enrollment cannot modify an external gallery.")
        if crop_to_detector_box and gallery_path is not None:
            raise ValueError("Read-only galleries require the original square crop.")
        self.face_inference = FaceInference(
            face_checkpoint, crop_to_detector_box=crop_to_detector_box
        ).load()
        self.controlled_registration = controlled_registration
        if controlled_registration:
            self.matcher = ControlledTestEnrollment(self.face_inference)
        elif gallery_path is None:
            self.matcher = GalleryMatcher(empty_gallery())
        else:
            policy = FaceInferenceContract()
            self.matcher = GalleryMatcher(
                FaceGallery.load(
                    gallery_path,
                    model_id=policy.model_id,
                    model_version=policy.model_version,
                    embedding_dim=policy.embedding_dim,
                )
            )
        self.recognition_adapter = TrainedRecognitionAdapter(
            self.face_inference, self.matcher
        )
        self.trained_emotion = emotion_checkpoint is not None
        self.emotion_adapter = (
            TrainedEmotionAdapter(emotion_checkpoint) if self.trained_emotion else None
        )
        if self.emotion_adapter is not None:
            self.emotion_adapter.inference.load()

    def recognition(self, request):
        return self.recognition_adapter(request)

    def emotion(self, request):
        return (
            self.emotion_adapter(request)
            if self.emotion_adapter is not None
            else super().emotion(request)
        )


class ControlledCamera:
    """Explicit generated-input UI demo. Cannot open or read a physical camera."""

    def __init__(self):
        self.frame = controlled_frame()

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def release(self):
        pass
