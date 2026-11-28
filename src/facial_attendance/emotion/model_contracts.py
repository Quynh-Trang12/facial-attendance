"""Label orders and identity for the supported trained Emotion checkpoints."""

LEGACY_LABELS = ("Negative", "Neutral", "Positive")
RESNET18_LABELS = ("Sad", "Happy", "Neutral")
RESNET18_MODEL_ID = "emotion-resnet18-v1"
RESNET18_SHA256 = "0dcecbdbc7195bfb8cfd4529e0aa287d9b40c43da323311ad3803d687fcbfd24"
RESNET18_INTERVAL = 3.0
DIAGNOSTIC_LABELS = (*LEGACY_LABELS, "Sad", "Happy")


def label_order(source, model_id, checkpoint_sha256):
    if model_id == RESNET18_MODEL_ID or checkpoint_sha256 == RESNET18_SHA256:
        if source == "model" and model_id == RESNET18_MODEL_ID and checkpoint_sha256 == RESNET18_SHA256:
            return RESNET18_LABELS
        return ()
    return LEGACY_LABELS
