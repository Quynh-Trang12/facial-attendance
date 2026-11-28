"""UI-03.1 contracts only: no checkpoint load, webcam recognition or gallery writes.

The selected policy is copied from FAC-05 validation at cddc07049d9e526f41e067c048548ed5a2a8fce2.
Pure validators make the design testable without pretending the face model is integrated.
"""

import math
from dataclasses import dataclass
from numbers import Real

SELECTED_MODEL_ID = "20261114-125242-metric-74a8657-s42"
SELECTED_MODEL_VERSION = "cb38ea47ceff"
SELECTED_COSINE_THRESHOLD = 0.4619967254593038


@dataclass(frozen=True)
class FaceInferenceContract:
    model_id: str = SELECTED_MODEL_ID
    model_version: str = SELECTED_MODEL_VERSION
    approach: str = "metric"
    input_size: int = 64
    embedding_dim: int = 512
    cosine_threshold: float = SELECTED_COSINE_THRESHOLD

    def validate_checkpoint_metadata(self, meta):
        expected = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "approach": self.approach,
            "input_size": self.input_size,
            "embedding_dim": self.embedding_dim,
            "architecture": "resnet18-3x3-stem-no-maxpool",
        }
        if not isinstance(meta, dict) or any(
            meta.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "Face checkpoint does not match the selected FAC-05 model."
            )
        for field, expected_values in (
            ("mean", [0.485, 0.456, 0.406]),
            ("std", [0.229, 0.224, 0.225]),
        ):
            value = meta.get(field)
            if not isinstance(value, (list, tuple)) or list(value) != expected_values:
                raise ValueError(
                    "Face checkpoint preprocessing differs from the shared input contract."
                )
        # FAC-05 selects after training. An unchanged best.pt may have no embedded
        # threshold; bind the separate selection record to its verified model ID/version.
        if (
            meta.get("threshold") is not None
            and meta["threshold"] != self.cosine_threshold
        ):
            raise ValueError(
                "Face checkpoint threshold conflicts with the FAC-05 selection."
            )

    def accepts_cosine(self, score):
        """Decision-rule contract test, not a trained recognition invocation."""
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
            or not -1 <= score <= 1
        ):
            raise ValueError("Cosine similarity must be finite in [-1,1].")
        return score >= self.cosine_threshold

    def validate_normalized_embedding(self, values):
        try:
            valid = len(values) == self.embedding_dim and all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in values
            )
        except TypeError:
            valid = False
        if not valid or not math.isclose(
            math.hypot(*values), 1.0, rel_tol=0, abs_tol=1e-3
        ):
            raise ValueError(
                "Gallery-facing embeddings must be finite, normalized 512D vectors."
            )


@dataclass(frozen=True)
class EnrollmentContract:
    gallery_schema: str = "0.2"
    target_samples: int = 5
    minimum_samples: int = 3
    maximum_samples: int = 10

    def validate_identity(self, person_id, name, existing_ids=()):
        for value in (person_id, name):
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    "Enrollment ID/name must be nonempty text without surrounding whitespace."
                )
        if len(name) > 64:
            raise ValueError("Enrollment name must contain 1–64 characters.")
        if person_id.casefold() in {value.casefold() for value in existing_ids}:
            raise ValueError("An existing person ID cannot be overwritten.")

    def validate_sample_count(self, count):
        if (
            type(count) is not int
            or not self.minimum_samples <= count <= self.maximum_samples
        ):
            raise ValueError(
                "Enrollment requires 3–10 independently live, accepted samples."
            )


def identity_display_state(*, liveness, accepted):
    """UI state contract only; eligibility never follows this string alone."""
    if liveness not in ("live", "spoof", "uncertain"):
        return "unavailable"
    if liveness != "live":
        return "blocked"
    if type(accepted) is not bool:
        return "processing"
    return "known_name" if accepted else "unknown_registration_prompt"
