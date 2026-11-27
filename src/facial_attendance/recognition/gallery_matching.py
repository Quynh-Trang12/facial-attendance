"""Model-compatible read snapshots, cosine matching and controlled test enrollment.

The webcam has no enrollment writer. Test enrollment only embeds internally
generated fixtures into a separate in-memory gallery; it is not live evidence.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np

from .contracts import EnrollmentContract, FaceInferenceContract
from .decision import decide, load_margin
from .inference import cosine, normalized_vector
from ..ui.gallery import FaceGallery, IncompatibleGalleryError

DECISION_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "recognition.toml"


@dataclass(frozen=True)
class Match:
    person_id: str | None
    name: str | None
    score: float | None
    accepted: bool


class GalleryMatcher:
    def __init__(self, gallery, margin=None):
        self._gallery = gallery
        self._lock = RLock()
        self.contract = FaceInferenceContract()
        # Top-1/top-2 margin from configs/recognition.toml unless a test passes one.
        self.margin = load_margin(DECISION_CONFIG) if margin is None else margin
        self._validate()

    def _validate(self):
        if (self._gallery.model_id, self._gallery.model_version, self._gallery.embedding_dim) != (
                self.contract.model_id, self.contract.model_version, self.contract.embedding_dim):
            raise IncompatibleGalleryError("Gallery does not match the selected Face checkpoint.")

    def _scores(self, embedding):
        """Cosine of one normalized embedding against each person's averaged template."""
        self.contract.validate_normalized_embedding(embedding)
        self._validate()
        scored = []
        for person in self._gallery.people:
            if len(person.embeddings) != 1:
                raise IncompatibleGalleryError("Gallery must contain one averaged template per identity.")
            scored.append((cosine(embedding, person.embeddings[0]), person))
        return scored

    def match(self, embedding):
        """Name a person only if the best score reaches tau and beats every other person by the margin.

        A best score at or above tau that is too close to the runner-up is unknown: the
        result keeps that score but no identity. With one person enrolled tau stands in for
        the runner-up, so the score must reach tau + margin.
        """
        with self._lock:
            scored = self._scores(embedding)
            if not scored:
                return Match(None, None, None, False)
            decision = decide([person.person_id for _, person in scored], [score for score, _ in scored],
                              self.contract.cosine_threshold, self.margin)
            if not decision.accepted:
                return Match(None, None, decision.best_score, False)
            person = next(person for _, person in scored if person.person_id == decision.label)
            return Match(person.person_id, person.name, decision.best_score, True)

    def match_threshold_only(self, embedding):
        """The best person whenever the score reaches tau, with no margin.

        Registration uses this to refuse a duplicate: a new face close to two enrolled
        people is still a duplicate, even though check-in would call it unknown.
        """
        with self._lock:
            scored = self._scores(embedding)
            if not scored:
                return Match(None, None, None, False)
            score, person = max(scored, key=lambda item: item[0])
            accepted = self.contract.accepts_cosine(score)
            return Match(person.person_id if accepted else None, person.name if accepted else None, score, accepted)

    def register_live(self, *args, **kwargs):
        raise PermissionError("Genuine enrollment is disabled: trained liveness authorization is unavailable.")


def empty_gallery():
    policy = FaceInferenceContract()
    return FaceGallery(model_id=policy.model_id, model_version=policy.model_version, embedding_dim=policy.embedding_dim)


def controlled_frame(sample=0):
    """Generated pattern, no downloaded/evaluation/person image or camera pixels."""
    y, x = np.indices((360, 640))
    frame = np.stack(((x//3+y//2+sample) % 256, (x//2+y//4+sample) % 256,
                      (x//4+y//3+sample) % 256), axis=-1).astype(np.uint8)
    frame.flags.writeable = False
    return frame


CONTROLLED_BOX = (160, 70, 480, 300)


@dataclass(frozen=True)
class PreparedTestEnrollment:
    person_id: str
    name: str
    embeddings: tuple


class ControlledTestEnrollment(GalleryMatcher):
    """Separate generated-fixture gallery. No API accepts live enrollment frames."""

    def __init__(self, inference):
        super().__init__(empty_gallery())
        self.inference = inference

    def prepare(self, identity):
        person_id, name = identity
        EnrollmentContract().validate_identity(person_id, name)
        vectors = tuple(self.inference.embed(controlled_frame(i), CONTROLLED_BOX) for i in range(5))
        return PreparedTestEnrollment(person_id, name, vectors)

    def commit(self, prepared):
        if not isinstance(prepared, PreparedTestEnrollment):
            raise ValueError("Expected a prepared controlled-test enrollment.")
        with self._lock:
            EnrollmentContract().validate_identity(prepared.person_id, prepared.name,
                                                   [person.person_id for person in self._gallery.people])
            EnrollmentContract().validate_sample_count(len(prepared.embeddings))
            for vector in prepared.embeddings:
                self.contract.validate_normalized_embedding(vector)
            template = normalized_vector(np.mean(prepared.embeddings, axis=0))
            if any(not self.contract.accepts_cosine(cosine(vector, template)) for vector in prepared.embeddings):
                raise ValueError("Controlled samples fail the UI-01 consistency guard; restart.")
            duplicate = self.match_threshold_only(template)
            if duplicate.accepted:
                raise ValueError(f"Already enrolled or indistinguishable: {duplicate.name}")
            return self._gallery.register(prepared.person_id, prepared.name, template,
                                          n_images=len(prepared.embeddings))
