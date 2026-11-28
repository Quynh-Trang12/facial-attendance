"""Model-compatible read snapshots, cosine matching and controlled test enrollment.

The webcam has no enrollment writer. Test enrollment only embeds internally
generated fixtures into a separate in-memory gallery; it is not live evidence.
"""

from dataclasses import dataclass
from threading import RLock

import numpy as np

from .contracts import EnrollmentContract, FaceInferenceContract
from .inference import cosine, normalized_vector
from ..ui.gallery import FaceGallery, IncompatibleGalleryError


@dataclass(frozen=True)
class Match:
    person_id: str | None
    name: str | None
    score: float | None
    accepted: bool


class GalleryMatcher:
    def __init__(self, gallery):
        self._gallery = gallery
        self._lock = RLock()
        self.contract = FaceInferenceContract()
        self._validate()

    def _validate(self):
        if (self._gallery.model_id, self._gallery.model_version, self._gallery.embedding_dim) != (
                self.contract.model_id, self.contract.model_version, self.contract.embedding_dim):
            raise IncompatibleGalleryError("Gallery does not match the selected Face checkpoint.")

    def match(self, embedding):
        self.contract.validate_normalized_embedding(embedding)
        with self._lock:
            self._validate()
            best = None
            for person in self._gallery.people:
                if len(person.embeddings) != 1:
                    raise IncompatibleGalleryError("Gallery must contain one averaged template per identity.")
                score = cosine(embedding, person.embeddings[0])
                if best is None or score > best[0]:
                    best = score, person
            if best is None:
                return Match(None, None, None, False)
            score, person = best
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
            duplicate = self.match(template)
            if duplicate.accepted:
                raise ValueError(f"Already enrolled or indistinguishable: {duplicate.name}")
            return self._gallery.register(prepared.person_id, prepared.name, template,
                                          n_images=len(prepared.embeddings))
