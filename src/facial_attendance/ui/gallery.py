"""Versioned, model-independent gallery for UI-01.2 / UI-01.4.

The caller owns liveness, capture quality, embedding inference, normalization
and the selection of the model-specific matching threshold.  Each registered
identity stores one already-averaged, L2-normalized embedding template.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "0.2"
MIN_ENROLLMENT_IMAGES = 3
MAX_ENROLLMENT_IMAGES = 10
NORM_TOLERANCE = 1e-3


class GalleryError(Exception):
    """Base exception for gallery operations."""


class DuplicatePersonIDError(GalleryError):
    """Registration would reuse an existing ID, including a case variant."""


class UnknownPersonIDError(GalleryError):
    """An explicit template update targets an unregistered person."""


class GalleryFormatError(GalleryError):
    """Stored gallery data is invalid or uses an unsupported schema."""


class IncompatibleGalleryError(GalleryError):
    """Gallery embeddings do not match the selected recognition checkpoint."""


class GalleryStorageError(GalleryError):
    """A filesystem operation prevented saving or reading the gallery."""


@dataclass(frozen=True)
class Identity:
    """One registered person with one averaged embedding template."""

    person_id: str
    name: str
    embeddings: tuple[tuple[float, ...], ...]
    created_at: str
    n_images: int


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding spaces")
    return value


def _person_key(person_id: str) -> str:
    """Compare identity IDs without case while retaining original display text."""
    return _required_text(person_id, "person_id").casefold()


def _timestamp(value: str | None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    else:
        value = _required_text(value, "created_at")
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO 8601 timestamp") from exc
        if timestamp.utcoffset() is None:
            raise ValueError("created_at must contain a timezone offset")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _image_count(value: Any) -> int:
    if type(value) is not int or not MIN_ENROLLMENT_IMAGES <= value <= MAX_ENROLLMENT_IMAGES:
        raise ValueError("n_images must be an integer between 3 and 10")
    return value


def _vector(value: Any, dimension: int) -> tuple[float, ...]:
    """Accept 1-D Python or NumPy vectors without requiring NumPy as a dependency."""
    if isinstance(value, (str, bytes, bytearray)) or getattr(value, "ndim", 1) != 1:
        raise ValueError("embedding must be a one-dimensional numeric vector")
    try:
        components = list(value)
    except TypeError as exc:
        raise ValueError("embedding must be an iterable vector") from exc
    if len(components) != dimension:
        raise ValueError(f"embedding must contain exactly {dimension} numeric values")
    result = []
    for component in components:
        if isinstance(component, (str, bytes, bytearray, bool)) or type(component).__name__ == "bool_":
            raise ValueError("embedding values must be numeric, not bool or text")
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding values must be convertible to finite floats") from exc
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite")
        result.append(number)
    norm = math.sqrt(math.fsum(number * number for number in result))
    if not math.isfinite(norm) or abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError("embedding template must be L2-normalized")
    return tuple(result)


class FaceGallery:
    """Store averaged templates with checkpoint compatibility and atomic persistence.

    The caller passes an already-computed normalized template and the number of
    individually accepted live images that produced it.  The gallery does not
    itself infer embeddings, evaluate liveness or match against a threshold.

    IDs retain their spelling, but casefold-equal IDs refer to the same person.
    A failed save restores the last successfully persisted in-memory snapshot.
    """

    def __init__(self, *, model_id: str, model_version: str, embedding_dim: int):
        self.model_id = _required_text(model_id, "model_id")
        self.model_version = _required_text(model_version, "model_version")
        if type(embedding_dim) is not int or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        self.embedding_dim = embedding_dim
        self._people: dict[str, Identity] = {}
        self._committed_people: dict[str, Identity] = {}

    @property
    def people(self) -> tuple[Identity, ...]:
        return tuple(self._people[key] for key in sorted(self._people))

    def get(self, person_id: str) -> Identity | None:
        return self._people.get(_person_key(person_id))

    def register(
        self,
        person_id: str,
        name: str,
        embedding: Any,
        *,
        n_images: int,
        created_at: str | None = None,
    ) -> Identity:
        """Stage a new identity; call save() before reporting enrollment success."""
        person_id = _required_text(person_id, "person_id")
        name = _required_text(name, "name")
        if len(name) > 64:
            raise ValueError("name must contain at most 64 characters")
        key = _person_key(person_id)
        if key in self._people:
            raise DuplicatePersonIDError(f"person_id {person_id!r} already exists")
        vector = _vector(embedding, self.embedding_dim)
        count = _image_count(n_images)
        identity = Identity(person_id, name, (vector,), _timestamp(created_at), count)
        self._people[key] = identity
        return identity

    def update_template(self, person_id: str, embedding: Any, *, n_images: int) -> Identity:
        """Stage an explicit replacement averaged template, not an extra raw embedding."""
        key = _person_key(person_id)
        identity = self._people.get(key)
        if identity is None:
            raise UnknownPersonIDError(f"person_id {person_id!r} is not registered")
        vector = _vector(embedding, self.embedding_dim)
        count = _image_count(n_images)
        updated = replace(identity, embeddings=(vector,), n_images=count)
        self._people[key] = updated
        return updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "people": [
                {
                    "person_id": identity.person_id,
                    "name": identity.name,
                    "embeddings": [list(identity.embeddings[0])],
                    "created_at": identity.created_at,
                    "n_images": identity.n_images,
                }
                for identity in self.people
            ],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> FaceGallery:
        header_fields = {"schema_version", "model_id", "model_version", "embedding_dim", "people"}
        if not isinstance(payload, dict) or set(payload) != header_fields:
            raise GalleryFormatError("invalid gallery fields")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise GalleryFormatError(
                f"unsupported gallery schema {payload['schema_version']!r}; "
                "old multi-embedding records require explicit migration/re-enrollment"
            )
        try:
            gallery = cls(
                model_id=payload["model_id"],
                model_version=payload["model_version"],
                embedding_dim=payload["embedding_dim"],
            )
            people = payload["people"]
            if not isinstance(people, list):
                raise ValueError("people must be a list")
            for record in people:
                if not isinstance(record, dict) or set(record) != {
                    "person_id", "name", "embeddings", "created_at", "n_images"
                }:
                    raise ValueError("invalid identity record fields")
                embeddings = record["embeddings"]
                if not isinstance(embeddings, list) or len(embeddings) != 1:
                    raise ValueError("identity requires exactly one averaged template")
                gallery.register(
                    record["person_id"], record["name"], embeddings[0],
                    n_images=record["n_images"], created_at=record["created_at"],
                )
            gallery._committed_people = dict(gallery._people)
        except (ValueError, TypeError, DuplicatePersonIDError) as exc:
            raise GalleryFormatError(f"invalid gallery content: {exc}") from exc
        return gallery

    def save(self, path: str | Path) -> None:
        """Atomically write pending records; roll back staged changes on disk failure."""
        path = Path(path)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            self._people = dict(self._committed_people)
            raise GalleryStorageError(f"could not save gallery to {path}; staged changes rolled back") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        self._committed_people = dict(self._people)

    @classmethod
    def load(
        cls, path: str | Path, *, model_id: str, model_version: str, embedding_dim: int
    ) -> FaceGallery:
        """Load a compatible gallery or return an empty one if the file is absent."""
        expected = cls(model_id=model_id, model_version=model_version, embedding_dim=embedding_dim)
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return expected
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise GalleryFormatError(f"invalid JSON gallery at {path}") from exc
        except OSError as exc:
            raise GalleryStorageError(f"could not load gallery from {path}") from exc
        loaded = cls.from_dict(payload)
        if (loaded.model_id, loaded.model_version, loaded.embedding_dim) != (
            expected.model_id, expected.model_version, expected.embedding_dim
        ):
            raise IncompatibleGalleryError("gallery model metadata does not match the active model")
        return loaded
