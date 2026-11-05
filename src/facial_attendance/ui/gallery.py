"""Minimal, model-independent face gallery for UI-01.2.

The caller is responsible for face quality, liveness and embedding inference.
No matching threshold, normalization or embedding model is selected here.
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


SCHEMA_VERSION = "0.1"


class GalleryError(Exception):
    """Base exception for gallery operations."""


class DuplicatePersonIDError(GalleryError):
    """An enrollment attempted to reuse an existing person ID."""


class UnknownPersonIDError(GalleryError):
    """An embedding update targets an ID that is not registered."""


class GalleryFormatError(GalleryError):
    """Stored gallery data is invalid or unsupported."""


class IncompatibleGalleryError(GalleryError):
    """Stored embeddings do not match the expected model metadata."""


class GalleryStorageError(GalleryError):
    """A filesystem operation prevented saving or reading the gallery."""


@dataclass(frozen=True)
class Identity:
    """An immutable identity record, including one or more embeddings."""

    person_id: str
    name: str
    embeddings: tuple[tuple[float, ...], ...]
    created_at: str


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding spaces")
    return value


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    value = _required_text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("created_at must be an ISO 8601 timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError("created_at must contain a timezone offset")
    return value


def _vector(value: Any, dimension: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != dimension:
        raise ValueError(f"embedding must contain exactly {dimension} numeric values")
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError("embedding values must be numbers, not bool or text")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite")
        result.append(number)
    return tuple(result)


class FaceGallery:
    """Register identity embeddings without owning the recognition model.

    Embedding dimension and model metadata are provided by the caller and must
    be agreed with the recognition owner before integration. IDs are
    case-sensitive; names are not used as uniqueness keys.
    """

    def __init__(self, *, model_id: str, model_version: str, embedding_dim: int):
        self.model_id = _required_text(model_id, "model_id")
        self.model_version = _required_text(model_version, "model_version")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        self.embedding_dim = embedding_dim
        self._people: dict[str, Identity] = {}

    @property
    def people(self) -> tuple[Identity, ...]:
        """Return identities without exposing the mutable internal mapping."""
        return tuple(self._people[key] for key in sorted(self._people))

    def get(self, person_id: str) -> Identity | None:
        return self._people.get(person_id)

    def register(
        self,
        person_id: str,
        name: str,
        embedding: list[float] | tuple[float, ...],
        *,
        created_at: str | None = None,
    ) -> Identity:
        """Create a new identity. Updating an existing ID is a separate action."""
        person_id = _required_text(person_id, "person_id")
        name = _required_text(name, "name")
        if person_id in self._people:
            raise DuplicatePersonIDError(f"person_id {person_id!r} already exists")
        vector = _vector(embedding, self.embedding_dim)
        identity = Identity(person_id, name, (vector,), _timestamp(created_at))
        self._people[person_id] = identity
        return identity

    def add_embedding(self, person_id: str, embedding: list[float] | tuple[float, ...]) -> Identity:
        """Explicitly add a sample to an existing ID without creating a duplicate."""
        identity = self._people.get(person_id)
        if identity is None:
            raise UnknownPersonIDError(f"person_id {person_id!r} is not registered")
        vector = _vector(embedding, self.embedding_dim)
        updated = replace(identity, embeddings=identity.embeddings + (vector,))
        self._people[person_id] = updated
        return updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "people": [
                {
                    "person_id": person.person_id,
                    "name": person.name,
                    "embeddings": [list(vector) for vector in person.embeddings],
                    "created_at": person.created_at,
                }
                for person in self.people
            ],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> FaceGallery:
        """Validate disk content; never silently drop unknown fields or IDs."""
        fields = {"schema_version", "model_id", "model_version", "embedding_dim", "people"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise GalleryFormatError("invalid gallery fields")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise GalleryFormatError("unsupported gallery schema version")
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
                    "person_id", "name", "embeddings", "created_at"
                }:
                    raise ValueError("invalid identity record fields")
                embeddings = record["embeddings"]
                if not isinstance(embeddings, list) or not embeddings:
                    raise ValueError("identity must contain at least one embedding")
                gallery.register(
                    record["person_id"], record["name"], embeddings[0],
                    created_at=record["created_at"],
                )
                for vector in embeddings[1:]:
                    gallery.add_embedding(record["person_id"], vector)
        except (ValueError, TypeError, DuplicatePersonIDError) as exc:
            raise GalleryFormatError(f"invalid gallery content: {exc}") from exc
        return gallery

    def save(self, path: str | Path) -> None:
        """Write via a temporary sibling file, replacing the destination on success."""
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
            raise GalleryStorageError(f"could not save gallery to {path}") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @classmethod
    def load(
        cls, path: str | Path, *, model_id: str, model_version: str, embedding_dim: int
    ) -> FaceGallery:
        """Load a compatible gallery, or return an empty one if no file exists."""
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
