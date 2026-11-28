"""Model-independent gallery storage for UI-01.2/UI-01.4.

The caller supplies an averaged, L2-normalized enrollment template after face
quality, liveness and recognition checks. Storage does not infer or normalize.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from collections.abc import Mapping, Set
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "0.2"


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
    """An immutable identity with one averaged template and its sample count."""

    person_id: str
    name: str
    embeddings: tuple[tuple[float, ...], ...]
    created_at: str
    n_images: int


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
    return parsed.astimezone(timezone.utc).isoformat()


def _image_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 3 <= value <= 10:
        raise ValueError("n_images must be an integer from 3 to 10 accepted images")
    return int(value)


def _vector(value: Any, dimension: int) -> tuple[float, ...]:
    # Do not coerce the container first: NumPy coercion can turn mixed bool/text
    # inputs into numbers and hide invalid values. No NumPy import is required.
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)) or getattr(value, "ndim", 1) != 1:
        raise ValueError("embedding must be a one-dimensional numeric vector")
    try:
        if len(value) != dimension:
            raise ValueError(f"embedding must contain exactly {dimension} numeric values")
        components = iter(value)
    except TypeError as exc:
        raise ValueError("embedding must be a one-dimensional numeric vector") from exc
    result = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ValueError("embedding values must be numbers, not bool or text")
        try:
            number = float(component)
        except (ValueError, OverflowError) as exc:
            raise ValueError("embedding values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite")
        if not -1.0 <= number <= 1.0:
            raise ValueError("embedding values must be in [-1, 1]")
        result.append(number)
    if len(result) != dimension:
        raise ValueError(f"embedding must contain exactly {dimension} numeric values")
    if abs(math.hypot(*result) - 1.0) > 1e-3:
        raise ValueError("embedding must be L2-normalized (norm 1 +/- 1e-3)")
    return tuple(result)


class FaceGallery:
    """Register identity embeddings without owning the recognition model.

    Embedding dimension and model metadata are provided by the caller and must
    match the active checkpoint (currently D=512). IDs use casefold comparisons
    while preserving original spelling; names are not uniqueness keys.
    Mutations are staged until save succeeds. Failed saves roll back to the
    last successful save/load. Use one serialized writer per logical gallery.
    """

    def __init__(self, *, model_id: str, model_version: str, embedding_dim: int):
        self.model_id = _required_text(model_id, "model_id")
        self.model_version = _required_text(model_version, "model_version")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        self.embedding_dim = embedding_dim
        self._people: dict[str, Identity] = {}
        self._saved_people: dict[str, Identity] = {}

    @property
    def people(self) -> tuple[Identity, ...]:
        """Return identities without exposing the mutable internal mapping."""
        return tuple(self._people[key] for key in sorted(self._people))

    def get(self, person_id: str) -> Identity | None:
        return self._people.get(_required_text(person_id, "person_id").casefold())

    def register(
        self,
        person_id: str,
        name: str,
        embedding: Any,
        *,
        n_images: int,
        created_at: str | None = None,
    ) -> Identity:
        """Create a new identity. Updating an existing ID is a separate action."""
        person_id = _required_text(person_id, "person_id")
        name = _required_text(name, "name")
        if len(name) > 64:
            raise ValueError("name must contain 1 to 64 characters")
        key = person_id.casefold()
        if key in self._people:
            raise DuplicatePersonIDError(f"person_id {person_id!r} already exists")
        vector = _vector(embedding, self.embedding_dim)
        identity = Identity(person_id, name, (vector,), _timestamp(created_at), _image_count(n_images))
        self._people[key] = identity
        return identity

    def add_embedding(self, person_id: str, embedding: Any) -> Identity:
        """Reject the v0.1 sample-append API instead of silently changing its meaning."""
        raise ValueError("schema 0.2 stores one template; use update_template(..., n_images=...)")

    def update_template(self, person_id: str, embedding: Any, *, n_images: int) -> Identity:
        """Stage a replacement averaged template from a new accepted enrollment."""
        key = _required_text(person_id, "person_id").casefold()
        identity = self._people.get(key)
        if identity is None:
            raise UnknownPersonIDError(f"person_id {person_id!r} is not registered")
        vector = _vector(embedding, self.embedding_dim)
        updated = replace(identity, embeddings=(vector,), n_images=_image_count(n_images))
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
                    "person_id": person.person_id,
                    "name": person.name,
                    "embeddings": [list(vector) for vector in person.embeddings],
                    "created_at": person.created_at,
                    "n_images": person.n_images,
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
            raise GalleryFormatError(
                "unsupported gallery schema version; expected 0.2. Keep legacy files intact; "
                "v0.1 requires verified manual migration or re-enrollment (see docs/ui-01-4-gallery-revisions.md)"
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
                    raise ValueError("identity must contain exactly one averaged template")
                _required_text(record["created_at"], "created_at")
                gallery.register(
                    record["person_id"], record["name"], embeddings[0],
                    created_at=record["created_at"], n_images=record["n_images"],
                )
        except (ValueError, TypeError, DuplicatePersonIDError) as exc:
            raise GalleryFormatError(f"invalid gallery content: {exc}") from exc
        gallery._saved_people = gallery._people.copy()
        return gallery

    def save(self, path: str | Path) -> None:
        """Atomically save, or discard all changes since the last save/load.

        After failure the caller must explicitly reapply registration/updates
        before retrying; a later save cannot accidentally persist failed work.
        """
        temporary: str | None = None
        try:
            path = Path(path)
            payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            self._saved_people = self._people.copy()
        except (OSError, ValueError, TypeError) as exc:
            self._people = self._saved_people.copy()
            raise GalleryStorageError(
                f"could not save gallery to {path}; pending changes rolled back"
            ) from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    # Best-effort cleanup must not hide the original save error.
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
