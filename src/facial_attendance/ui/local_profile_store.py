"""Local profile database, independent of verified attendance and its gallery."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from ..recognition.gallery_matching import GalleryMatcher, empty_gallery
from .gallery import FaceGallery, GalleryError

LOCAL_PROFILE_DB = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "local-recognition"
    / "profiles.sqlite3"
)
LOCAL_BOX_PROFILE_DB = LOCAL_PROFILE_DB.with_name("profiles-box.sqlite3")
BOX_PROFILE_VERSION = 1  # Untagged databases contain square-1.2 embeddings.
CROP_MISMATCH = "This profile database uses a different face crop. Select a separate profile database."


class LocalProfileStore:
    def __init__(self, path, *, crop_to_detector_box=False):
        self.path = Path(path).resolve()
        self._profile_version = BOX_PROFILE_VERSION if crop_to_detector_box else 0
        self.revision = 0

    def load(self):
        if not self.path.exists():
            return empty_gallery()
        try:
            with closing(
                sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=0)
            ) as db:
                profile_version = db.execute("PRAGMA user_version").fetchone()[0]
                record = db.execute(
                    "SELECT revision, gallery FROM profiles WHERE id = 1"
                ).fetchone()
            if record is None:
                raise ValueError("Missing local profile database record.")
            revision, payload = record
            gallery = FaceGallery.from_dict(json.loads(payload))
            GalleryMatcher(gallery)
        except (ValueError, GalleryError, sqlite3.Error) as error:
            raise ValueError(
                "The local profile database is unavailable or uses an incompatible recognition model."
            ) from error
        if profile_version != self._profile_version:
            raise ValueError(CROP_MISMATCH)
        self.revision = revision
        return gallery

    def save(self, gallery):
        """Publish a whole compatible gallery or leave the previous database intact."""
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = json.dumps(gallery.to_dict(), ensure_ascii=False, allow_nan=False)
            with closing(sqlite3.connect(self.path, timeout=0)) as db, db:
                db.execute("PRAGMA secure_delete = ON")
                db.execute("BEGIN IMMEDIATE")
                profile_version = db.execute("PRAGMA user_version").fetchone()[0]
                if profile_version not in (0, self._profile_version):
                    raise ValueError(CROP_MISMATCH)
                db.execute(
                    "CREATE TABLE IF NOT EXISTS profiles "
                    "(id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, gallery TEXT NOT NULL)"
                )
                record = db.execute(
                    "SELECT revision FROM profiles WHERE id = 1"
                ).fetchone()
                if record is not None and profile_version != self._profile_version:
                    raise ValueError(CROP_MISMATCH)
                if (record[0] if record else 0) != self.revision:
                    raise ValueError(
                        "Profiles changed in another window. Reopen this workspace before saving."
                    )
                next_revision = self.revision + 1
                db.execute(
                    "INSERT INTO profiles VALUES (1, ?, ?) ON CONFLICT (id) "
                    "DO UPDATE SET revision = excluded.revision, gallery = excluded.gallery",
                    (next_revision, payload),
                )
                if profile_version != self._profile_version:
                    db.execute(f"PRAGMA user_version = {self._profile_version}")
        except (OSError, sqlite3.Error) as error:
            raise ValueError(
                "Could not save local profiles. Check storage access and try again."
            ) from error
        self.revision = next_revision
