"""Local profile database, independent of verified attendance and its gallery."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from ..recognition.gallery_matching import GalleryMatcher, empty_gallery
from .gallery import FaceGallery, GalleryError


LOCAL_PROFILE_DB = Path(__file__).resolve().parents[3] / "artifacts" / "local-recognition" / "profiles.sqlite3"


class LocalProfileStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.revision = 0

    def load(self):
        if not self.path.exists():
            return empty_gallery()
        try:
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=0)) as db:
                record = db.execute("SELECT revision, gallery FROM profiles WHERE id = 1").fetchone()
            if record is None:
                raise ValueError("Missing local profile database record.")
            revision, payload = record
            gallery = FaceGallery.from_dict(json.loads(payload))
            GalleryMatcher(gallery)
        except (ValueError, GalleryError, sqlite3.Error) as error:
            raise ValueError("The local profile database is unavailable or uses an incompatible recognition model.") from error
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
                db.execute("CREATE TABLE IF NOT EXISTS profiles "
                           "(id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, gallery TEXT NOT NULL)")
                record = db.execute("SELECT revision FROM profiles WHERE id = 1").fetchone()
                if (record[0] if record else 0) != self.revision:
                    raise ValueError("Profiles changed in another window. Reopen this workspace before saving.")
                next_revision = self.revision + 1
                db.execute("INSERT INTO profiles VALUES (1, ?, ?) ON CONFLICT (id) "
                           "DO UPDATE SET revision = excluded.revision, gallery = excluded.gallery",
                           (next_revision, payload))
        except (OSError, sqlite3.Error) as error:
            raise ValueError("Could not save local profiles. Check storage access and try again.") from error
        self.revision = next_revision
