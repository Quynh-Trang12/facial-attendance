"""Keep attendance events in a SQLite database separate from face profiles."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class AttendanceRecord:
    attendance_id: str
    person_id: str
    person_name: str
    session_id: str
    recorded_at: str
    status: str = "present"


class AttendanceStore:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def _insert(self, session_id, person_id, person_name):
        """Called only after the service has authorized this frozen event."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=0)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if tables - {"attendance"}:
                raise sqlite3.DatabaseError("Select a dedicated attendance database.")
            db.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status = 'present'),
                    UNIQUE (session_id, person_id)
                )
            """)
            record = AttendanceRecord(
                uuid4().hex,
                person_id,
                person_name,
                session_id,
                datetime.now(UTC).isoformat(),
            )
            inserted = db.execute(
                """
                INSERT INTO attendance (attendance_id, person_id, person_name, session_id, recorded_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id, person_id) DO NOTHING
            """,
                (
                    record.attendance_id,
                    record.person_id,
                    record.person_name,
                    record.session_id,
                    record.recorded_at,
                    record.status,
                ),
            ).rowcount
        return record if inserted else None
