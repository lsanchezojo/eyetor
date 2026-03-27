"""SQLite-backed memory store for agents."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Memory:
    """A single memory entry."""

    id: int
    session_id: str
    type: str  # "fact" | "preference" | "conversation_summary"
    key: str
    value: str
    created_at: str
    updated_at: str


_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    UNIQUE (session_id, type, key)
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
"""


class MemoryStore:
    """Persistent storage for agent memories using SQLite."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def save(self, session_id: str, type: str, key: str, value: str) -> None:
        """Insert or update a memory entry."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO memories (session_id, type, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, type, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (session_id, type, key, value, now, now),
        )
        self._conn.commit()

    def get(self, session_id: str, type: str, key: str) -> str | None:
        """Retrieve a single memory value."""
        row = self._conn.execute(
            "SELECT value FROM memories WHERE session_id=? AND type=? AND key=?",
            (session_id, type, key),
        ).fetchone()
        return row["value"] if row else None

    def get_by_session(self, session_id: str) -> list[Memory]:
        """All memories for a session."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE session_id=? ORDER BY updated_at DESC",
            (session_id,),
        ).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[Memory]:
        """Full-text search across key and value fields."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE key LIKE ? OR value LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def delete(self, id: int) -> None:
        """Delete a memory entry by id."""
        self._conn.execute("DELETE FROM memories WHERE id=?", (id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
