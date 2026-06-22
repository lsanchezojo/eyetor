"""SQLite-backed conversation archive for group chats.

Unlike the active :class:`~eyetor.chat.session.ChatSession` history (which is
loaded into the model context every turn and therefore kept small), this store
keeps the *full* transcript of a chat partitioned by day. It never enters the
context automatically — the model reads it on demand via the ``chat_history_*``
tools. This lets the bot answer "what did we talk about yesterday?" without
saturating a small context window.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eyetor._sqlite_util import apply_concurrency_pragmas
from eyetor.knowledge.store import sanitize_fts5_query

logger = logging.getLogger(__name__)


@dataclass
class ChatLogMessage:
    """A single archived chat message."""

    id: int
    session_id: str
    day: str  # local date, YYYY-MM-DD
    ts: str  # ISO timestamp
    sender: str  # display name ("Eyetor" for the bot)
    content: str


_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    day         TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    sender      TEXT    NOT NULL,
    content     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chatlog_session_day ON messages(session_id, day);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    sender,
    content='messages',
    content_rowid='id',
    tokenize = "unicode61 remove_diacritics 2 tokenchars '_-.'"
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, sender)
    VALUES (new.id, new.content, new.sender);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, sender)
    VALUES ('delete', old.id, old.content, old.sender);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, sender)
    VALUES ('delete', old.id, old.content, old.sender);
    INSERT INTO messages_fts(rowid, content, sender)
    VALUES (new.id, new.content, new.sender);
END;
"""


class ChatLogStore:
    """Persistent per-day, per-chat message archive using SQLite + FTS5."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_concurrency_pragmas(self._conn)
        self._conn.executescript(_DDL)
        self._conn.commit()

    def record(
        self,
        session_id: str,
        sender: str,
        content: str,
        *,
        when: datetime | None = None,
        retention_days: int = 0,
    ) -> None:
        """Archive a single message. ``day`` is derived from the local time.

        If *retention_days* > 0, messages for this session older than that many
        days are purged after the insert.
        """
        content = (content or "").strip()
        if not content:
            return
        moment = when or datetime.now()
        day = moment.date().isoformat()
        self._conn.execute(
            "INSERT INTO messages (session_id, day, ts, sender, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, day, moment.isoformat(), sender, content),
        )
        if retention_days and retention_days > 0:
            cutoff = (
                moment.date().toordinal() - int(retention_days)
            )
            cutoff_day = datetime.fromordinal(cutoff).date().isoformat()
            self._conn.execute(
                "DELETE FROM messages WHERE session_id=? AND day < ?",
                (session_id, cutoff_day),
            )
        self._conn.commit()

    def search(
        self,
        session_id: str,
        query: str,
        *,
        day: str | None = None,
        limit: int = 10,
    ) -> list[ChatLogMessage]:
        """Full-text search within a single chat's archive (newest first)."""
        limit = max(1, min(int(limit or 10), 50))
        fts_query = sanitize_fts5_query(query)
        if fts_query:
            try:
                sql = (
                    "SELECT m.* FROM messages_fts "
                    "JOIN messages m ON m.id = messages_fts.rowid "
                    "WHERE messages_fts MATCH ? AND m.session_id = ?"
                )
                params: list[object] = [fts_query, session_id]
                if day:
                    sql += " AND m.day = ?"
                    params.append(day)
                sql += " ORDER BY m.id DESC LIMIT ?"
                params.append(limit)
                rows = self._conn.execute(sql, params).fetchall()
                return [ChatLogMessage(**dict(r)) for r in rows]
            except sqlite3.DatabaseError as exc:
                logger.debug("chatlog FTS search failed (%s); falling back to LIKE", exc)
        # Fallback: literal substring match
        sql = "SELECT * FROM messages WHERE session_id=? AND content LIKE ?"
        params = [session_id, f"%{query}%"]
        if day:
            sql += " AND day = ?"
            params.append(day)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [ChatLogMessage(**dict(r)) for r in rows]

    def read_day(
        self, session_id: str, day: str, *, limit: int = 200
    ) -> list[ChatLogMessage]:
        """Return a chat's transcript for a given day, in chronological order."""
        limit = max(1, min(int(limit or 200), 1000))
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id=? AND day=? "
            "ORDER BY id ASC LIMIT ?",
            (session_id, day, limit),
        ).fetchall()
        return [ChatLogMessage(**dict(r)) for r in rows]

    def list_days(self, session_id: str, *, limit: int = 30) -> list[dict]:
        """List days that have logs for a chat (newest first) with counts."""
        limit = max(1, min(int(limit or 30), 365))
        rows = self._conn.execute(
            "SELECT day, COUNT(*) AS count FROM messages WHERE session_id=? "
            "GROUP BY day ORDER BY day DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"day": r["day"], "count": r["count"]} for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
