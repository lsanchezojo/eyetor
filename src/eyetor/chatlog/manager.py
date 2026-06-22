"""ChatLog manager — thin high-level API over :class:`ChatLogStore`."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.chatlog.store import ChatLogMessage, ChatLogStore

logger = logging.getLogger(__name__)


class ChatLogManager:
    """High-level API for the per-day, per-chat conversation archive.

    Used by the Telegram channel to *record* group messages out of context,
    and by the ``chat_history_*`` tools to *query* them on demand.
    """

    def __init__(self, store: ChatLogStore, *, retention_days: int = 0) -> None:
        self._store = store
        self._retention_days = retention_days

    @classmethod
    def from_path(
        cls, db_path: str | Path, *, retention_days: int = 0
    ) -> "ChatLogManager":
        """Create a ChatLogManager from a database path (expands ~)."""
        path = Path(db_path).expanduser()
        return cls(ChatLogStore(path), retention_days=retention_days)

    def record(self, session_id: str, sender: str, content: str) -> None:
        """Archive a single message for a chat."""
        try:
            self._store.record(
                session_id,
                sender,
                content,
                retention_days=self._retention_days,
            )
        except Exception as exc:  # never let archiving break the chat
            logger.warning("Failed to archive chat message: %s", exc)

    def search(
        self, session_id: str, query: str, *, day: str | None = None, limit: int = 10
    ) -> list[ChatLogMessage]:
        """Full-text search within a chat's archive."""
        return self._store.search(session_id, query, day=day, limit=limit)

    def read_day(
        self, session_id: str, day: str, *, limit: int = 200
    ) -> list[ChatLogMessage]:
        """Transcript of a chat for a given day."""
        return self._store.read_day(session_id, day, limit=limit)

    def list_days(self, session_id: str, *, limit: int = 30) -> list[dict]:
        """Days with logs for a chat, plus message counts."""
        return self._store.list_days(session_id, limit=limit)
