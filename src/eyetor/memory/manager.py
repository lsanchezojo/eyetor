"""Memory manager — injects context and extracts facts from conversations."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.memory.store import MemoryStore
from eyetor.models.messages import Message

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level memory API for agents.

    Responsibilities:
    - Build a context string to inject into system prompts.
    - Extract and persist key facts from completed conversations.
    - Provide explicit remember/forget methods.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @classmethod
    def from_path(cls, db_path: str | Path) -> "MemoryManager":
        """Create a MemoryManager from a database path (expands ~)."""
        path = Path(db_path).expanduser()
        return cls(MemoryStore(path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(self, session_id: str) -> str:
        """Build a memory context string to inject at start of system prompt.

        Returns an empty string if there are no stored memories.
        """
        memories = self._store.get_by_session(session_id)
        if not memories:
            return ""
        lines = ["## Memory from previous sessions"]
        for m in memories:
            lines.append(f"- [{m.type}] {m.key}: {m.value}")
        return "\n".join(lines)

    def remember(self, session_id: str, key: str, value: str, type: str = "fact") -> None:
        """Explicitly save a fact to memory."""
        self._store.save(session_id, type, key, value)
        logger.debug("Remembered [%s] %s = %s (session=%s)", type, key, value, session_id)

    def forget(self, session_id: str, key: str, type: str = "fact") -> None:
        """Delete a memory entry by key."""
        memories = self._store.get_by_session(session_id)
        for m in memories:
            if m.key == key and m.type == type:
                self._store.delete(m.id)
                logger.debug("Forgot [%s] %s (session=%s)", type, key, session_id)
                return

    def save_summary(self, session_id: str, summary: str) -> None:
        """Save a conversation summary."""
        self._store.save(session_id, "conversation_summary", "last_summary", summary)

    def list_memories(self, session_id: str) -> list[dict]:
        """Return all memories for a session as dicts."""
        return [
            {"id": m.id, "type": m.type, "key": m.key, "value": m.value, "updated_at": m.updated_at}
            for m in self._store.get_by_session(session_id)
        ]
