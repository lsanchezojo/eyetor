"""Per-chat conversation index — ChatGPT-style multiple conversations.

Each chat (stable channel identity, e.g. ``telegram-123`` or ``cli-luis``) can
hold many conversations, each with a short id, a title derived from its first
user message, and timestamps. The index lives in a single JSON file next to the
session JSONLs:

    <sessions.dir>/conversations.json
    { "<chat_id>": { "active": "<conv_id>",
                     "items": { "<conv_id>": {"title", "created_at", "updated_at"} } } }

Conversation history itself stays in per-conversation JSONL files named
``<safe_chat_id>.<conv_id>.jsonl`` (see ChatSession).

Concurrency: the Telegram daemon and an interactive ``eyetor start`` may run at
once, so every mutation locks the file (``fcntl.flock``), re-reads, mutates and
writes atomically via ``os.replace``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Strip a group-chat sender prefix ("[Luis]: hola") when deriving titles.
_SENDER_PREFIX_RE = re.compile(r"^\[[^\]]{1,64}\]:\s*")

TITLE_MAX_CHARS = 48


def sanitize_id(session_id: str) -> str:
    """Filesystem-safe form of a session/chat id (shared with ChatSession)."""
    return re.sub(r"[/:@\\]", "_", session_id)


def _now() -> str:
    # Microseconds: /chats sorts by updated_at, so same-second ties must break.
    return datetime.now().isoformat(timespec="microseconds")


def _new_conv_id() -> str:
    return uuid.uuid4().hex[:8]


def derive_title(user_text: str) -> str:
    """Title = first ~48 chars of the first user message, prefix-stripped."""
    text = _SENDER_PREFIX_RE.sub("", user_text or "")
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) > TITLE_MAX_CHARS:
        text = text[:TITLE_MAX_CHARS].rstrip() + "…"
    return text


class ConversationIndex:
    """Index of conversations per chat, backed by one JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    # -- read helpers ------------------------------------------------------

    def _read(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Conversation index unreadable (%s); starting empty", exc)
            return {}

    def _mutate(self, fn) -> object:
        """Lock, re-read, apply ``fn(data) -> result``, write atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(".lock")
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self._read()
            result = fn(data)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        return result

    # -- public API --------------------------------------------------------

    def active(self, chat_id: str, legacy_file: Path | None = None) -> str:
        """Return the chat's active conversation id, creating one if needed.

        On first contact for a chat, if ``legacy_file`` (the old single
        ``<chat_id>.jsonl``) exists, it is renamed into the new per-conversation
        scheme so prior history becomes conversation #1.
        """
        data = self._read()
        entry = data.get(chat_id)
        if entry and entry.get("active") in entry.get("items", {}):
            return entry["active"]

        def _create(data: dict) -> str:
            entry = data.get(chat_id)
            if entry and entry.get("active") in entry.get("items", {}):
                return entry["active"]  # raced with another process
            conv_id = _new_conv_id()
            title = ""
            if legacy_file is not None and legacy_file.exists():
                target = legacy_file.with_name(f"{legacy_file.stem}.{conv_id}.jsonl")
                try:
                    legacy_file.rename(target)
                    logger.info("Migrated legacy session %s -> %s", legacy_file.name, target.name)
                except OSError as exc:
                    logger.warning("Could not migrate %s (%s); starting fresh", legacy_file, exc)
            data[chat_id] = {
                "active": conv_id,
                "items": {conv_id: {"title": title, "created_at": _now(), "updated_at": _now()}},
            }
            return conv_id

        return self._mutate(_create)

    def new(self, chat_id: str) -> str:
        """Start a new conversation; reuse the active one if still empty."""

        def _new(data: dict) -> str:
            entry = data.setdefault(chat_id, {"active": "", "items": {}})
            items = entry.setdefault("items", {})
            active = entry.get("active")
            # Empty (never titled = never received a user message): reuse.
            if active in items and not items[active].get("title"):
                return active
            conv_id = _new_conv_id()
            items[conv_id] = {"title": "", "created_at": _now(), "updated_at": _now()}
            entry["active"] = conv_id
            return conv_id

        return self._mutate(_new)

    def switch(self, chat_id: str, conv_id: str) -> bool:
        """Make ``conv_id`` the chat's active conversation. False if unknown."""

        def _switch(data: dict) -> bool:
            entry = data.get(chat_id)
            if not entry or conv_id not in entry.get("items", {}):
                return False
            entry["active"] = conv_id
            return True

        return bool(self._mutate(_switch))

    def list(self, chat_id: str) -> list[dict]:
        """Conversations of a chat, newest-updated first.

        Each item: {"id", "title", "created_at", "updated_at", "active": bool}.
        """
        data = self._read()
        entry = data.get(chat_id)
        if not entry:
            return []
        active = entry.get("active")
        items = [
            {"id": cid, "active": cid == active, **meta}
            for cid, meta in entry.get("items", {}).items()
        ]
        items.sort(key=lambda it: it.get("updated_at", ""), reverse=True)
        return items

    def touch(self, chat_id: str, conv_id: str, user_text: str) -> None:
        """Bump updated_at; set the title from the first user message."""

        def _touch(data: dict) -> None:
            entry = data.get(chat_id)
            if not entry:
                return
            meta = entry.get("items", {}).get(conv_id)
            if meta is None:
                return
            meta["updated_at"] = _now()
            if not meta.get("title"):
                meta["title"] = derive_title(user_text)

        self._mutate(_touch)

    def active_id(self, chat_id: str) -> str | None:
        """Read-only: the chat's active conversation id, or None."""
        entry = self._read().get(chat_id)
        if entry and entry.get("active") in entry.get("items", {}):
            return entry["active"]
        return None
