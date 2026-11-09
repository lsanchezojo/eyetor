"""Chat history tools over the per-day chat archive.

Scoped to the active chat via current_session_id, so one chat can never read
another's logs.
"""

from __future__ import annotations

import json

from eyetor.models.tools import ToolDefinition, ToolRegistry


def register_chat_history_tools(tool_registry: ToolRegistry, chatlog) -> None:
    """Register chat_history_search / read_day / list_days for a ``ChatLogManager``."""
    from eyetor.providers.tracking import current_session_id

    def _active_session_id() -> str | None:
        try:
            return current_session_id.get()
        except Exception:
            return None

    def _msg_to_dict(m) -> dict:
        return {"day": m.day, "ts": m.ts, "sender": m.sender, "content": m.content}

    async def chat_history_search_handler(
        query: str, day: str | None = None, limit: int = 10
    ) -> str:
        sid = _active_session_id()
        if not sid:
            return json.dumps(
                {"error": "No active chat session.", "results": []},
                ensure_ascii=False,
            )
        hits = chatlog.search(sid, query, day=day, limit=int(limit or 10))
        return json.dumps(
            {"query": query, "count": len(hits),
             "results": [_msg_to_dict(m) for m in hits]},
            ensure_ascii=False,
        )

    async def chat_history_read_day_handler(day: str, limit: int = 100) -> str:
        sid = _active_session_id()
        if not sid:
            return json.dumps(
                {"error": "No active chat session.", "messages": []},
                ensure_ascii=False,
            )
        msgs = chatlog.read_day(sid, day, limit=int(limit or 100))
        return json.dumps(
            {"day": day, "count": len(msgs),
             "messages": [_msg_to_dict(m) for m in msgs]},
            ensure_ascii=False,
        )

    async def chat_history_list_days_handler(limit: int = 30) -> str:
        sid = _active_session_id()
        if not sid:
            return json.dumps(
                {"error": "No active chat session.", "days": []},
                ensure_ascii=False,
            )
        days = chatlog.list_days(sid, limit=int(limit or 30))
        return json.dumps({"days": days}, ensure_ascii=False)

    tool_registry.register(
        ToolDefinition(
            name="chat_history_search",
            description=(
                "Search the full archived history of THIS chat (group "
                "conversation), including messages not addressed to you. "
                "Use it to recall what was discussed earlier or on another "
                "day, or when a user asks about past conversation. Each "
                "result has the sender's name, day and timestamp."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search terms. Supports FTS5 syntax: AND, OR, "
                            'NEAR, "quoted phrase".'
                        ),
                    },
                    "day": {
                        "type": "string",
                        "description": "Optional day filter, format YYYY-MM-DD.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (1-50, default 10).",
                    },
                },
                "required": ["query"],
            },
            handler=chat_history_search_handler,
            # Always-on (not gated): recalling past conversation is a
            # core group capability and is too varied to keyword-match
            # reliably, so these tools must never be silently hidden.
            group=None,
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="chat_history_read_day",
            description=(
                "Read the full transcript of THIS chat for a given day, in "
                "chronological order. Use after chat_history_list_days or to "
                "review everything said on a specific date."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "Day to read, format YYYY-MM-DD.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages (default 100).",
                    },
                },
                "required": ["day"],
            },
            handler=chat_history_read_day_handler,
            group=None,  # always-on; see chat_history_search above
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="chat_history_list_days",
            description=(
                "List the days that have archived history for THIS chat, "
                "newest first, with message counts. Use to discover what "
                "past days are available before reading or searching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max days to list (default 30).",
                    },
                },
                "required": [],
            },
            handler=chat_history_list_days_handler,
            group=None,  # always-on; see chat_history_search above
        )
    )
