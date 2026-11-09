"""remember / forget tools backed by persistent memory (per session)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from eyetor.models.tools import ToolDefinition, ToolRegistry

if TYPE_CHECKING:
    from eyetor.memory.manager import MemoryManager


def register_memory_tools(
    tool_registry: ToolRegistry, memory_manager: "MemoryManager", session_id: str
) -> None:
    """Register remember/forget tools backed by persistent memory."""

    async def remember_handler(key: str, value: str, type: str = "fact") -> str:
        memory_manager.remember(session_id, key, value, type)
        return json.dumps({"status": "ok", "key": key, "type": type})

    async def forget_handler(key: str, type: str = "fact") -> str:
        memory_manager.forget(session_id, key, type)
        return json.dumps({"status": "ok", "key": key})

    tool_registry.register(
        ToolDefinition(
            name="remember",
            description=(
                "Save a fact, preference or note to persistent memory so it is available "
                "in future conversations. Use whenever the user shares something important "
                "about themselves, their preferences, or context you should not forget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short identifier (e.g. 'user_name', 'preferred_language', 'project')",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to remember",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["fact", "preference", "note"],
                        "default": "fact",
                    },
                },
                "required": ["key", "value"],
            },
            handler=remember_handler,
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="forget",
            description="Delete a previously saved memory entry by key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the memory to delete",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["fact", "preference", "note"],
                        "default": "fact",
                    },
                },
                "required": ["key"],
            },
            handler=forget_handler,
        )
    )
