"""Tool definitions and registry."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ToolDefinition(BaseModel):
    """Definition of a tool that agents can call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    handler: Callable[..., Awaitable[str]] | None = None

    def to_openai_format(self) -> dict[str, Any]:
        """Serialize to OpenAI tools array element."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Central registry for tool definitions. Agents and skills register tools here."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool definition by name."""
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_openai(self) -> list[dict[str, Any]]:
        """All tools serialized in OpenAI format."""
        return [t.to_openai_format() for t in self._tools.values()]

    async def execute(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON arguments string."""
        tool = self.get(name)
        if tool.handler is None:
            return json.dumps({"error": f"Tool '{name}' has no handler"})
        try:
            args = json.loads(arguments) if arguments else {}
            result = await tool.handler(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.error("Tool execution error for '%s': %s", name, e)
            return json.dumps({"error": str(e)})
