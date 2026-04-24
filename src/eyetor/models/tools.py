"""Tool definitions and registry."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from eyetor.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class ToolDefinition(BaseModel):
    """Definition of a tool that agents can call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    handler: Callable[..., Awaitable[str]] | None = None
    max_output_chars: int | None = None  # per-tool override; None → use registry default

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

    def __init__(
        self,
        plugin_registry: "PluginRegistry | None" = None,
        default_max_output_chars: int = 8000,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._plugin_registry = plugin_registry
        self._default_max_output_chars = default_max_output_chars

    def _cap_result(self, name: str, result: str) -> str:
        """Apply size cap: per-tool override first, then registry default."""
        tool = self._tools.get(name)
        cap = tool.max_output_chars if tool and tool.max_output_chars else self._default_max_output_chars
        if cap <= 0 or len(result) <= cap:
            return result
        original_len = len(result)
        truncated = result[:cap]
        logger.warning(
            "Tool '%s' output truncated: %d → %d chars", name, original_len, cap
        )
        return (
            truncated
            + f"\n[truncated from {original_len} chars — refine the query or use a more specific tool]"
        )

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

        # Pre-hook
        if self._plugin_registry:
            decision = await self._plugin_registry.run_pre_hooks(name, arguments)
            if decision.deny:
                return json.dumps({"error": f"Blocked by plugin: {decision.deny_reason}"})
            if decision.provided_result is not None:
                return self._cap_result(name, decision.provided_result)
            if decision.modified_input:
                arguments = decision.modified_input

        t0 = time.monotonic()
        try:
            args = json.loads(arguments) if arguments else {}
            result = await tool.handler(**args)
            result = result if isinstance(result, str) else json.dumps(result)
            result = self._cap_result(name, result)
            duration_ms = int((time.monotonic() - t0) * 1000)
            # Post-hook (fire & forget)
            if self._plugin_registry:
                asyncio.create_task(
                    self._plugin_registry.run_post_hooks(name, arguments, result, duration_ms)
                )
            return result
        except Exception as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.error("Tool execution error for '%s': %s", name, e)
            # Failure hook (fire & forget)
            if self._plugin_registry:
                asyncio.create_task(
                    self._plugin_registry.run_failure_hooks(name, arguments, str(e), duration_ms)
                )
            return json.dumps({"error": str(e)})
