"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, TYPE_CHECKING

from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider

if TYPE_CHECKING:
    from eyetor.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class ChatSession:
    """A single ongoing conversation with an agent.

    Manages message history and runs the tool-calling loop transparently.
    Both streaming and non-streaming interfaces are provided.
    """

    def __init__(
        self,
        session_id: str,
        config: AgentConfig,
        provider: BaseProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt_suffix: str = "",
        memory_manager: "MemoryManager | None" = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.provider = provider
        self._messages: list[Message] = []
        self._system_prompt_suffix = system_prompt_suffix
        self._memory = memory_manager

        if memory_manager is not None:
            # Copy the shared registry so memory tools are session-specific
            shared = tool_registry or ToolRegistry()
            self.tool_registry = ToolRegistry()
            for tool in shared._tools.values():
                self.tool_registry.register(tool)
            self._register_memory_tools(memory_manager)
        else:
            self.tool_registry = tool_registry or ToolRegistry()

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def get_history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def send(self, user_input: str) -> AsyncIterator[str]:
        """Send a user message; yield streaming tokens from the assistant.

        Tool calls are executed silently. The final response is streamed.
        """
        self._messages.append(Message(role="user", content=user_input))
        tool_defs = list(self.tool_registry._tools.values()) if self.tool_registry._tools else None

        full_messages = self._get_full_messages()
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1
            # Non-streaming call to detect tool calls
            response = await self.provider.complete(
                messages=full_messages,
                tools=tool_defs,
                temperature=self.config.temperature,
            )
            self._messages.append(response)
            full_messages.append(response)

            if not response.tool_calls:
                # Final answer — yield it token by token (character-level)
                content = response.content or ""
                yield content
                return

            # Execute tool calls
            for tc in response.tool_calls:
                result = await self.tool_registry.execute(tc.function.name, tc.function.arguments)
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=result)
                self._messages.append(tool_msg)
                full_messages.append(tool_msg)
                logger.debug("Tool '%s' → %d chars", tc.function.name, len(result))

        # Max iterations reached
        yield "I reached the maximum number of reasoning steps. Please try a simpler question."

    async def send_sync(self, user_input: str) -> str:
        """Send a user message and return the complete response (non-streaming)."""
        result = ""
        async for chunk in self.send(user_input):
            result += chunk
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_full_messages(self) -> list[Message]:
        """Build the full messages list including system prompt."""
        system_content = self.config.system_prompt
        if self._system_prompt_suffix:
            system_content = f"{system_content}\n\n{self._system_prompt_suffix}"
        if self._memory:
            memory_context = self._memory.build_context(self.session_id)
            if memory_context:
                system_content = f"{system_content}\n\n{memory_context}"
        messages: list[Message] = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.extend(self._messages)
        return messages

    def _register_memory_tools(self, memory_manager: "MemoryManager") -> None:
        """Register remember/forget tools backed by persistent memory."""
        session_id = self.session_id

        async def remember_handler(key: str, value: str, type: str = "fact") -> str:
            memory_manager.remember(session_id, key, value, type)
            return json.dumps({"status": "ok", "key": key, "type": type})

        async def forget_handler(key: str, type: str = "fact") -> str:
            memory_manager.forget(session_id, key, type)
            return json.dumps({"status": "ok", "key": key})

        self.tool_registry.register(ToolDefinition(
            name="remember",
            description=(
                "Save a fact, preference or note to persistent memory so it is available "
                "in future conversations. Use whenever the user shares something important "
                "about themselves, their preferences, or context you should not forget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier (e.g. 'user_name', 'preferred_language', 'project')"},
                    "value": {"type": "string", "description": "The value to remember"},
                    "type": {"type": "string", "enum": ["fact", "preference", "note"], "default": "fact"},
                },
                "required": ["key", "value"],
            },
            handler=remember_handler,
        ))

        self.tool_registry.register(ToolDefinition(
            name="forget",
            description="Delete a previously saved memory entry by key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key of the memory to delete"},
                    "type": {"type": "string", "enum": ["fact", "preference", "note"], "default": "fact"},
                },
                "required": ["key"],
            },
            handler=forget_handler,
        ))
