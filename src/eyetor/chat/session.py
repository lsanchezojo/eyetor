"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry
from eyetor.providers.base import BaseProvider

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
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()
        self._messages: list[Message] = []
        self._system_prompt_suffix = system_prompt_suffix

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
        messages: list[Message] = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.extend(self._messages)
        return messages
