"""BaseAgent — single LLM call without tool loop."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from eyetor.models.agents import AgentConfig, AgentResult
from eyetor.models.messages import Message
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class BaseAgent:
    """Simple agent: builds messages, calls LLM once, returns result.

    Does NOT execute tools or loop. Use ToolAgent for agentic behavior.
    """

    def __init__(self, config: AgentConfig, provider: BaseProvider) -> None:
        self.config = config
        self.provider = provider

    async def run(self, user_input: str, history: list[Message] | None = None) -> AgentResult:
        """Run the agent with a single user input.

        Args:
            user_input: The user's message.
            history: Optional prior conversation messages to include.

        Returns:
            AgentResult with the assistant response.
        """
        messages = self._build_messages(user_input, history)
        result = await self.provider.complete(
            messages=messages,
            temperature=self.config.temperature,
        )
        response = result.message
        messages.append(response)
        return AgentResult(
            messages=messages,
            final_output=response.content or "",
            iterations=1,
        )

    async def stream(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from a single LLM call."""
        messages = self._build_messages(user_input, history)
        async for token in self.provider.stream(
            messages=messages,
            temperature=self.config.temperature,
        ):
            yield token

    def _build_messages(
        self, user_input: str, history: list[Message] | None
    ) -> list[Message]:
        messages: list[Message] = []
        if self.config.system_prompt:
            messages.append(Message(role="system", content=self.config.system_prompt))
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=user_input))
        return messages
