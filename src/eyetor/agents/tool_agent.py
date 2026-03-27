"""ToolAgent — agentic loop with tool calling."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from eyetor.models.agents import AgentConfig, AgentResult
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ToolAgent:
    """Agent that loops until the LLM stops requesting tool calls.

    Loop (up to max_iterations):
      1. Call LLM with current messages + tool definitions
      2. If response has no tool_calls → return final answer
      3. Execute each tool_call via ToolRegistry
      4. Append tool results as Message(role="tool")
      5. Repeat
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: BaseProvider,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()

    async def run(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """Run the agentic loop and return the final result."""
        messages = self._build_messages(user_input, history)
        tools = self.tool_registry.list_openai()
        tool_defs = None
        if tools:
            # Pass ToolDefinition objects so provider can serialize them
            tool_defs = list(self.tool_registry._tools.values())

        all_tool_calls: list[ToolCall] = []
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1
            response = await self.provider.complete(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                temperature=self.config.temperature,
            )
            messages.append(response)

            if not response.tool_calls:
                # No tool calls → final answer
                return AgentResult(
                    messages=messages,
                    final_output=response.content or "",
                    iterations=iterations,
                    tool_calls_made=all_tool_calls,
                )

            # Execute each tool call and collect results
            for tc in response.tool_calls:
                all_tool_calls.append(tc)
                result_content = await self.tool_registry.execute(
                    tc.function.name, tc.function.arguments
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=result_content,
                    )
                )
                logger.debug(
                    "Tool '%s' executed → %s chars",
                    tc.function.name,
                    len(result_content),
                )

        # Reached max_iterations — return last assistant message
        last_content = next(
            (m.content for m in reversed(messages) if m.role == "assistant"),
            "Max iterations reached without a final answer.",
        )
        logger.warning("ToolAgent '%s' reached max_iterations=%d", self.config.name, self.config.max_iterations)
        return AgentResult(
            messages=messages,
            final_output=last_content or "",
            iterations=iterations,
            tool_calls_made=all_tool_calls,
        )

    async def stream(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the final response (tool calls are executed silently)."""
        # Run to get final messages, then stream the last turn
        result = await self.run(user_input, history)
        # Re-stream the final output token by token (character-level fallback)
        # For real streaming, callers should use chat/session.py which
        # handles the loop with streaming awareness.
        yield result.final_output

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
