"""ToolAgent — agentic loop with tool calling."""

from __future__ import annotations

import asyncio
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
        recent_calls: list[str] = []  # track "name:args" for loop detection
        max_repeat = 3  # max consecutive identical tool calls before forcing answer

        while iterations < self.config.max_iterations:
            iterations += 1
            result = await self.provider.complete(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                temperature=self.config.temperature,
            )
            response = result.message
            messages.append(response)

            if not response.tool_calls:
                # No tool calls → final answer
                return AgentResult(
                    messages=messages,
                    final_output=response.content or "",
                    iterations=iterations,
                    tool_calls_made=all_tool_calls,
                )

            # Log tool calls at INFO level
            call_names = [tc.function.name for tc in response.tool_calls]
            logger.info(
                "ToolAgent '%s' iter %d/%d — tool calls: %s",
                self.config.name, iterations, self.config.max_iterations,
                ", ".join(call_names),
            )

            # Loop detection
            call_signatures = [
                f"{tc.function.name}:{tc.function.arguments}"
                for tc in response.tool_calls
            ]
            current_sig = "|".join(sorted(call_signatures))
            recent_calls.append(current_sig)
            if len(recent_calls) > max_repeat:
                recent_calls = recent_calls[-max_repeat:]

            if len(recent_calls) == max_repeat and len(set(recent_calls)) == 1:
                logger.warning(
                    "ToolAgent '%s' — loop detected: same tool call(s) repeated %d times: %s. "
                    "Forcing final answer.",
                    self.config.name, max_repeat, call_names,
                )
                messages.append(Message(
                    role="user",
                    content=(
                        "IMPORTANT: You seem to be stuck in a loop calling the same tools repeatedly. "
                        "Please provide your final answer now using the information you already have. "
                        "Do NOT call any more tools."
                    ),
                ))
                result = await self.provider.complete(
                    messages=messages,
                    tools=None,
                    temperature=self.config.temperature,
                )
                forced = result.message
                messages.append(forced)
                return AgentResult(
                    messages=messages,
                    final_output=forced.content or "",
                    iterations=iterations,
                    tool_calls_made=all_tool_calls,
                )

            # Execute tool calls in parallel
            all_tool_calls.extend(response.tool_calls)

            async def _exec(tc: ToolCall) -> tuple[ToolCall, str]:
                content = await self.tool_registry.execute(
                    tc.function.name, tc.function.arguments
                )
                return tc, content

            results = await asyncio.gather(
                *[_exec(tc) for tc in response.tool_calls],
                return_exceptions=True,
            )
            for entry in results:
                if isinstance(entry, BaseException):
                    logger.error("ToolAgent '%s' — tool error: %s", self.config.name, entry)
                    continue
                tc, result_content = entry
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=result_content,
                    )
                )
                logger.info(
                    "ToolAgent '%s' — tool '%s' → %d chars",
                    self.config.name, tc.function.name, len(result_content),
                )

        # Reached max_iterations — return last assistant message
        last_content = next(
            (m.content for m in reversed(messages) if m.role == "assistant" and m.content),
            "Max iterations reached without a final answer.",
        )
        logger.warning(
            "ToolAgent '%s' reached max_iterations=%d. Tool calls: %s",
            self.config.name, self.config.max_iterations,
            ", ".join(tc.function.name for tc in all_tool_calls),
        )
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
