"""Google Gemini LLM provider (OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from eyetor.models.messages import CompletionResult, FunctionCall, Message, TokenUsage, ToolCall
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.streaming.parsers import extract_delta_content, parse_sse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini via its OpenAI-compatible endpoint.

    Expects base_url like ``https://generativelanguage.googleapis.com/v1beta``.
    The OpenAI-compatible path ``/openai/chat/completions`` is appended automatically.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        payload = self._build_payload(messages, tools, temperature, stream=False)
        async with self._client(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/openai/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return _parse_completion_response(data)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, tools, temperature, stream=True)
        async with self._client(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/openai/chat/completions",
                json=payload,
                headers=self._build_headers(),
            ) as response:
                response.raise_for_status()
                async for chunk in parse_sse(response):
                    text = extract_delta_content(chunk)
                    if text:
                        yield text


def _parse_completion_response(data: dict[str, Any]) -> CompletionResult:
    """Parse a non-streaming /chat/completions response into a CompletionResult."""
    choice = data["choices"][0]
    msg = choice["message"]
    role = msg.get("role", "assistant")
    content = msg.get("content")
    raw_tool_calls = msg.get("tool_calls")

    tool_calls: list[ToolCall] | None = None
    if raw_tool_calls:
        tool_calls = [
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in raw_tool_calls
        ]

    message = Message(role=role, content=content, tool_calls=tool_calls)

    raw_usage = data.get("usage")
    usage = None
    if raw_usage:
        usage = TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

    return CompletionResult(
        message=message,
        usage=usage,
        model=data.get("model"),
        finish_reason=choice.get("finish_reason"),
    )
