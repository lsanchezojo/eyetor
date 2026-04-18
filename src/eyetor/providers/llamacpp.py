"""llama.cpp server LLM provider (OpenAI-compatible)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from eyetor.models.messages import CompletionResult, FunctionCall, Message, StreamingResponse, ToolCall
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, parse_sse

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseProvider):
    """Provider adapter for llama.cpp server's OpenAI-compatible API.

    Authentication is optional — only used if api_key is set.

    When ``thinking=True``, each request includes ``chat_template_kwargs``
    with ``enable_thinking: true``, which activates the reasoning channel on
    models that support it (e.g. Gemma-4, QwQ).  The ``<think>`` block is
    stripped from the visible response but logged at DEBUG level.
    """

    def __init__(
        self,
        *args: Any,
        thinking: bool = False,
        request_timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, tools, temperature, stream)
        if self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        return payload

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        payload = self._build_payload(messages, tools, temperature, stream=False)
        async with self._client(timeout=self.request_timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
            if response.status_code >= 400:
                body = response.text[:500]
                logger.error(
                    "llama.cpp %d error: %s",
                    response.status_code,
                    body,
                )
            response.raise_for_status()
            data = response.json()
            result = _parse_completion_response(data)
            if self.thinking:
                reasoning = _extract_reasoning(data)
                if reasoning:
                    result.reasoning_content = reasoning
                    logger.debug("llama.cpp reasoning:\n%s", reasoning.strip())
            return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        payload = self._build_payload(messages, tools, temperature, stream=True)
        sr = StreamingResponse(iter([]), None)  # placeholder, replaced below
        reasoning_parts: list[str] = []

        async def _stream_tokens() -> AsyncIterator[str]:
            async with self._client(timeout=self.request_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    response.raise_for_status()
                    async for chunk in parse_sse(response):
                        if self.thinking:
                            r_token = _extract_reasoning_delta(chunk)
                            if r_token:
                                reasoning_parts.append(r_token)
                        text = extract_delta_content(chunk)
                        if text:
                            yield text
            # After stream exhaustion, populate reasoning on the response object
            if reasoning_parts:
                sr.reasoning_content = "".join(reasoning_parts)

        sr._iterator = _stream_tokens()
        return sr


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_reasoning(data: dict[str, Any]) -> str | None:
    """Extract reasoning_content from a non-streaming response."""
    try:
        return data["choices"][0]["message"].get("reasoning_content") or None
    except (KeyError, IndexError):
        return None


def _extract_reasoning_delta(chunk: dict[str, Any]) -> str | None:
    """Extract a reasoning_content delta token from a streaming chunk."""
    try:
        return chunk["choices"][0]["delta"].get("reasoning_content") or None
    except (KeyError, IndexError):
        return None
