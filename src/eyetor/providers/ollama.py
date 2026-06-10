"""Ollama LLM provider (OpenAI-compatible endpoint)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from eyetor.models.messages import (
    CompletionResult,
    Message,
    StreamingResponse,
    TokenUsage,
)
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, extract_usage, parse_sse

logger = logging.getLogger(__name__)

_NO_THINKING_PHASES = frozenset(
    {
        "compaction",
        "degeneration_recovery",
        "loop_break",
        "chain_synthesize",
    }
)


class OllamaProvider(BaseProvider):
    """Provider adapter for Ollama's OpenAI-compatible API.

    Ollama does not require authentication for local instances.
    """

    def __init__(
        self,
        *args: Any,
        thinking: bool = False,
        request_timeout: float = 300.0,
        max_tokens: int | None = None,
        max_tokens_by_phase: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.max_tokens_by_phase = max_tokens_by_phase or {}

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, tools, temperature, stream)
        thinking_enabled = self._thinking_enabled_for_current_phase()
        if thinking_enabled:
            # Ollama accepts `think: true` on its OpenAI-compatible endpoint;
            # send it explicitly so model/template defaults cannot silently
            # leave the reasoning channel disabled.
            payload["think"] = True
        else:
            # Ollama's OpenAI-compatible endpoint ignores `think: false` for
            # Gemma-4; this is the knob that reliably disables reasoning.
            payload["reasoning"] = {"effort": "none"}
        max_tokens = self._max_tokens_for_current_phase()
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        logger.debug(
            "Ollama payload thinking=%s phase=%r has_reasoning=%s has_think=%s max_tokens=%s",
            thinking_enabled,
            self._current_phase(),
            "reasoning" in payload,
            "think" in payload,
            payload.get("max_tokens"),
        )
        return payload

    def _thinking_enabled_for_current_phase(self) -> bool:
        if not self.thinking:
            return False
        phase = self._current_phase()
        return phase not in _NO_THINKING_PHASES

    def _max_tokens_for_current_phase(self) -> int | None:
        if not self.max_tokens_by_phase:
            return self.max_tokens
        phase = self._current_phase()
        if phase and phase in self.max_tokens_by_phase:
            return int(self.max_tokens_by_phase[phase])
        return self.max_tokens

    def _current_phase(self) -> str:
        try:
            from eyetor.tracking.context import current_phase

            return current_phase.get()
        except Exception:  # pragma: no cover - defensive fallback
            return ""

    def _build_headers(self) -> dict[str, str]:
        # No auth for local Ollama; api_key would only apply if proxied
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
            response.raise_for_status()
            data = response.json()
            result = _parse_completion_response(data)
            result.reasoning_content = _extract_reasoning(data)
            return result

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        payload = self._build_payload(messages, tools, temperature, stream=True)
        sr = StreamingResponse(iter([]), None)  # placeholder, replaced below

        async def _stream_tokens() -> AsyncIterator[str]:
            usage: TokenUsage | None = None
            async with self._client(timeout=self.request_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    response.raise_for_status()
                    async for chunk in parse_sse(response):
                        text = extract_delta_content(chunk)
                        if text:
                            yield text
                        extracted = extract_usage(chunk)
                        if extracted:
                            usage = extracted
            # After stream exhaustion, attach real usage (if the server
            # emitted a usage block with stream_options.include_usage).
            if usage is not None:
                sr._usage = usage

        sr._iterator = _stream_tokens()
        return sr


def _extract_reasoning(data: dict[str, Any]) -> str | None:
    """Extract reasoning from Ollama's OpenAI-compatible response."""
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError):
        return None
    return msg.get("reasoning") or msg.get("reasoning_content") or None
