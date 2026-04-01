"""llama.cpp server LLM provider (OpenAI-compatible)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from eyetor.models.messages import CompletionResult, Message
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, parse_sse

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseProvider):
    """Provider adapter for llama.cpp server's OpenAI-compatible API.

    Authentication is optional — only used if api_key is set.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        payload = self._build_payload(messages, tools, temperature, stream=False)
        async with self._client(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
            if response.status_code >= 400:
                body = response.text[:500]
                logger.error(
                    "llama.cpp %d error: %s", response.status_code, body,
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
        async with self._client(timeout=300.0) as client:
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
