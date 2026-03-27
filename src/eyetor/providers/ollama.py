"""Ollama LLM provider (OpenAI-compatible endpoint)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from eyetor.models.messages import Message
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, parse_sse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Provider adapter for Ollama's OpenAI-compatible API.

    Ollama does not require authentication for local instances.
    """

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
    ) -> Message:
        payload = self._build_payload(messages, tools, temperature, stream=False)
        async with self._client(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
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
