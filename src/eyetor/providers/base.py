"""Abstract base class for LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from eyetor.models.messages import CompletionResult, Message
from eyetor.models.tools import ToolDefinition

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base for all LLM providers.

    All providers expose the same async interface; callers never need
    to know which backend is in use.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        ssl_verify: bool | str = True,
        temperature: float = 0.7,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send messages and return the assistant reply (non-streaming)."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Send messages and yield text tokens as they arrive (streaming)."""

    # ------------------------------------------------------------------
    # Helpers shared by all concrete providers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _client(self, timeout: float = 120.0) -> "httpx.AsyncClient":
        """Return a configured AsyncClient respecting ssl_verify setting."""
        import httpx
        if not self.ssl_verify:
            logger.warning(
                "SSL verification disabled for %s — insecure, only use behind a trusted proxy.",
                self.base_url,
            )
        return httpx.AsyncClient(timeout=timeout, verify=self.ssl_verify)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
