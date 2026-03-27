"""FallbackProvider — tries providers in order on transient failures."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from eyetor.models.messages import Message
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({"500", "502", "503", "529"})
_RETRYABLE_ERRORS = frozenset({"timeout", "connection_error"})


class FallbackProvider(BaseProvider):
    """Tries providers in order, falling back on transient failures.

    Retries are triggered by:
    - httpx.TimeoutException   → "timeout"
    - httpx.ConnectError       → "connection_error"
    - httpx.HTTPStatusError with status codes in retry_on set
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        retry_on: set[str] | None = None,
    ) -> None:
        first = providers[0]
        super().__init__(base_url=first.base_url, model=first.model, api_key=first.api_key)
        self._providers = providers
        self._retry_on = retry_on or (_RETRYABLE_ERRORS | _RETRYABLE_STATUS)

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException) and "timeout" in self._retry_on:
            return True
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)) and "connection_error" in self._retry_on:
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return str(exc.response.status_code) in self._retry_on
        return False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> Message:
        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                return await provider.complete(messages, tools, temperature)
            except Exception as exc:
                if self._should_retry(exc):
                    logger.warning(
                        "Provider %s failed (%s), trying next in chain",
                        provider,
                        type(exc).__name__,
                    )
                    last_exc = exc
                else:
                    raise
        raise RuntimeError("All providers in fallback chain failed") from last_exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                async for token in provider.stream(messages, tools, temperature):
                    yield token
                return
            except Exception as exc:
                if self._should_retry(exc):
                    logger.warning(
                        "Provider %s stream failed (%s), trying next in chain",
                        provider,
                        type(exc).__name__,
                    )
                    last_exc = exc
                else:
                    raise
        raise RuntimeError("All providers in fallback chain failed") from last_exc
