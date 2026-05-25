"""FallbackProvider — tries providers in order on transient failures."""

from __future__ import annotations

import logging

import httpx

from eyetor.models.messages import CompletionResult, Message, StreamingResponse
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
        # IMPORTANT: forward ``temperature`` from the first inner provider.
        # Without this, ``BaseProvider.__init__`` defaults to 0.7 and
        # ``FallbackProvider.temperature`` silently overrides the configured
        # value when callers do ``temperature=prov.temperature`` (e.g.
        # ``cli._run_start`` building the ``AgentConfig``).
        super().__init__(
            base_url=first.base_url,
            model=first.model,
            api_key=first.api_key,
            temperature=first.temperature,
        )
        self._providers = providers
        self._retry_on = retry_on or (_RETRYABLE_ERRORS | _RETRYABLE_STATUS)

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException) and "timeout" in self._retry_on:
            return True
        if (
            isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError))
            and "connection_error" in self._retry_on
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return str(exc.response.status_code) in self._retry_on
        return False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        last_exc: Exception | None = None
        last_empty: CompletionResult | None = None
        for idx, provider in enumerate(self._providers):
            try:
                result = await provider.complete(messages, tools, temperature)
                if _is_empty_completion(result) and idx < len(self._providers) - 1:
                    logger.warning(
                        "Provider %s returned empty completion, trying next in chain",
                        provider,
                    )
                    last_empty = result
                    continue
                return result
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
        if last_empty is not None:
            return last_empty
        raise RuntimeError("All providers in fallback chain failed") from last_exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                return await provider.stream(messages, tools, temperature)
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


def _is_empty_completion(result: CompletionResult) -> bool:
    message = result.message
    if message.tool_calls:
        return False
    return not (message.content or "").strip()
