"""FallbackProvider — tries providers in order on transient failures."""

from __future__ import annotations

import logging
from json import JSONDecodeError

import httpx

from eyetor.models.messages import CompletionResult, Message, StreamingResponse
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset(
    {"400", "408", "413", "422", "429", "500", "502", "503", "529"}
)
_RETRYABLE_ERRORS = frozenset({"timeout", "connection_error", "malformed_response"})


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
        self.last_used_provider_index: int | None = None
        self.last_used_provider: BaseProvider | None = None

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException) and "timeout" in self._retry_on:
            return True
        if (
            isinstance(exc, httpx.TransportError)
            and "connection_error" in self._retry_on
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return str(exc.response.status_code) in self._retry_on
        if isinstance(exc, (JSONDecodeError, KeyError, IndexError)):
            return "malformed_response" in self._retry_on
        return False

    def _log_provider_failure(self, provider: BaseProvider, exc: Exception) -> None:
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text[:200]
            logger.warning(
                "Provider %s failed with HTTP %d (%s), trying next in chain",
                provider,
                exc.response.status_code,
                body,
            )
            return
        logger.warning(
            "Provider %s failed (%s), trying next in chain",
            provider,
            type(exc).__name__,
        )

    def _mark_used(self, idx: int, provider: BaseProvider) -> None:
        self.last_used_provider_index = idx
        self.last_used_provider = provider
        if idx > 0:
            logger.info(
                "Fallback chain resolved with provider #%d: %s", idx + 1, provider
            )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        last_exc: Exception | None = None
        last_empty: CompletionResult | None = None
        last_empty_idx: int | None = None
        last_empty_provider: BaseProvider | None = None
        self.last_used_provider_index = None
        self.last_used_provider = None
        for idx, provider in enumerate(self._providers):
            try:
                result = await provider.complete(messages, tools, temperature)
                if _is_empty_completion(result) and idx < len(self._providers) - 1:
                    logger.warning(
                        "Provider %s returned empty completion, trying next in chain",
                        provider,
                    )
                    last_empty = result
                    last_empty_idx = idx
                    last_empty_provider = provider
                    continue
                self._mark_used(idx, provider)
                return result
            except Exception as exc:
                if self._should_retry(exc):
                    self._log_provider_failure(provider, exc)
                    last_exc = exc
                else:
                    logger.error(
                        "Provider %s failed with non-retryable %s",
                        provider,
                        type(exc).__name__,
                    )
                    raise
        if last_empty is not None:
            if last_empty_idx is not None and last_empty_provider is not None:
                self._mark_used(last_empty_idx, last_empty_provider)
            return last_empty
        raise RuntimeError("All providers in fallback chain failed") from last_exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        last_exc: Exception | None = None
        self.last_used_provider_index = None
        self.last_used_provider = None
        for idx, provider in enumerate(self._providers):
            try:
                result = await provider.stream(messages, tools, temperature)
                self._mark_used(idx, provider)
                return result
            except Exception as exc:
                if self._should_retry(exc):
                    self._log_provider_failure(provider, exc)
                    last_exc = exc
                else:
                    logger.error(
                        "Provider %s stream failed with non-retryable %s",
                        provider,
                        type(exc).__name__,
                    )
                    raise
        raise RuntimeError("All providers in fallback chain failed") from last_exc


def _is_empty_completion(result: CompletionResult) -> bool:
    message = result.message
    if message.tool_calls:
        return False
    return not (message.content or "").strip()
