"""Abstract base class for LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from eyetor.models.messages import CompletionResult, Message, StreamingResponse
from eyetor.models.tools import ToolDefinition

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base class for provider-level errors that callers may want to catch."""


class ContextOverflowError(ProviderError):
    """Raised when the request exceeds the provider's context window.

    Treated as retryable by FallbackProvider so a longer-context backend
    can take over.
    """

    def __init__(self, message: str, *, n_prompt_tokens: int | None = None, n_ctx: int | None = None) -> None:
        super().__init__(message)
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx


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
        max_tokens: int | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_predict = num_predict
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.stop = stop
        self.extra_body = extra_body or {}
        self.options = options or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        thinking: bool | None = None,
    ) -> CompletionResult:
        """Send messages and return the assistant reply (non-streaming).

        ``thinking`` overrides the provider's default reasoning mode for this
        call only. ``None`` uses the provider's configured default; ``False``
        forces reasoning off even if the provider is configured with
        ``thinking=True`` (used for cheap auxiliary calls — classifier,
        synthesis — where deep reasoning just adds latency without value).
        Providers that don't support reasoning ignore the flag.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        """Send messages and return a StreamingResponse with text tokens and usage."""

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
        serialized_msgs = []
        for m in messages:
            d = m.model_dump(exclude_none=True)
            # Ensure 'content' is always present — some servers (llama.cpp)
            # reject messages that omit it (e.g. assistant messages with tool_calls).
            if "content" not in d:
                d["content"] = None
            serialized_msgs.append(d)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": serialized_msgs,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]
        optional_fields = {
            "max_tokens": self.max_tokens,
            "num_predict": self.num_predict,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stop": self.stop,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
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
