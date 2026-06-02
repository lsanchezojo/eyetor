"""Tests for provider fallback behavior."""

from __future__ import annotations

import asyncio

import httpx

from eyetor.models.messages import CompletionResult, Message, TokenUsage
from eyetor.providers.base import BaseProvider
from eyetor.providers.fallback import FallbackProvider


class _Provider(BaseProvider):
    def __init__(
        self,
        content: str,
        *,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        super().__init__(base_url="http://example.test", model="fake")
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage
        self.reasoning_content = reasoning_content
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        return CompletionResult(
            message=Message(role="assistant", content=self.content),
            usage=self.usage,
            model=self.model,
            finish_reason=self.finish_reason,
            reasoning_content=self.reasoning_content,
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


class _FailingProvider(BaseProvider):
    def __init__(self, exc: Exception) -> None:
        super().__init__(base_url="http://example.test", model="fake")
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        raise self.exc

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://example.test/chat/completions")
    response = httpx.Response(
        status_code,
        request=request,
        text="context exceeded",
    )
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


def test_empty_completion_falls_back_to_next_provider() -> None:
    first = _Provider("")
    second = _Provider("respuesta")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == "respuesta"
    assert first.calls == 1
    assert second.calls == 1
    assert provider.last_used_provider_index == 1


def test_empty_completion_logs_metadata(caplog) -> None:
    first = _Provider(
        "",
        finish_reason="length",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=768, total_tokens=778),
        reasoning_content="internal",
    )
    second = _Provider("respuesta")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == "respuesta"
    assert "finish_reason=length" in caplog.text
    assert "completion_tokens=768" in caplog.text
    assert "reasoning_len=8" in caplog.text


def test_last_empty_completion_is_returned_if_all_are_empty() -> None:
    first = _Provider("")
    second = _Provider("")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == ""
    assert first.calls == 1
    assert second.calls == 1
    assert provider.last_used_provider_index == 1


def test_http_400_falls_back_to_next_provider() -> None:
    first = _FailingProvider(_http_status_error(400))
    second = _Provider("fallback")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == "fallback"
    assert first.calls == 1
    assert second.calls == 1
    assert provider.last_used_provider_index == 1


def test_read_error_falls_back_to_next_provider() -> None:
    first = _FailingProvider(httpx.ReadError("local model disconnected"))
    second = _Provider("fallback")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == "fallback"
    assert first.calls == 1
    assert second.calls == 1
    assert provider.last_used_provider_index == 1


def test_malformed_response_falls_back_to_next_provider() -> None:
    first = _FailingProvider(KeyError("choices"))
    second = _Provider("fallback")
    provider = FallbackProvider([first, second])

    result = asyncio.run(provider.complete([Message(role="user", content="hola")]))

    assert result.message.content == "fallback"
    assert first.calls == 1
    assert second.calls == 1
    assert provider.last_used_provider_index == 1
