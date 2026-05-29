"""Tests for provider fallback behavior."""

from __future__ import annotations

import asyncio

import httpx

from eyetor.models.messages import CompletionResult, Message
from eyetor.providers.base import BaseProvider
from eyetor.providers.fallback import FallbackProvider


class _Provider(BaseProvider):
    def __init__(self, content: str) -> None:
        super().__init__(base_url="http://example.test", model="fake")
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        return CompletionResult(message=Message(role="assistant", content=self.content))

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
