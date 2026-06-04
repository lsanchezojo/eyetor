"""llama.cpp payload construction."""

from __future__ import annotations

import asyncio
import json

import httpx

from eyetor.models.messages import Message
from eyetor.providers.llamacpp import LlamaCppProvider
from eyetor.tracking.context import tracking_context


def _completion_response(content: str = "", reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "model": "default",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class _QueuedTransport:
    """Serves queued JSON responses and records each request body."""

    def __init__(self, provider: LlamaCppProvider, responses: list[dict]) -> None:
        self.responses = responses
        self.requests: list[dict] = []
        self._transport = httpx.MockTransport(self._handler)
        orig_client = provider._client

        def _client(timeout: float = 120.0) -> httpx.AsyncClient:
            client = orig_client(timeout=timeout)
            client._transport = self._transport
            return client

        provider._client = _client  # type: ignore[method-assign]

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.responses.pop(0)
        return httpx.Response(200, json=payload)


def test_complete_retries_without_thinking_on_think_only_degeneration() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=True,
        reasoning_budget=384,
    )
    transport = _QueuedTransport(
        provider,
        [
            _completion_response(content="", reasoning="hmm"),  # degenerate
            _completion_response(content="aquí tienes"),         # retry succeeds
        ],
    )

    with tracking_context(phase="main"):
        result = asyncio.run(
            provider.complete([Message(role="user", content="lista")])
        )

    assert result.message.content == "aquí tienes"
    assert len(transport.requests) == 2
    assert transport.requests[0]["chat_template_kwargs"]["enable_thinking"] is True
    assert transport.requests[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_budget_tokens" not in transport.requests[1]


def test_complete_no_retry_when_first_pass_usable() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=True,
    )
    transport = _QueuedTransport(
        provider, [_completion_response(content="respuesta directa")]
    )

    with tracking_context(phase="main"):
        result = asyncio.run(
            provider.complete([Message(role="user", content="hola")])
        )

    assert result.message.content == "respuesta directa"
    assert len(transport.requests) == 1


def test_complete_propagates_double_empty(caplog) -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=True,
    )
    transport = _QueuedTransport(
        provider,
        [
            _completion_response(content="", reasoning="a"),
            _completion_response(content=""),
        ],
    )

    with tracking_context(phase="main"):
        result = asyncio.run(
            provider.complete([Message(role="user", content="hola")])
        )

    # Both passes empty → the empty retry is returned so FallbackProvider escalates.
    assert (result.message.content or "") == ""
    assert len(transport.requests) == 2
    assert "no-thinking retry still empty" in caplog.text


def test_llamacpp_sets_global_generation_cap() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        max_tokens=123,
    )

    payload = provider._build_payload(
        [Message(role="user", content="hi")],
        tools=None,
        temperature=0.0,
    )

    assert payload["max_tokens"] == 123
    assert payload["n_predict"] == 123


def test_llamacpp_phase_generation_cap_overrides_global() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        max_tokens=123,
        max_tokens_by_phase={"main": 45},
    )

    with tracking_context(phase="main"):
        payload = provider._build_payload(
            [Message(role="user", content="hi")],
            tools=None,
            temperature=0.0,
        )

    assert payload["max_tokens"] == 45
    assert payload["n_predict"] == 45


def test_llamacpp_sets_reasoning_budget_when_thinking() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=True,
        reasoning_budget=1024,
    )

    payload = provider._build_payload(
        [Message(role="user", content="hi")],
        tools=None,
        temperature=0.0,
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["reasoning_budget_tokens"] == 1024


def test_llamacpp_explicitly_disables_thinking_when_config_disabled() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=False,
    )

    payload = provider._build_payload(
        [Message(role="user", content="hi")],
        tools=None,
        temperature=0.0,
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_llamacpp_disables_thinking_for_direct_text_phases() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="default",
        thinking=True,
        reasoning_budget=1024,
    )

    for phase in (
        "compaction",
        "degeneration_recovery",
        "loop_break",
        "chain_synthesize",
    ):
        with tracking_context(phase=phase):
            payload = provider._build_payload(
                [Message(role="user", content="hi")],
                tools=None,
                temperature=0.0,
            )

        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "reasoning_budget_tokens" not in payload
