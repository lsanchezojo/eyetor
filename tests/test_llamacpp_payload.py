"""llama.cpp payload construction."""

from __future__ import annotations

from eyetor.models.messages import Message
from eyetor.providers.llamacpp import LlamaCppProvider
from eyetor.tracking.context import tracking_context


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
