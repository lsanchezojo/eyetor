"""Ollama payload construction."""

from __future__ import annotations

from eyetor.config import ProviderConfig
from eyetor.models.messages import Message
from eyetor.providers import create_provider
from eyetor.providers.ollama import OllamaProvider
from eyetor.tracking.context import tracking_context


def test_ollama_disables_reasoning_by_default() -> None:
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        model="gemma4:12b-qat",
        thinking=False,
    )

    payload = provider._build_payload(
        [Message(role="user", content="hi")],
        tools=None,
        temperature=0.0,
    )

    assert payload["reasoning"] == {"effort": "none"}


def test_ollama_keeps_reasoning_available_when_enabled() -> None:
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        model="gemma4:12b-qat",
        thinking=True,
    )

    payload = provider._build_payload(
        [Message(role="user", content="hi")],
        tools=None,
        temperature=0.0,
    )

    assert "reasoning" not in payload
    assert payload["think"] is True


def test_ollama_disables_reasoning_for_direct_text_phases() -> None:
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        model="gemma4:12b-qat",
        thinking=True,
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

        assert payload["reasoning"] == {"effort": "none"}
        assert "think" not in payload


def test_ollama_phase_generation_cap_overrides_global() -> None:
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        model="gemma4:12b-qat",
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


def test_factory_passes_ollama_local_options() -> None:
    provider = create_provider(
        ProviderConfig(
            type="ollama",
            base_url="http://localhost:11434/v1",
            model="gemma4:12b-qat",
            thinking=True,
            request_timeout=12,
            max_tokens=34,
            max_tokens_by_phase={"main": 56},
        )
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.thinking is True
    assert provider.request_timeout == 12
    assert provider.max_tokens == 34
    assert provider.max_tokens_by_phase == {"main": 56}
