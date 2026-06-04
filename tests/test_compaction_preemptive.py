"""Tests for pre-LLM and post-fallback compaction behavior."""

from __future__ import annotations

import asyncio

from eyetor.chat.session import ChatSession
from eyetor.config import CompactionConfig, SessionsConfig, VectorConfig
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, Message


class _CompactionAwareProvider:
    model = "fake"
    last_used_provider_index: int | None = 0
    last_used_provider = None

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append(list(messages))
        if _is_summary_call(messages):
            return CompletionResult(message=Message(role="assistant", content="summary"))
        return CompletionResult(message=Message(role="assistant", content="final"))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


class _FallbackMarkerProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.actual_calls = 0
        self.last_used_provider_index: int | None = 0
        self.last_used_provider = None

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append(list(messages))
        if _is_summary_call(messages):
            self.last_used_provider_index = 0
            return CompletionResult(message=Message(role="assistant", content="summary"))

        self.actual_calls += 1
        if self.actual_calls == 1:
            self.last_used_provider_index = 1
            self.last_used_provider = "remote"
        else:
            self.last_used_provider_index = 0
            self.last_used_provider = None
        return CompletionResult(
            message=Message(role="assistant", content=f"final {self.actual_calls}")
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def _is_summary_call(messages: list[Message]) -> bool:
    return len(messages) == 1 and (messages[0].content or "").startswith("Summarize")


def _session(provider, root_config: VectorConfig) -> ChatSession:
    return ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake"),
        provider=provider,
        root_config=root_config,
    )


def test_preventive_compaction_runs_before_model_call() -> None:
    provider = _CompactionAwareProvider()
    root = VectorConfig(
        sessions=SessionsConfig(
            compaction=CompactionConfig(
                enabled=True,
                context_window=20,
                trigger_at_percent=0.5,
                keep_last_n_user_turns=1,
            )
        )
    )
    session = _session(provider, root)
    session._messages = [
        Message(role="user", content="old " * 80),
        Message(role="assistant", content="old answer"),
    ]

    assert asyncio.run(session.send_sync("new question")) == "final"

    assert len(provider.calls) == 2
    assert _is_summary_call(provider.calls[0])
    assert provider.calls[1][-1] == Message(role="user", content="new question")


def test_fallback_skips_forced_compaction_when_context_low() -> None:
    """A fallback on a low-context degeneration must NOT trigger compaction.

    The escalation was not caused by context overflow (e.g. an empty
    think-only completion at 46% of the window), so compacting is wasteful.
    """
    provider = _FallbackMarkerProvider()
    root = VectorConfig(
        sessions=SessionsConfig(
            compaction=CompactionConfig(
                enabled=True,
                context_window=100_000,
                trigger_at_percent=0.99,
                keep_last_n_user_turns=1,
            )
        )
    )
    session = _session(provider, root)

    assert asyncio.run(session.send_sync("first")) == "final 1"
    assert asyncio.run(session.send_sync("second")) == "final 2"

    # No summary call: forced compaction is skipped because context is low.
    assert len(provider.calls) == 2
    assert not any(_is_summary_call(c) for c in provider.calls)


def test_fallback_forces_compaction_when_context_high() -> None:
    """When context is genuinely under pressure, the forced flag is still set."""
    provider = _FallbackMarkerProvider()
    root = VectorConfig(
        sessions=SessionsConfig(
            compaction=CompactionConfig(
                enabled=True,
                context_window=20,
                trigger_at_percent=0.5,
                keep_last_n_user_turns=1,
            )
        )
    )
    session = _session(provider, root)
    provider.last_used_provider_index = 1
    session._messages = [Message(role="user", content="old " * 80)]

    session._mark_force_compact_after_fallback("main")

    assert session._force_compact_next is True
