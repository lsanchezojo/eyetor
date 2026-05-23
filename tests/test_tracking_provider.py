"""Tests for eyetor.providers.tracking.TrackingProvider."""

from __future__ import annotations

import asyncio

import pytest

from eyetor.models.messages import (
    CompletionResult,
    Message,
    StreamingResponse,
    TokenUsage,
)
from eyetor.providers.base import BaseProvider
from eyetor.providers.tracking import TrackingProvider, UsageLimitExceeded
from eyetor.tracking.context import tracking_context


class _StubProvider(BaseProvider):
    """Inner provider returning canned usage."""

    def __init__(self, usage: TokenUsage, text: str = "Hello world!!!"):
        super().__init__(base_url="http://x", model="stub-model")
        self._usage = usage
        self._text = text

    async def complete(self, messages, tools=None, temperature=0.0):
        return CompletionResult(
            message=Message(role="assistant", content=self._text),
            usage=self._usage,
            model="stub-model",
            finish_reason="stop",
        )

    async def stream(self, messages, tools=None, temperature=0.0):
        async def _gen():
            for ch in self._text:
                yield ch

        return StreamingResponse(_gen(), self._usage)


class _FakeTracker:
    def __init__(self, limit_ok: bool = True):
        self._limit_ok = limit_ok
        self.records: list[dict] = []

    def check_limits(self, provider: str) -> bool:
        return self._limit_ok

    def record(self, **kwargs):
        self.records.append(kwargs)


def test_stream_records_real_tokens_not_char_count():
    usage = TokenUsage(prompt_tokens=3, completion_tokens=7, total_tokens=10)
    inner = _StubProvider(usage, text="Hello world!!!")  # 14 chars
    tracker = _FakeTracker()
    tp = TrackingProvider(inner, tracker, "openrouter")

    async def _drive():
        resp = await tp.stream([Message(role="user", content="hi")])
        return "".join([tok async for tok in resp])

    out = asyncio.run(_drive())
    assert out == "Hello world!!!"
    assert len(tracker.records) == 1
    rec = tracker.records[0]
    # Real token count from usage (7), NOT the 14-char fallback
    assert rec["completion_tokens"] == 7
    assert rec["prompt_tokens"] == 3
    assert rec["provider"] == "openrouter"


def test_complete_records_dimensions_from_context():
    usage = TokenUsage(prompt_tokens=5, completion_tokens=9)
    tp = TrackingProvider(_StubProvider(usage), _FakeTracker(), "gemini")
    tracker = tp._tracker

    async def _drive():
        with tracking_context(
            session_id="sess-1", agent="planner", phase="main",
            channel="cli", trace_id="trace-9",
        ):
            await tp.complete([Message(role="user", content="hi")])

    asyncio.run(_drive())
    rec = tracker.records[0]
    assert rec["session_id"] == "sess-1"
    assert rec["agent"] == "planner"
    assert rec["phase"] == "main"
    assert rec["channel"] == "cli"
    assert rec["trace_id"] == "trace-9"
    assert rec["msg_count"] == 1
    assert rec["prompt_digest"].startswith("sha256:")


def test_skip_limit_bypasses_exceeded_limit_but_still_records():
    tp = TrackingProvider(
        _StubProvider(TokenUsage(prompt_tokens=1, completion_tokens=1)),
        _FakeTracker(limit_ok=False),  # daily limit exceeded
        "openrouter",
    )
    tracker = tp._tracker

    # Without skip_limit: the call is blocked
    with pytest.raises(UsageLimitExceeded):
        asyncio.run(tp.complete([Message(role="user", content="hi")]))
    assert tracker.records == []

    # With skip_limit (e.g. compaction/routing): runs and is still recorded
    async def _drive():
        with tracking_context(skip_limit_flag=True):
            await tp.complete([Message(role="user", content="hi")])

    asyncio.run(_drive())
    assert len(tracker.records) == 1
    assert tracker.records[0]["provider"] == "openrouter"
