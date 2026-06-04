"""Tests for the /usage report formatting in Telegram."""

from __future__ import annotations

from datetime import datetime, timezone

from eyetor.channels.telegram import _format_usage_text
from eyetor.tracking.store import UsageRecord


def _rec(**kw) -> UsageRecord:
    base = dict(
        id=0,
        session_id="telegram-1",
        provider="llamacpp",
        model="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        prompt_tokens=1000,
        completion_tokens=100,
        estimated_cost=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        speed_tps=9.1,
        finish_reason="stop",
        phase="main",
    )
    base.update(kw)
    return UsageRecord(**base)


class _FakeTracker:
    def __init__(self, records: list[UsageRecord]) -> None:
        self._records = records

    def get_records(self, period: str = "day"):
        return list(self._records)


def test_usage_report_shows_provider_in_group_header() -> None:
    tracker = _FakeTracker(
        [
            _rec(provider="llamacpp", model="Qwen3.6-35B-A3B", phase="main"),
            _rec(
                provider="openrouter",
                model="nvidia/nemotron-3-super-120b-a12b:free",
                phase="chain_plan",
                completion_tokens=248,
            ),
        ]
    )

    text = _format_usage_text(tracker, session_id="telegram-1")

    # Provider name appears explicitly in each model group header.
    assert "llamacpp" in text
    assert "openrouter" in text
    # Model short name is rendered alongside the provider.
    assert "nemotron-3-super-120b-a12b:free" in text


def test_usage_report_omits_daily_average() -> None:
    tracker = _FakeTracker([_rec(), _rec(phase="chain_plan")])

    text = _format_usage_text(tracker, session_id="telegram-1")

    assert "Promedio diario" not in text


def test_usage_report_phase_breakdown_includes_tokens_and_cost() -> None:
    tracker = _FakeTracker(
        [
            _rec(phase="main", prompt_tokens=5000, completion_tokens=200),
            _rec(phase="chain_plan", prompt_tokens=800, completion_tokens=50),
        ]
    )

    text = _format_usage_text(tracker, session_id="telegram-1")

    assert "Por fase" in text
    # Internal phase names are kept (not translated) and show consumption.
    assert "main:" in text
    assert "chain_plan:" in text
    assert "tok" in text


def test_usage_report_handles_no_activity() -> None:
    text = _format_usage_text(_FakeTracker([]), session_id="telegram-1")
    assert "Sin actividad registrada" in text
