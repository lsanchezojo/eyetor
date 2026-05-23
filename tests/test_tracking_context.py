"""Tests for eyetor.tracking.context — contextvars, digest, helpers."""

from __future__ import annotations

from eyetor.tracking.context import (
    current_agent,
    current_phase,
    effective_phase,
    make_digest,
    new_trace_id,
    tracking_context,
)


def test_tracking_context_sets_and_resets():
    assert current_agent.get() == ""
    with tracking_context(agent="planner", phase="main"):
        assert current_agent.get() == "planner"
        assert current_phase.get() == "main"
    # Reset on exit
    assert current_agent.get() == ""
    assert current_phase.get() == ""


def test_tracking_context_nested_restores_outer():
    with tracking_context(phase="routing"):
        assert current_phase.get() == "routing"
        with tracking_context(phase="agent"):
            assert current_phase.get() == "agent"
        # Inner reset restores the outer value, not the default
        assert current_phase.get() == "routing"
    assert current_phase.get() == ""


def test_tracking_context_only_touches_passed_vars():
    with tracking_context(agent="a"):
        with tracking_context(phase="compaction"):
            # agent untouched by the inner context
            assert current_agent.get() == "a"
            assert current_phase.get() == "compaction"
        assert current_agent.get() == "a"
        assert current_phase.get() == ""


def test_effective_phase_precedence():
    # Default: returns the fallback
    assert effective_phase("agent") == "agent"
    # When a phase is already set, that wins (routing beats agent)
    with tracking_context(phase="routing"):
        assert effective_phase("agent") == "routing"


def test_new_trace_id_unique_and_short():
    a, b = new_trace_id(), new_trace_id()
    assert a != b
    assert len(a) == 16 and a.isalnum()


def test_make_digest_format_and_truncation():
    d = make_digest("  hello\n  world  ", preview_chars=5)
    prefix, preview = d.split("|", 1)
    assert prefix.startswith("sha256:")
    assert len(prefix) == len("sha256:") + 16
    # Whitespace collapsed, then truncated to 5 chars
    assert preview == "hello"


def test_make_digest_handles_none():
    d = make_digest(None)
    assert d.startswith("sha256:")
    assert d.endswith("|")  # empty preview, separator always present
