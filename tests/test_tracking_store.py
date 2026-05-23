"""Tests for eyetor.tracking.store — schema migration and new dimensions."""

from __future__ import annotations

from pathlib import Path

from eyetor.tracking.store import TrackingStore


def _record(store: TrackingStore, **over):
    kw = dict(
        session_id="s1",
        provider="openrouter",
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=20,
        estimated_cost=0.01,
    )
    kw.update(over)
    store.record(**kw)


def test_migrations_are_idempotent(tmp_path: Path):
    db = tmp_path / "usage.db"
    s1 = TrackingStore(db)
    _record(s1)
    s1.close()
    # Reopening re-runs _DDL + _apply_migrations on an existing DB
    s2 = TrackingStore(db)
    cols = {
        row[1]
        for row in s2._conn.execute("PRAGMA table_info(usage)").fetchall()
    }
    for c in (
        "agent",
        "phase",
        "channel",
        "tool_count",
        "msg_count",
        "trace_id",
        "prompt_digest",
        "response_digest",
    ):
        assert c in cols, f"missing column {c}"
    # Old row still queryable after migration
    assert len(s2.get_recent(limit=10)) == 1
    s2.close()


def test_record_roundtrips_new_dimensions(tmp_path: Path):
    store = TrackingStore(tmp_path / "u.db")
    _record(
        store,
        agent="planner",
        phase="chain_plan",
        channel="cli",
        tool_count=2,
        msg_count=5,
        trace_id="abc123",
        prompt_digest="sha256:deadbeef|hola",
        response_digest="sha256:cafe|resp",
    )
    (r,) = store.get_recent(limit=1)
    assert r.agent == "planner"
    assert r.phase == "chain_plan"
    assert r.channel == "cli"
    assert r.tool_count == 2
    assert r.msg_count == 5
    assert r.trace_id == "abc123"
    assert r.prompt_digest == "sha256:deadbeef|hola"
    assert r.response_digest == "sha256:cafe|resp"
    store.close()


def test_get_summary_default_groups_provider_model(tmp_path: Path):
    store = TrackingStore(tmp_path / "u.db")
    _record(store, phase="main", model="m1")
    _record(store, phase="compaction", model="m1")
    # No grouping args: identical legacy behaviour (one row per provider+model)
    summ = store.get_summary(period="day")
    assert len(summ) == 1
    assert summ[0].calls == 2
    assert summ[0].agent == "" and summ[0].phase == ""
    store.close()


def test_get_summary_group_and_filter_by_phase(tmp_path: Path):
    store = TrackingStore(tmp_path / "u.db")
    _record(store, phase="main")
    _record(store, phase="main")
    _record(store, phase="compaction")

    by_phase = {s.phase: s for s in store.get_summary(group_by_phase=True)}
    assert by_phase["main"].calls == 2
    assert by_phase["compaction"].calls == 1

    only = store.get_summary(phase="compaction")
    assert len(only) == 1 and only[0].calls == 1
    store.close()
