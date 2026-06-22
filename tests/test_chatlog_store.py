"""Tests for eyetor.chatlog.store — the per-day, per-chat conversation archive."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from eyetor.chatlog.store import ChatLogStore


def _store(tmp_path: Path) -> ChatLogStore:
    return ChatLogStore(tmp_path / "chatlog.db")


def test_record_and_read_day_chronological(tmp_path: Path):
    s = _store(tmp_path)
    day = datetime(2026, 6, 22, 10, 0, 0)
    s.record("telegram-1", "Luis", "hola a todos", when=day)
    s.record("telegram-1", "Ana", "buenas", when=datetime(2026, 6, 22, 10, 1, 0))
    s.record("telegram-1", "Eyetor", "hola, ¿en qué ayudo?", when=day)

    msgs = s.read_day("telegram-1", "2026-06-22")
    assert [m.sender for m in msgs] == ["Luis", "Ana", "Eyetor"]
    assert msgs[0].content == "hola a todos"
    assert all(m.day == "2026-06-22" for m in msgs)


def test_search_scoped_to_session(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "mañana vamos de excursión al monte")
    s.record("telegram-2", "Otro", "excursión secreta en otro grupo")

    hits = s.search("telegram-1", "excursión")
    assert len(hits) == 1
    assert hits[0].session_id == "telegram-1"
    assert "excursión" in hits[0].content


def test_search_day_filter(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "tema presupuesto", when=datetime(2026, 6, 20))
    s.record("telegram-1", "Luis", "otra vez presupuesto", when=datetime(2026, 6, 21))

    hits = s.search("telegram-1", "presupuesto", day="2026-06-21")
    assert len(hits) == 1
    assert hits[0].day == "2026-06-21"


def test_list_days_newest_first_with_counts(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "a", when=datetime(2026, 6, 20))
    s.record("telegram-1", "Ana", "b", when=datetime(2026, 6, 22))
    s.record("telegram-1", "Ana", "c", when=datetime(2026, 6, 22))

    days = s.list_days("telegram-1")
    assert days[0] == {"day": "2026-06-22", "count": 2}
    assert days[1] == {"day": "2026-06-20", "count": 1}


def test_empty_content_is_ignored(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "   ")
    assert s.list_days("telegram-1") == []


def test_retention_purges_old_messages(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "viejo", when=datetime(2026, 6, 1))
    # recording a new message with retention should purge the old day
    s.record(
        "telegram-1", "Luis", "nuevo", when=datetime(2026, 6, 22), retention_days=5
    )
    days = {d["day"] for d in s.list_days("telegram-1")}
    assert days == {"2026-06-22"}


def test_search_falls_back_for_special_chars(tmp_path: Path):
    s = _store(tmp_path)
    s.record("telegram-1", "Luis", "el código es C-137 según dijo")
    # hyphen/punctuation should still match via sanitization or LIKE fallback
    hits = s.search("telegram-1", "C-137")
    assert len(hits) == 1
