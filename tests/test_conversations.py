"""Unit tests for the per-chat conversation index."""

from __future__ import annotations

import json
from pathlib import Path

from eyetor.chat.conversations import ConversationIndex, derive_title, sanitize_id


def _index(tmp_path: Path) -> ConversationIndex:
    return ConversationIndex(tmp_path / "conversations.json")


def test_active_creates_first_conversation(tmp_path):
    idx = _index(tmp_path)
    conv = idx.active("telegram-1")
    assert len(conv) == 8
    # Stable on repeat calls.
    assert idx.active("telegram-1") == conv
    assert idx.active_id("telegram-1") == conv


def test_new_reuses_empty_conversation(tmp_path):
    idx = _index(tmp_path)
    first = idx.active("cli-luis")
    # Never titled -> still empty -> reused.
    assert idx.new("cli-luis") == first
    # After a user message it has a title -> new() mints a fresh one.
    idx.touch("cli-luis", first, "hola que tal")
    second = idx.new("cli-luis")
    assert second != first
    assert idx.active_id("cli-luis") == second


def test_switch_and_list_ordering(tmp_path):
    idx = _index(tmp_path)
    a = idx.active("t-1")
    idx.touch("t-1", a, "primera conversacion")
    b = idx.new("t-1")
    idx.touch("t-1", b, "segunda conversacion")

    items = idx.list("t-1")
    assert [it["id"] for it in items][0] == b  # newest-updated first
    assert items[0]["active"] is True

    assert idx.switch("t-1", a) is True
    assert idx.active_id("t-1") == a
    assert idx.switch("t-1", "nope1234") is False


def test_touch_sets_title_once_and_strips_sender_prefix(tmp_path):
    idx = _index(tmp_path)
    conv = idx.active("t-2")
    idx.touch("t-2", conv, "[Luis]: arregla el   login\nde odysseus")
    title = idx.list("t-2")[0]["title"]
    assert title == "arregla el login de odysseus"
    # Title is set only once.
    idx.touch("t-2", conv, "otro mensaje distinto")
    assert idx.list("t-2")[0]["title"] == title


def test_derive_title_truncates():
    long = "palabra " * 20
    title = derive_title(long)
    assert len(title) <= 49 and title.endswith("…")


def test_legacy_migration_renames_file(tmp_path):
    legacy = tmp_path / "telegram-99.jsonl"
    legacy.write_text('{"role": "user", "content": "hola"}\n')
    idx = _index(tmp_path)
    conv = idx.active("telegram-99", legacy_file=legacy)
    assert not legacy.exists()
    assert (tmp_path / f"telegram-99.{conv}.jsonl").exists()


def test_corrupt_index_recovers(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text("{ not json !!!")
    idx = ConversationIndex(path)
    conv = idx.active("t-3")
    assert idx.active_id("t-3") == conv  # rewritten cleanly
    json.loads(path.read_text())  # valid again


def test_sanitize_id():
    assert sanitize_id("a/b:c@d\\e") == "a_b_c_d_e"
