"""Integration: SessionManager with ChatGPT-style conversations enabled."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from eyetor.chat.manager import SessionManager
from eyetor.config import VectorConfig
from eyetor.models.agents import AgentConfig


def _manager(tmp_path, persist: bool = True, enable: bool = True) -> SessionManager:
    root = VectorConfig(
        sessions={"persist": persist, "dir": str(tmp_path)},
    )
    cfg = AgentConfig(name="x", provider="p", model="m", system_prompt="")
    return SessionManager(
        cfg, MagicMock(), root_config=root, enable_conversations=enable
    )


def test_get_or_create_binds_active_conversation(tmp_path):
    mgr = _manager(tmp_path)
    s = mgr.get_or_create("telegram-1")
    assert s.conversation_id and len(s.conversation_id) == 8
    assert s._persist_path.name == f"telegram-1.{s.conversation_id}.jsonl"
    # Same instance while cached.
    assert mgr.get_or_create("telegram-1") is s


def test_new_conversation_pops_cache_and_changes_file(tmp_path):
    mgr = _manager(tmp_path)
    s1 = mgr.get_or_create("telegram-1")
    # Titled (non-empty) so new() mints a fresh id instead of reusing.
    mgr._conv_index.touch("telegram-1", s1.conversation_id, "hola mundo")
    conv2 = mgr.new_conversation("telegram-1")
    s2 = mgr.get_or_create("telegram-1")
    assert s2 is not s1
    assert s2.conversation_id == conv2 != s1.conversation_id


def test_switch_conversation_restores_history(tmp_path):
    mgr = _manager(tmp_path)
    s1 = mgr.get_or_create("cli-luis")
    conv1 = s1.conversation_id
    # Simulate a persisted user turn in conversation 1.
    s1._persist_path.write_text('{"role": "user", "content": "hola"}\n')
    mgr._conv_index.touch("cli-luis", conv1, "hola")
    mgr.new_conversation("cli-luis")
    s2 = mgr.get_or_create("cli-luis")
    assert s2.conversation_id != conv1
    assert len(s2.get_history()) == 0

    assert mgr.switch_conversation("cli-luis", conv1) is True
    s3 = mgr.get_or_create("cli-luis")
    assert s3.conversation_id == conv1
    assert [m.content for m in s3.get_history()] == ["hola"]


def test_legacy_file_migrates_on_first_contact(tmp_path):
    legacy = tmp_path / "telegram-7.jsonl"
    legacy.write_text('{"role": "user", "content": "viejo"}\n')
    mgr = _manager(tmp_path)
    s = mgr.get_or_create("telegram-7")
    assert not legacy.exists()
    assert [m.content for m in s.get_history()] == ["viejo"]


def test_disabled_conversations_fall_back(tmp_path):
    mgr = _manager(tmp_path, enable=False)
    s = mgr.get_or_create("telegram-1")
    assert s.conversation_id is None
    assert s._persist_path.name == "telegram-1.jsonl"
    assert mgr.new_conversation("telegram-1") is None  # wipes in place
    assert mgr.list_conversations("telegram-1") == []
    assert mgr.switch_conversation("telegram-1", "x") is False


def test_no_persist_no_index(tmp_path):
    mgr = _manager(tmp_path, persist=False)
    s = mgr.get_or_create("telegram-1")
    assert s.conversation_id is None
    assert mgr.new_conversation("telegram-1") is None
    assert not (tmp_path / "conversations.json").exists()
