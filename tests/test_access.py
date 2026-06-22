"""Tests for eyetor.access — per-chat tool/skill access resolution."""

from __future__ import annotations

from eyetor.access import resolve
from eyetor.config import AccessPolicy, VectorConfig


def _cfg(access: dict | None) -> VectorConfig:
    return VectorConfig(access=access or {})


def test_no_access_config_is_unrestricted():
    acc = resolve(_cfg(None), "telegram-123")
    assert acc.unrestricted is True
    assert acc.allows_tool("anything") is True
    assert acc.allows_tool("skill_shell") is True
    assert acc.allowed_summary() == ""


def test_listed_chat_allowlist_by_name():
    cfg = _cfg({"telegram-1": AccessPolicy(tools=["kb_search"], skills=["shopping"])})
    acc = resolve(cfg, "telegram-1")
    assert acc.unrestricted is False
    assert acc.allows_tool("kb_search") is True
    assert acc.allows_tool("generate_image") is False
    assert acc.allows_tool("skill_shopping") is True
    assert acc.allows_tool("skill_shell") is False


def test_skill_name_hyphen_maps_to_underscore_tool():
    cfg = _cfg({"telegram-1": AccessPolicy(skills=["my-skill"])})
    acc = resolve(cfg, "telegram-1")
    assert acc.allows_tool("skill_my_skill") is True


def test_wildcard_allows_all():
    cfg = _cfg({"cli": AccessPolicy(tools=["*"], skills=["*"])})
    acc = resolve(cfg, "cli")
    assert acc.unrestricted is False
    assert acc.all_tools and acc.all_skills
    assert acc.allows_tool("whatever") is True
    assert acc.allows_tool("skill_anything") is True
    assert acc.allowed_summary() == ""  # nothing to warn about when all allowed


def test_unlisted_chat_falls_back_to_default():
    cfg = _cfg(
        {
            "default": AccessPolicy(tools=["kb_search"]),
            "cli": AccessPolicy(tools=["*"], skills=["*"]),
        }
    )
    acc = resolve(cfg, "telegram-999")
    assert acc.allows_tool("kb_search") is True
    assert acc.allows_tool("generate_image") is False


def test_unlisted_chat_with_no_default_denies_everything():
    cfg = _cfg({"cli": AccessPolicy(tools=["*"], skills=["*"])})
    acc = resolve(cfg, "telegram-999")
    assert acc.unrestricted is False
    assert acc.allows_tool("kb_search") is False
    assert acc.allows_tool("skill_shopping") is False


def test_glob_pattern_matches_dynamic_cli_ids():
    cfg = _cfg(
        {
            "default": AccessPolicy(tools=[]),
            "cli-*": AccessPolicy(tools=["*"], skills=["*"]),
        }
    )
    acc = resolve(cfg, "cli-haziel-3892")
    assert acc.all_tools and acc.all_skills
    # default still applies to non-matching ids
    assert resolve(cfg, "telegram-1").allows_tool("kb_search") is False


def test_exact_match_wins_over_glob():
    cfg = _cfg(
        {
            "telegram-*": AccessPolicy(tools=["kb_search"]),
            "telegram-1": AccessPolicy(tools=["generate_image"]),
        }
    )
    acc = resolve(cfg, "telegram-1")
    assert acc.allows_tool("generate_image") is True
    assert acc.allows_tool("kb_search") is False


def test_allowed_summary_mentions_allowed_items_when_restricted():
    cfg = _cfg({"telegram-1": AccessPolicy(tools=["kb_search"], skills=["shopping"])})
    acc = resolve(cfg, "telegram-1")
    summary = acc.allowed_summary()
    assert "kb_search" in summary
    assert "shopping" in summary
