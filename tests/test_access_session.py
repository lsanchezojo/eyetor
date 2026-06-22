"""Integration: ChatSession prunes its tool registry per access policy."""

from __future__ import annotations

from unittest.mock import MagicMock

from eyetor.access import resolve
from eyetor.chat.session import ChatSession
from eyetor.config import AccessPolicy, VectorConfig
from eyetor.models.agents import AgentConfig
from eyetor.models.tools import ToolDefinition, ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name, group in [
        ("kb_search", "kb"),
        ("generate_image", "image"),
        ("skill_shell", None),
        ("skill_shopping", None),
    ]:
        reg.register(
            ToolDefinition(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                handler=None,
                group=group,
            )
        )
    return reg


def _session(root: VectorConfig, session_id: str) -> ChatSession:
    cfg = AgentConfig(name="x", provider="p", model="m", system_prompt="")
    return ChatSession(
        session_id=session_id,
        config=cfg,
        provider=MagicMock(),
        tool_registry=_registry(),
        root_config=root,
        access=resolve(root, session_id),
    )


def test_restricted_session_only_keeps_allowed_tools():
    root = VectorConfig(
        access={"telegram-1": AccessPolicy(tools=["kb_search"], skills=["shopping"])}
    )
    s = _session(root, "telegram-1")
    assert set(s.tool_registry.list_names()) == {"kb_search", "skill_shopping"}
    # the model never sees the blocked tools
    defs, _ = s._turn_tool_defs("genera una imagen de un gato")
    assert "generate_image" not in {t.name for t in (defs or [])}


def test_unrestricted_when_no_access_config():
    root = VectorConfig()
    s = _session(root, "telegram-1")
    assert set(s.tool_registry.list_names()) == {
        "kb_search",
        "generate_image",
        "skill_shell",
        "skill_shopping",
    }


def test_unlisted_chat_without_default_keeps_nothing():
    root = VectorConfig(access={"cli": AccessPolicy(tools=["*"], skills=["*"])})
    s = _session(root, "telegram-999")
    assert s.tool_registry.list_names() == []


def test_does_not_mutate_shared_registry():
    root = VectorConfig(access={"telegram-1": AccessPolicy(tools=["kb_search"])})
    cfg = AgentConfig(name="x", provider="p", model="m", system_prompt="")
    shared = _registry()
    ChatSession(
        session_id="telegram-1",
        config=cfg,
        provider=MagicMock(),
        tool_registry=shared,
        root_config=root,
        access=resolve(root, "telegram-1"),
    )
    # shared registry is untouched
    assert len(shared.list_names()) == 4
