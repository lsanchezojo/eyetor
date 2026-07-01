"""Tests for conditional tool loading (keyword gating + sticky window)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from eyetor.chat.session import ChatSession
from eyetor.chat.tool_gating import KNOWN_GROUPS, select_groups
from eyetor.config import VectorConfig
from eyetor.models.agents import AgentConfig
from eyetor.models.tools import ToolDefinition, ToolRegistry


# ----------------------------------------------------------------------
# select_groups — keyword triggers
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("hola", set()),
        ("¿qué tal estás?", set()),
        ("recuérdame comprar pan mañana a las 9", {"scheduler"}),
        ("prográmame una tarea cada lunes", {"scheduler"}),
        ("avísame el jueves", {"scheduler"}),
        ("genera una imagen de un gato", {"image"}),
        ("dibújame un logo", {"image"}),
        ("instala el paquete ffmpeg", {"install"}),
        ("busca en la documentación cómo configurar esto", {"kb"}),
    ],
)
def test_select_groups(text, expected):
    assert select_groups(text) == expected


def test_select_groups_accent_insensitive():
    # 'recuérdame' (accented) and 'recuerdame' (plain) must both trigger.
    assert "scheduler" in select_groups("recuérdame algo")
    assert "scheduler" in select_groups("recuerdame algo")


def test_known_groups_match_patterns():
    assert KNOWN_GROUPS == {"scheduler", "image", "kb", "install"}


# ----------------------------------------------------------------------
# ChatSession integration
# ----------------------------------------------------------------------

def _make_session(
    *,
    enabled: bool = True,
    sticky_turns: int = 2,
    always_on_groups: list[str] | None = None,
) -> ChatSession:
    cfg = AgentConfig(name="x", provider="p", model="m", system_prompt="")
    root = VectorConfig()
    root.sessions.tool_gating.enabled = enabled
    root.sessions.tool_gating.sticky_turns = sticky_turns
    root.sessions.tool_gating.always_on_groups = always_on_groups or []

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="skill_shell", description="run", parameters={"type": "object", "properties": {}},
            handler=None,
        )
    )
    registry.register(
        ToolDefinition(
            name="shopping_receipt_add",
            description="Register a shopping receipt from structured data.",
            parameters={"type": "object", "properties": {}},
            handler=None,
        )
    )
    registry.register(
        ToolDefinition(
            name="schedule_task", description="sched", parameters={"type": "object", "properties": {}},
            handler=None, group="scheduler",
        )
    )
    registry.register(
        ToolDefinition(
            name="generate_image", description="img", parameters={"type": "object", "properties": {}},
            handler=None, group="image",
        )
    )
    registry.register(
        ToolDefinition(
            name="kb_search", description="kb", parameters={"type": "object", "properties": {}},
            handler=None, group="kb",
        )
    )
    return ChatSession(
        session_id="telegram-1",
        config=cfg,
        provider=MagicMock(),
        tool_registry=registry,
        root_config=root,
    )


def _names(tool_defs):
    return {t.name for t in (tool_defs or [])}


def test_trivial_turn_only_always_on():
    s = _make_session()
    defs, groups = s._turn_tool_defs("hola")
    assert _names(defs) == {"skill_shell", "shopping_receipt_add"}
    assert groups == set()


def test_triggered_group_included():
    s = _make_session()
    defs, groups = s._turn_tool_defs("recuérdame comprar pan mañana a las 9")
    assert _names(defs) == {"skill_shell", "shopping_receipt_add", "schedule_task"}
    assert groups == {"scheduler"}


def test_gating_disabled_sends_all():
    s = _make_session(enabled=False)
    defs, _ = s._turn_tool_defs("hola")
    assert _names(defs) == {
        "skill_shell",
        "shopping_receipt_add",
        "schedule_task",
        "generate_image",
        "kb_search",
    }


def test_always_on_group_sent_without_trigger():
    # kb is not triggered by a plain content question, but always_on_groups
    # keeps it available so the model can hit the knowledge base.
    s = _make_session(always_on_groups=["kb"])
    defs, groups = s._turn_tool_defs("¿cuáles son las capacidades heroicas del mago?")
    assert "kb_search" in _names(defs)
    assert "kb" in groups
    # Non-always-on gated groups are still excluded.
    assert "generate_image" not in _names(defs)


def test_sticky_keeps_group_for_followups():
    s = _make_session(sticky_turns=2)
    # Turn 1: trigger scheduler and simulate the tool being used.
    s._turn_tool_defs("recuérdame algo a las 9")
    s._mark_group_used("schedule_task")
    # Turn 2: a follow-up with no scheduler keywords still sees scheduler.
    defs2, groups2 = s._turn_tool_defs("cancélalo")
    assert "schedule_task" in _names(defs2)
    assert "scheduler" in groups2
    # Turn 3: still within the 2-turn sticky window.
    defs3, _ = s._turn_tool_defs("vale")
    assert "schedule_task" in _names(defs3)
    # Turn 4: sticky window expired.
    defs4, _ = s._turn_tool_defs("gracias")
    assert "schedule_task" not in _names(defs4)


def test_confirmation_inherits_previous_groups():
    s = _make_session()
    # Turn 1: scheduler active (proposed but not executed → no sticky).
    s._turn_tool_defs("prográmame una tarea el jueves")
    # Turn 2: bare confirmation with no keywords inherits previous active set.
    # ("vale" is a single-word confirmation recognized by _is_user_confirmation;
    # multi-word forms like "sí, hazlo" are not, by existing design.)
    defs, groups = s._turn_tool_defs("vale")
    assert "schedule_task" in _names(defs)
    assert "scheduler" in groups
