"""Guard against claimed-but-unexecuted smart-home actions.

Incident 2026-10-30: asked to "apaga la luz del salón", the model read the
state with GetLiveContext and replied "He apagado la luz del salón (Luz Sofa)"
without ever calling HassTurnOff, so the light stayed on.
"""

from __future__ import annotations

import asyncio

from eyetor.chat.guards import _HOME_ACTION_CLAIM_RE, _is_home_action_tool
from eyetor.chat.session import ChatSession
from eyetor.config import VectorConfig
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, FunctionCall, Message, ToolCall
from eyetor.models.tools import ToolDefinition, ToolRegistry


def _claims(text: str) -> bool:
    return bool(_HOME_ACTION_CLAIM_RE.search(text))


def test_claim_regex():
    assert _claims("He apagado la luz del salón (Luz Sofa).")
    assert _claims("Hemos encendido el flexo.")
    assert _claims("He también apagado la luz de la cocina.")
    assert not _claims("La luz del sofá ya está apagada.")
    assert not _claims("En el salón hace 24 °C.")
    assert not _claims("No he podido encender el flexo.")


def test_is_home_action_tool():
    def tool(name, group="home"):
        return ToolDefinition(name=name, description="", parameters={}, group=group)

    assert _is_home_action_tool(tool("intent__HassTurnOff"))
    assert not _is_home_action_tool(tool("homeassistant__GetLiveContext"))
    assert not _is_home_action_tool(tool("llm__GetDateTime"))
    assert not _is_home_action_tool(tool("skill_shell", group=None))
    assert not _is_home_action_tool(None)


class _ScriptedProvider:
    model = "fake"

    def __init__(self, replies: list[Message]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append(list(messages))
        return CompletionResult(message=self.replies.pop(0))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def _call(name: str, args: str = "{}") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=name, function=FunctionCall(name=name, arguments=args))],
    )


def _session(provider) -> tuple[ChatSession, list[str]]:
    executed: list[str] = []
    registry = ToolRegistry()

    def make(name: str):
        async def handler(**_):
            executed.append(name)
            return '{"response_type": "action_done"}'

        registry.register(
            ToolDefinition(
                name=name, description="", parameters={"type": "object"},
                handler=handler, group="home",
            )
        )

    make("homeassistant__GetLiveContext")
    make("intent__HassTurnOff")
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake"),
        provider=provider,
        tool_registry=registry,
        root_config=VectorConfig(),
    )
    return session, executed


def test_claim_without_action_is_nudged_into_the_call():
    provider = _ScriptedProvider([
        _call("homeassistant__GetLiveContext"),
        Message(role="assistant", content="He apagado la luz del salón."),
        _call("intent__HassTurnOff", '{"name": "Luz Sofa"}'),
        Message(role="assistant", content="He apagado la luz del salón."),
    ])
    session, executed = _session(provider)

    reply = asyncio.run(session.send_sync("apaga la luz del salón"))

    assert executed == ["homeassistant__GetLiveContext", "intent__HassTurnOff"]
    assert reply == "He apagado la luz del salón."


def test_persistent_false_claim_is_flagged():
    provider = _ScriptedProvider([
        Message(role="assistant", content="He apagado la luz del salón."),
        Message(role="assistant", content="He apagado la luz del salón."),
    ])
    session, executed = _session(provider)

    reply = asyncio.run(session.send_sync("apaga la luz del salón"))

    assert executed == []
    assert "no se ha llegado a ejecutar" in reply
