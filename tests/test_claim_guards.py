"""Claims of saving or scheduling something that no tool actually did.

Both Bolt and Gemma sometimes reply "¡Apuntado!" or "te avisaré a las 8"
without calling remember / schedule_task, so nothing is stored.
"""

from __future__ import annotations

import asyncio

from eyetor.chat.session import ChatSession, _MEMORY_CLAIM_RE, _REMINDER_CLAIM_RE
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, FunctionCall, Message, ToolCall
from eyetor.models.tools import ToolDefinition, ToolRegistry


def test_memory_claim_regex():
    yes = ["¡Apuntado! Tu color favorito es el azul.", "Lo he guardado en memoria.",
           "Vale, lo recordaré.", "Tomo nota.", "Queda anotado.", "He guardado tu dirección."]
    no = ["No he guardado nada todavía.", "Te lo recordaré mañana.",
          "Tu color favorito es el azul.", "Lo tengo guardado: azul."]
    assert all(_MEMORY_CLAIM_RE.search(t) for t in yes)
    assert not any(_MEMORY_CLAIM_RE.search(t) for t in no)


def test_reminder_claim_regex():
    yes = ["Te avisaré a las 8.", "Perfecto, te lo recordaré mañana.",
           "He programado un recordatorio para las 20:00.", "Recordatorio creado."]
    no = ["Es la hora: saca la basura.", "No he programado nada.", "¿A qué hora te aviso?"]
    assert all(_REMINDER_CLAIM_RE.search(t) for t in yes)
    assert not any(_REMINDER_CLAIM_RE.search(t) for t in no)


class _Scripted:
    model = "fake"

    def __init__(self, *replies: Message) -> None:
        self.replies = list(replies)
        self.last: list[str] = []

    async def complete(self, messages, tools=None, temperature=0.0):
        self.last.append(messages[-1].content or "")
        return CompletionResult(message=self.replies.pop(0))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def _call(name: str, args: str = "{}") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=name, function=FunctionCall(name=name, arguments=args))],
    )


def _text(content: str) -> Message:
    return Message(role="assistant", content=content)


def _session(provider, results: dict[str, str] | None = None) -> tuple[ChatSession, list[str]]:
    executed: list[str] = []
    registry = ToolRegistry()
    for name in ("remember", "schedule_task", "fs_write"):
        async def handler(_name=name, **_):
            executed.append(_name)
            return (results or {}).get(_name, '{"ok": true}')

        registry.register(
            ToolDefinition(name=name, description="", parameters={"type": "object"}, handler=handler)
        )
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake"),
        provider=provider,
        tool_registry=registry,
    )
    return session, executed


def test_memory_claim_is_nudged_into_remember():
    provider = _Scripted(
        _text("¡Apuntado! Tu color favorito es el azul."),
        _call("remember", '{"key": "color", "value": "azul"}'),
        _text("¡Apuntado! Tu color favorito es el azul."),
    )
    session, executed = _session(provider)
    reply = asyncio.run(session.send_sync("mi color favorito es el azul"))
    assert executed == ["remember"]
    assert reply == "¡Apuntado! Tu color favorito es el azul."
    assert provider.last[1].startswith("Recordatorio del sistema: has dicho que guardaste")


def test_persistent_reminder_claim_is_flagged():
    provider = _Scripted(_text("Te avisaré a las 8."), _text("Te avisaré a las 8."))
    session, executed = _session(provider)
    reply = asyncio.run(session.send_sync("avísame a las 8"))
    assert executed == []
    assert "no se ha llegado a programar" in reply


def test_failed_schedule_does_not_back_the_claim():
    provider = _Scripted(
        _call("schedule_task", '{"schedule": "mañana"}'),
        _text("Te avisaré mañana."),
        _text("No he podido programarlo: dime la hora exacta."),
    )
    session, _ = _session(provider, {"schedule_task": '{"ok": false, "error": "bad schedule"}'})
    reply = asyncio.run(session.send_sync("avísame mañana"))
    assert reply == "No he podido programarlo: dime la hora exacta."


def test_saved_file_backs_a_memory_like_claim():
    provider = _Scripted(_call("fs_write", '{"path": "n.txt", "content": "x"}'), _text("He guardado el fichero."))
    session, executed = _session(provider)
    reply = asyncio.run(session.send_sync("guarda esto en n.txt"))
    assert executed == ["fs_write"]
    assert reply == "He guardado el fichero."
