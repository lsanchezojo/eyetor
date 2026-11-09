"""Recovery paths of ChatSession.send() and chain mode, driven by scripted providers."""

from __future__ import annotations

import asyncio

from eyetor.chat.session import ChatSession
from eyetor.config import VectorConfig
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, FunctionCall, Message, ToolCall
from eyetor.models.tools import ToolDefinition, ToolRegistry


def _call(i: int, q: str = "x") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id=f"c{i}", function=FunctionCall(name="fake_tool", arguments=f'{{"q":"{q}"}}'))
        ],
    )


def _text(content: str) -> Message:
    return Message(role="assistant", content=content)


class _Scripted:
    """Returns the scripted messages in order and records each request."""

    model = "fake"

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.requests: list[tuple[bool, str]] = []  # (tools offered, last message text)

    async def complete(self, messages, tools=None, temperature=0.0):
        self.requests.append((tools is not None, messages[-1].content or ""))
        return CompletionResult(message=self._replies.pop(0))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


class _Observer:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_iteration(self, n):
        self.events.append(f"iter{n}")

    def on_llm_response(self, content, tool_calls):
        pass

    def on_tool_start(self, name, args):
        self.events.append(f"start:{name}")

    def on_tool_end(self, name, result):
        self.events.append(f"end:{name}")

    def on_tool_error(self, name, err):
        self.events.append("error")

    def on_done(self, content):
        self.events.append("done")


def _session(provider, *, root_config=None, observer=None) -> ChatSession:
    async def fake_tool(q: str) -> str:
        return f"resultado {q}"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=fake_tool,
        )
    )
    cfg = AgentConfig(name="t", provider="fake", model="fake", max_iterations=6)
    session = ChatSession(
        session_id="test", config=cfg, provider=provider, tool_registry=registry,
        root_config=root_config,
    )
    if observer is not None:
        session._observer = observer
    return session


def test_post_tool_empty_reply_forces_one_synthesis_without_tools():
    provider = _Scripted(_call(1), _text(""), _text("síntesis final"))
    obs = _Observer()
    reply = asyncio.run(_session(provider, observer=obs).send_sync("busca x"))
    assert reply == "síntesis final"
    offered, last = provider.requests[-1]
    assert offered is False and last.startswith("Sintetiza ahora")
    assert obs.events == ["iter1", "start:fake_tool", "end:fake_tool", "iter2", "done"]


def test_scaffolding_leak_after_nudge_forces_clean_answer():
    provider = _Scripted(
        _text("Voy a buscarlo ahora."),
        _text("Mis disculpas, no emití la tool_call correctamente."),
        _text("respuesta limpia"),
    )
    reply = asyncio.run(_session(provider).send_sync("dime x"))
    assert reply == "respuesta limpia"
    assert provider.requests[1][1].startswith("Recordatorio del sistema")
    offered, last = provider.requests[2]
    assert offered is False and last.startswith("Responde directamente")


def test_chain_mode_plans_executes_tools_and_synthesizes():
    root = VectorConfig()
    root.sessions.persist = False
    root.sessions.chain.mode = "always"
    root.sessions.compaction.enabled = False
    provider = _Scripted(_text("plan: usar fake_tool"), _call(1, "a"), _text("hecho"), _text("resumen"))
    session = _session(provider, root_config=root)
    reply = asyncio.run(session.send_sync("haz a y luego b"))
    assert reply == "resumen"
    assert [offered for offered, _ in provider.requests] == [False, True, True, False]
    tool_msgs = [m for m in session.get_history() if m.role == "tool"]
    assert [m.content for m in tool_msgs] == ["resultado a"]
