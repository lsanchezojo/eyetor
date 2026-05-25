"""ChatSession never leaks ``reasoning_content`` as the user-facing reply.

Reasoning is the model's internal scratchpad — it may contain tool-call
drafts or other noise that should not reach the user. If ``content`` is
empty, the session yields ``""`` and channels show their own "no response,
retry" message; we deliberately do NOT fall back to reasoning.
"""

from __future__ import annotations

import asyncio

from eyetor.chat.session import ChatSession
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, FunctionCall, Message, ToolCall
from eyetor.models.tools import ToolDefinition, ToolRegistry


class _FakeProvider:
    """Returns one final (no tool_calls) message with the given content."""

    model = "fake"

    def __init__(self, content: str, reasoning: str | None) -> None:
        self._content = content
        self._reasoning = reasoning

    async def complete(self, messages, tools=None, temperature=0.0):
        return CompletionResult(
            message=Message(role="assistant", content=self._content),
            reasoning_content=self._reasoning,
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def _send(provider) -> str:
    cfg = AgentConfig(name="t", provider="fake", model="fake")
    session = ChatSession(session_id="test", config=cfg, provider=provider)
    return asyncio.run(session.send_sync("hi"))


def test_empty_content_does_not_leak_reasoning():
    assert _send(_FakeProvider("", "hola desde reasoning")) == ""


def test_content_preferred_over_reasoning():
    assert _send(_FakeProvider("respuesta normal", "ruido")) == "respuesta normal"


class _LoopProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        if tools is None:
            return CompletionResult(
                message=Message(role="assistant", content=""),
                reasoning_content="internal loop-break reasoning",
            )
        return CompletionResult(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call-{self.calls}",
                        function=FunctionCall(
                            name="fake_tool",
                            arguments='{"q":"same"}',
                        ),
                    )
                ],
            )
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_empty_loop_break_response_is_not_remembered_as_assistant_turn():
    async def fake_tool(q: str) -> str:
        return f"tool saw {q}"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=fake_tool,
        )
    )
    cfg = AgentConfig(
        name="t",
        provider="fake",
        model="fake",
        max_iterations=5,
    )
    session = ChatSession(
        session_id="test",
        config=cfg,
        provider=_LoopProvider(),
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("loop please")) == ""
    ghosts = [
        msg
        for msg in session.get_history()
        if msg.role == "assistant"
        and not (msg.content or "").strip()
        and not msg.tool_calls
    ]
    assert ghosts == []


class _PostToolMentionProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="fake_tool",
                                arguments='{"q":"done"}',
                            ),
                        )
                    ],
                )
            )
        if self.calls == 2:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="He ejecutado fake_tool y el resultado es done.",
                )
            )
        raise AssertionError("unexpected extra LLM call")

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_post_tool_answer_mentioning_tool_name_does_not_trigger_nudge():
    async def fake_tool(q: str) -> str:
        return q

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=fake_tool,
        )
    )
    provider = _PostToolMentionProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    assert (
        asyncio.run(session.send_sync("use tool"))
        == "He ejecutado fake_tool y el resultado es done."
    )
    assert provider.calls == 2


class _AnnounceWithoutCallProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Voy a ejecutar fake_tool ahora.",
                )
            )
        return CompletionResult(message=Message(role="assistant", content="final"))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_future_tool_intent_before_any_tool_still_gets_one_nudge():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",  # type: ignore[arg-type]
        )
    )
    provider = _AnnounceWithoutCallProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("use tool")) == "final"
    assert provider.calls == 2


class _EmptyFirstPassProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                message=Message(role="assistant", content=""),
                reasoning_content="internal",
            )
        return CompletionResult(message=Message(role="assistant", content="recovered"))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_empty_first_pass_with_tools_gets_one_nudge():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",  # type: ignore[arg-type]
        )
    )
    provider = _EmptyFirstPassProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("use tool")) == "recovered"
    assert provider.calls == 2
