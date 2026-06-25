"""ChatSession never leaks ``reasoning_content`` as the user-facing reply.

Reasoning is the model's internal scratchpad — it may contain tool-call
drafts or other noise that should not reach the user. If ``content`` is
empty, the session yields ``""`` and channels show their own "no response,
retry" message; we deliberately do NOT fall back to reasoning.
"""

from __future__ import annotations

import asyncio
import json

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


class _RetryThenFinalProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        if tools is None:
            return CompletionResult(message=Message(role="assistant", content="looped"))

        self.calls += 1
        if self.calls == 1:
            args = '{"q":"mkdir tmp directory"}'
        elif self.calls == 2:
            args = '{"q":"megadl path tmp url"}'
        elif self.calls == 3:
            args = '{"q":"megadl path tmp url timeout 300"}'
        else:
            return CompletionResult(message=Message(role="assistant", content="done"))

        return CompletionResult(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call-{self.calls}",
                        function=FunctionCall(name="fake_tool", arguments=args),
                    )
                ],
            )
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_soft_loop_requires_all_recent_calls_to_be_similar():
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
    provider = _RetryThenFinalProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("retry with timeout")) == "done"
    assert provider.calls == 4


class _DuplicateToolCallProvider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            call = ToolCall(
                id="call-1",
                function=FunctionCall(name="fake_tool", arguments='{"q":"same"}'),
            )
            duplicate = ToolCall(
                id="call-2",
                function=FunctionCall(name="fake_tool", arguments='{"q":"same"}'),
            )
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[call, duplicate],
                )
            )
        return CompletionResult(message=Message(role="assistant", content="final"))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_duplicate_tool_calls_in_same_turn_execute_once():
    calls: list[str] = []

    async def fake_tool(q: str) -> str:
        calls.append(q)
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
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=_DuplicateToolCallProvider(),
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("dedupe")) == "final"
    assert calls == ["same"]


class _InvalidThenValidToolProvider:
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
                            id="bad-call",
                            function=FunctionCall(
                                name="fake_tool",
                                arguments=json.dumps(
                                    {
                                        "args": (
                                            "receipt.py add --items '[{\"name\": "
                                            "\"X\", \"price\": 2."
                                        )
                                    }
                                ),
                            ),
                        )
                    ],
                )
            )
        if self.calls == 2:
            assert messages[-1].role == "user"
            assert "arguments que no eran JSON" in (messages[-1].content or "")
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="good-call",
                            function=FunctionCall(
                                name="fake_tool",
                                arguments='{"q":"ok"}',
                            ),
                        )
                    ],
                )
            )
        return CompletionResult(message=Message(role="assistant", content="final"))

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_invalid_tool_call_arguments_are_not_executed_or_remembered():
    calls: list[str] = []

    async def fake_tool(q: str) -> str:
        calls.append(q)
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
    provider = _InvalidThenValidToolProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    assert asyncio.run(session.send_sync("use tool")) == "final"
    assert provider.calls == 3
    assert calls == ["ok"]
    assert all(
        json.loads(tc.function.arguments) is not None
        for msg in session.get_history()
        if msg.tool_calls
        for tc in msg.tool_calls
    )


class _ScaffoldLeakAfterNudgeProvider:
    """Announces a tool, then (after the nudge) replies to the nudge with a
    scaffolding-leak apology. The forced clean-synthesis pass (tools=None)
    returns the real answer."""

    model = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.synth_calls = 0

    async def complete(self, messages, tools=None, temperature=0.0):
        if tools is None:
            self.synth_calls += 1
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="BSEED es una marca de domótica compatible con Tuya.",
                )
            )
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Voy a ejecutar fake_tool ahora.",
                )
            )
        return CompletionResult(
            message=Message(
                role="assistant",
                content=(
                    "Lo siento, tienes razón. He cometido un error en la "
                    "ejecución del plan; no necesito realizar más llamadas a "
                    "herramientas."
                ),
            )
        )

    async def stream(self, messages, tools=None, temperature=0.0):  # pragma: no cover
        raise NotImplementedError


def test_scaffolding_leak_after_nudge_is_replaced_by_clean_synthesis():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="fake_tool",
            description="fake",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",  # type: ignore[arg-type]
        )
    )
    provider = _ScaffoldLeakAfterNudgeProvider()
    session = ChatSession(
        session_id="test",
        config=AgentConfig(name="t", provider="fake", model="fake", max_iterations=5),
        provider=provider,
        tool_registry=registry,
    )

    answer = asyncio.run(session.send_sync("¿qué es BSEED?"))
    # The user gets the clean synthesis, never the meta-apology.
    assert answer == "BSEED es una marca de domótica compatible con Tuya."
    assert provider.synth_calls == 1
    # calls: 1 announce + 1 leak reply (with tools) = 2; synthesis is tools=None.
    assert provider.calls == 2


def test_load_history_drops_invalid_tool_call_and_orphan_tool_result(tmp_path):
    cfg = AgentConfig(name="t", provider="fake", model="fake")
    root_cfg = type(
        "RootCfg",
        (),
        {
            "sessions": type(
                "Sessions",
                (),
                {
                    "persist": True,
                    "dir": str(tmp_path),
                    "max_messages": 200,
                    "tool_gating": type(
                        "Gating", (), {"enabled": False, "sticky_turns": 2}
                    )(),
                    "compaction": type("Compaction", (), {"enabled": False})(),
                },
            )(),
        },
    )()
    path = tmp_path / "test.jsonl"
    rows = [
        Message(role="user", content="hi").model_dump(exclude_none=True),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="bad-call",
                    function=FunctionCall(
                        name="fake_tool",
                        arguments=json.dumps(
                            {
                                "args": (
                                    "receipt.py add --items '[{\"name\": "
                                    "\"X\", \"price\": 2."
                                )
                            }
                        ),
                    ),
                )
            ],
        ).model_dump(exclude_none=True),
        Message(
            role="tool", tool_call_id="bad-call", content='{"error":"bad"}'
        ).model_dump(exclude_none=True),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    session = ChatSession(
        session_id="test",
        config=cfg,
        provider=_FakeProvider("ok", None),
        root_config=root_cfg,  # type: ignore[arg-type]
    )

    assert [m.role for m in session.get_history()] == ["user"]
    rewritten = path.read_text(encoding="utf-8").splitlines()
    assert len(rewritten) == 1
