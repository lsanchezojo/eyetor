"""Textual tool-call parsing and execution tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from eyetor.chat.session import ChatSession
from eyetor.channels.telegram import _sanitize_model_text
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import CompletionResult, Message, StreamingResponse
from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.providers.base import BaseProvider
from eyetor.utils.tool_calls import parse_textual_tool_calls


def test_parse_bracket_toolcall_with_args_dict_and_normalized_name() -> None:
    parsed = parse_textual_tool_calls(
        'Voy a ejecutar.\n[toolcall]{"name":"third-party/tool","args":{"x":1}}',
        available_tool_names={"third_party_tool"},
    )
    assert parsed.cleaned_text == "Voy a ejecutar."
    assert parsed.unresolved_names == []
    call = parsed.tool_calls[0]
    assert call.function.name == "third_party_tool"
    assert json.loads(call.function.arguments) == {"x": 1}


def test_parse_bracket_tool_call_with_arguments_list() -> None:
    parsed = parse_textual_tool_calls(
        '[tool_call]{"name":"collector","arguments":["a","b"]}',
        available_tool_names={"collector"},
    )
    assert json.loads(parsed.tool_calls[0].function.arguments) == ["a", "b"]


def test_parse_xml_tool_call() -> None:
    parsed = parse_textual_tool_calls(
        '<tool_call>{"name":"exact_tool","arguments":{"ok":true}}</tool_call>',
        available_tool_names={"exact_tool"},
    )
    assert parsed.tool_calls[0].function.name == "exact_tool"


def test_parse_chatml_tool_call() -> None:
    parsed = parse_textual_tool_calls(
        '<|tool_call|>{"name":"ExactTool","args":{}}<|/tool_call|>',
        available_tool_names={"ExactTool"},
    )
    assert parsed.tool_calls[0].function.name == "ExactTool"


def test_parse_hermes_function_block() -> None:
    parsed = parse_textual_tool_calls(
        "<function=third.tool><parameter=q>hola</parameter></function>",
        available_tool_names={"third_tool"},
    )
    assert parsed.tool_calls[0].function.name == "third_tool"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {"q": "hola"}


def test_case_insensitive_unique_match() -> None:
    parsed = parse_textual_tool_calls(
        '[toolcall]{"name":"MIXED_TOOL","args":{}}',
        available_tool_names={"mixed_tool"},
    )
    assert parsed.tool_calls[0].function.name == "mixed_tool"


def test_compact_unique_match_handles_missing_separator() -> None:
    parsed = parse_textual_tool_calls(
        '[toolcall]{"name":"thirdparty-tool","args":{}}',
        available_tool_names={"third_party_tool"},
    )
    assert parsed.tool_calls[0].function.name == "third_party_tool"


def test_ambiguous_case_insensitive_match_does_not_execute() -> None:
    parsed = parse_textual_tool_calls(
        '[toolcall]{"name":"MIXED_TOOL","args":{}}',
        available_tool_names={"mixed_tool", "Mixed_Tool"},
    )
    assert parsed.tool_calls == []
    assert parsed.ambiguous_names == ["MIXED_TOOL"]
    assert "[toolcall]" not in parsed.cleaned_text


def test_unknown_tool_is_cleaned_but_not_executed() -> None:
    parsed = parse_textual_tool_calls(
        'Antes [toolcall]{"name":"missing","args":{}} despues',
        available_tool_names={"present"},
    )
    assert parsed.tool_calls == []
    assert parsed.unknown_names == ["missing"]
    assert "[toolcall]" not in parsed.cleaned_text


class FakeProvider(BaseProvider):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(base_url="http://local", model="fake")
        self.outputs = outputs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        thinking: bool | None = None,
    ) -> CompletionResult:
        del messages, tools, temperature, thinking
        return CompletionResult(
            message=Message(role="assistant", content=self.outputs.pop(0))
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        raise NotImplementedError


def test_chat_session_executes_valid_textual_tool_call() -> None:
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()

    async def handler(value: int) -> str:
        calls.append({"value": value})
        return json.dumps({"ok": True})

    registry.register(
        ToolDefinition(
            name="third_party_tool",
            description="arbitrary test tool",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
    )
    session = ChatSession(
        "test",
        AgentConfig(name="test", provider="", model="fake", max_iterations=3),
        FakeProvider(
            [
                'Voy.\n[toolcall]{"name":"third-party-tool","args":{"value":7}}',
                "Hecho.",
            ]
        ),
        tool_registry=registry,
    )

    async def run() -> str:
        chunks = []
        async for chunk in session.send("hazlo"):
            chunks.append(chunk)
        return "".join(chunks)

    assert asyncio.run(run()) == "Hecho."
    assert calls == [{"value": 7}]


def test_chat_session_cleans_unknown_textual_tool_call() -> None:
    session = ChatSession(
        "test",
        AgentConfig(name="test", provider="", model="fake", max_iterations=1),
        FakeProvider(['[toolcall]{"name":"missing","args":{}}']),
        tool_registry=ToolRegistry(),
    )

    async def run() -> str:
        chunks = []
        async for chunk in session.send("hazlo"):
            chunks.append(chunk)
        return "".join(chunks)

    output = asyncio.run(run())
    assert "[toolcall]" not in output
    assert "missing" not in output


def test_telegram_sanitizer_removes_textual_tool_call_markup() -> None:
    text = 'Texto [toolcall]{"name":"missing","args":{}}'
    assert _sanitize_model_text(text) == "Texto"
