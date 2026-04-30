"""Provider parsing and payload behavior for local OpenAI-compatible servers."""

from __future__ import annotations

import json

from eyetor.models.messages import Message
from eyetor.providers.llamacpp import LlamaCppProvider
from eyetor.providers.ollama import OllamaProvider
from eyetor.providers.openrouter import _parse_completion_response


def _response(tool_calls):
    return {
        "model": "local",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def test_parse_tool_call_without_id_generates_id() -> None:
    result = _parse_completion_response(
        _response(
            [
                {
                    "type": "function",
                    "function": {"name": "kb_search", "arguments": "{}"},
                }
            ]
        )
    )
    call = result.message.tool_calls[0]
    assert len(call.id) == 24
    assert call.function.name == "kb_search"


def test_parse_tool_call_arguments_dict_serialized() -> None:
    result = _parse_completion_response(
        _response(
            [
                {
                    "id": "call_1",
                    "function": {"name": "kb_search", "arguments": {"query": "hola"}},
                }
            ]
        )
    )
    assert json.loads(result.message.tool_calls[0].function.arguments) == {"query": "hola"}


def test_parse_malformed_tool_call_ignored() -> None:
    result = _parse_completion_response(
        _response(
            [
                {"id": "bad", "function": {"arguments": "{}"}},
                {"id": "ok", "function": {"name": "done", "arguments": "{}"}},
            ]
        )
    )
    assert [tc.id for tc in result.message.tool_calls] == ["ok"]


def test_parse_normal_tool_call_unchanged() -> None:
    result = _parse_completion_response(
        _response(
            [
                {
                    "id": "call_abc",
                    "function": {"name": "tool", "arguments": "{\"x\": 1}"},
                }
            ]
        )
    )
    call = result.message.tool_calls[0]
    assert call.id == "call_abc"
    assert call.function.arguments == "{\"x\": 1}"


def test_llamacpp_payload_merges_extra_body_without_overwriting_core() -> None:
    provider = LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        model="m",
        top_p=0.9,
        extra_body={"grammar": "root ::= \"ok\"", "model": "wrong"},
    )
    payload = provider._build_payload(
        [Message(role="user", content="hola")],
        tools=None,
        temperature=0.1,
    )
    assert payload["model"] == "m"
    assert payload["top_p"] == 0.9
    assert payload["grammar"] == 'root ::= "ok"'


def test_ollama_payload_includes_options_when_set() -> None:
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        model="m",
        options={"num_ctx": 4096, "repeat_penalty": 1.1},
    )
    payload = provider._build_payload(
        [Message(role="user", content="hola")],
        tools=None,
        temperature=0.1,
    )
    assert payload["options"] == {"num_ctx": 4096, "repeat_penalty": 1.1}

