"""Robust JSON extraction and route fallback tests."""

from __future__ import annotations

from eyetor.utils.json import extract_json_object
from eyetor.workflows.router import Route, _parse_classification


def test_extract_json_object_valid() -> None:
    assert extract_json_object('{"route": "chat"}') == {"route": "chat"}


def test_extract_json_object_markdown() -> None:
    text = "```json\n{\"route\": \"kb_query\"}\n```"
    assert extract_json_object(text) == {"route": "kb_query"}


def test_extract_json_object_embedded() -> None:
    text = "Claro. {\"action\": \"final_answer\", \"content\": \"ok\"} Fin."
    assert extract_json_object(text) == {"action": "final_answer", "content": "ok"}


def test_router_plain_text_route_name() -> None:
    routes = {
        "chat": Route("chat", "small talk", ""),
        "kb_query": Route("kb_query", "knowledge base docs", ""),
    }
    assert _parse_classification("Use kb_query porque menciona docs", routes)[0] == "kb_query"


def test_router_lexical_description_fallback() -> None:
    routes = {
        "chat": Route("chat", "small talk greetings", ""),
        "kb_query": Route("kb_query", "knowledge base documentation manuals", ""),
    }
    assert _parse_classification("Pregunta sobre documentation manuals", routes)[0] == "kb_query"


def test_router_no_recognizable_route() -> None:
    routes = {"chat": Route("chat", "small talk", "")}
    assert _parse_classification("!!!", routes)[0] == ""

