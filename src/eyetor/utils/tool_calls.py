"""Helpers for textual tool-call markup emitted by small local models."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from eyetor.models.messages import FunctionCall, ToolCall
from eyetor.models.tools import ToolDefinition

_BRACKET_MARKER_RE = re.compile(r"\[(?:toolcall|tool_call)\]\s*", re.IGNORECASE)
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call\b[^>]*>(.*?)</tool_call>"
    r"|<\|tool_call\|>(.*?)<\|/tool_call\|>",
    re.DOTALL | re.IGNORECASE,
)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=([^>\s]+)\s*>(.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_NAME_SEPARATORS_RE = re.compile(r"[\s/\-.:]+")


@dataclass
class TextualToolCallParseResult:
    """Result of parsing textual tool-call markup."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    cleaned_text: str = ""
    had_markup: bool = False
    unknown_names: list[str] = field(default_factory=list)
    ambiguous_names: list[str] = field(default_factory=list)

    @property
    def unresolved_names(self) -> list[str]:
        return self.unknown_names + self.ambiguous_names


def offered_tool_names(tools: Iterable[ToolDefinition] | None) -> set[str]:
    """Return the offered tool names for a provider/session call."""
    return {tool.name for tool in tools or []}


def strip_textual_tool_calls(text: str) -> tuple[str, bool]:
    """Remove textual tool-call markup without trying to resolve or execute it."""
    result = parse_textual_tool_calls(text, available_tool_names=None)
    return result.cleaned_text, result.had_markup


def parse_textual_tool_calls(
    text: str,
    *,
    available_tool_names: Iterable[str] | None,
) -> TextualToolCallParseResult:
    """Extract textual tool calls and resolve them against registered tools.

    If ``available_tool_names`` is provided, only calls resolving to that set are
    returned. Unknown or ambiguous names are reported but never guessed.
    """
    result = TextualToolCallParseResult(cleaned_text=text or "")
    if not text:
        return result

    raw_calls, spans = _extract_raw_calls(text)
    if not spans:
        return result

    result.had_markup = True
    names = set(available_tool_names) if available_tool_names is not None else None
    for raw_name, raw_args in raw_calls:
        resolved = _resolve_tool_name(raw_name, names)
        if resolved is None:
            result.unknown_names.append(raw_name)
            continue
        if resolved == "":
            result.ambiguous_names.append(raw_name)
            continue
        result.tool_calls.append(
            ToolCall(
                id=uuid.uuid4().hex[:24],
                function=FunctionCall(
                    name=resolved,
                    arguments=_arguments_to_json(raw_args),
                ),
            )
        )

    result.cleaned_text = _strip_spans(text, spans)
    return result


def _extract_raw_calls(text: str) -> tuple[list[tuple[str, Any]], list[tuple[int, int]]]:
    calls: list[tuple[str, Any]] = []
    spans: list[tuple[int, int]] = []
    decoder = json.JSONDecoder()

    for match in _BRACKET_MARKER_RE.finditer(text):
        try:
            obj, end = decoder.raw_decode(text[match.end() :].lstrip())
        except json.JSONDecodeError:
            continue
        tail = text[match.end() :]
        start_offset = match.end() + (len(tail) - len(tail.lstrip()))
        if isinstance(obj, dict):
            maybe = _call_from_mapping(obj)
            if maybe:
                calls.append(maybe)
                spans.append((match.start(), start_offset + end))

    for match in _XML_TOOL_CALL_RE.finditer(text):
        payload = (match.group(1) or match.group(2) or "").strip()
        spans.append(match.span())
        parsed = False
        if payload.startswith("{"):
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                obj = None
            if isinstance(obj, dict):
                maybe = _call_from_mapping(obj)
                if maybe:
                    calls.append(maybe)
                    parsed = True
        if not parsed:
            calls.extend(_calls_from_function_blocks(payload))

    for match in _FUNCTION_BLOCK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        extracted = _call_from_function_match(match)
        if extracted:
            calls.append(extracted)
            spans.append(match.span())

    return calls, _merge_spans(spans)


def _call_from_mapping(obj: dict[str, Any]) -> tuple[str, Any] | None:
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None
    if "arguments" in obj:
        args = obj["arguments"]
    elif "args" in obj:
        args = obj["args"]
    else:
        args = {}
    return name.strip(), args


def _calls_from_function_blocks(payload: str) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    for match in _FUNCTION_BLOCK_RE.finditer(payload):
        extracted = _call_from_function_match(match)
        if extracted:
            calls.append(extracted)
    return calls


def _call_from_function_match(match: re.Match) -> tuple[str, Any] | None:
    name = match.group(1).strip()
    if not name:
        return None
    params = {
        pm.group(1).strip(): pm.group(2).strip()
        for pm in _PARAMETER_BLOCK_RE.finditer(match.group(2))
    }
    return name, params


def _arguments_to_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return ""
    return json.dumps(arguments, ensure_ascii=False)


def _resolve_tool_name(raw_name: str, available: set[str] | None) -> str | None:
    if available is None:
        return raw_name
    if raw_name in available:
        return raw_name

    normalized = _normalize_tool_name(raw_name)
    if normalized in available:
        return normalized

    lower_matches = [name for name in available if name.lower() == raw_name.lower()]
    if len(lower_matches) == 1:
        return lower_matches[0]
    if len(lower_matches) > 1:
        return ""

    normalized_lower_matches = [
        name for name in available if name.lower() == normalized.lower()
    ]
    if len(normalized_lower_matches) == 1:
        return normalized_lower_matches[0]
    if len(normalized_lower_matches) > 1:
        return ""

    compact = _compact_tool_name(raw_name)
    compact_matches = [
        name for name in available if _compact_tool_name(name) == compact
    ]
    if len(compact_matches) == 1:
        return compact_matches[0]
    if len(compact_matches) > 1:
        return ""

    return None


def _normalize_tool_name(name: str) -> str:
    return _NAME_SEPARATORS_RE.sub("_", name.strip()).strip("_")


def _compact_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    cleaned = text
    for start, end in sorted(spans, key=lambda span: span[0], reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged
