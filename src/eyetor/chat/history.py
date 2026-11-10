"""Helpers to keep the stored conversation history clean."""

from __future__ import annotations

import json
import shlex

from eyetor.models.messages import Message


def _is_ghost_assistant(msg: Message) -> bool:
    """True if ``msg`` is an empty assistant turn that should not be remembered.

    Small local models occasionally collapse mid-turn and produce an
    assistant message with no ``content`` and no ``tool_calls`` (the whole
    output ended up in the reasoning/think channel, or was an immediate
    EOS). Persisting these turns poisons future prompts — the model sees
    "the previous assistant reply was empty" and mimics that pattern.
    """
    if msg.role != "assistant":
        return False
    if msg.tool_calls:
        return False
    return not (msg.content or "").strip()


def _invalid_tool_call_arguments(msg: Message) -> list[str]:
    """Return tool-call validation errors for a message.

    Providers require ``function.arguments`` to be a JSON object string when a
    prior assistant tool call is replayed in history. A truncated argument
    string poisons the whole conversation: the next provider rejects the entire
    request before the model can recover.
    """
    errors: list[str] = []
    if not msg.tool_calls:
        return errors
    for tc in msg.tool_calls:
        raw = tc.function.arguments
        try:
            parsed = json.loads(raw) if raw else {}
        except (TypeError, ValueError) as exc:
            errors.append(f"{tc.function.name}: invalid JSON arguments ({exc})")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{tc.function.name}: arguments must be a JSON object")
            continue
        args = parsed.get("args")
        if isinstance(args, str):
            try:
                shlex.split(args)
            except ValueError as exc:
                errors.append(f"{tc.function.name}: invalid args string ({exc})")
    return errors


def _has_invalid_tool_call_arguments(msg: Message) -> bool:
    return bool(_invalid_tool_call_arguments(msg))


def _clean_loaded_history(messages: list[Message]) -> tuple[list[Message], int]:
    """Drop messages that would make provider payloads invalid on replay."""
    kept: list[Message] = []
    pending_tool_ids: set[str] = set()
    dropped = 0
    for msg in messages:
        if _is_ghost_assistant(msg) or _has_invalid_tool_call_arguments(msg):
            dropped += 1
            continue
        if msg.role == "assistant":
            if msg.tool_calls:
                pending_tool_ids.update(tc.id for tc in msg.tool_calls)
            kept.append(msg)
            continue
        if msg.role == "tool":
            if msg.tool_call_id and msg.tool_call_id in pending_tool_ids:
                pending_tool_ids.remove(msg.tool_call_id)
                kept.append(msg)
            else:
                dropped += 1
            continue
        kept.append(msg)
    return kept, dropped


def _final_text(content: str | None, reasoning: str | None) -> str:
    """Pick the user-facing answer.

    Only ``content`` is user-facing — ``reasoning`` is the model's internal
    scratchpad (``<think>`` channel) and may contain tool-call drafts,
    monologue, or other noise that degrades response quality if shown raw.
    The ``reasoning`` argument is accepted for call-site symmetry but
    deliberately not used as a fallback. Channels show their own
    "no response, retry" message when content is empty.
    """
    del reasoning  # intentionally unused
    return (content or "").strip()


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string for log output, collapsing whitespace."""
    if s is None:
        return ""
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(+{len(s) - max_len})"
