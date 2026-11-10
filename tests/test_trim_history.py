"""ChatSession.trim_history: idle-time shrinking of long-lived (voice) sessions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from eyetor.chat.session import ChatSession
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall


def _session(tmp_path, turns: int, tool_chars: int = 1000) -> ChatSession:
    cfg = AgentConfig(name="x", provider="p", model="m", system_prompt="")
    s = ChatSession(session_id="voice-test", config=cfg, provider=MagicMock())
    s._persist_path = tmp_path / "voice-test.jsonl"
    for n in range(turns):
        s._messages += [
            Message(role="user", content=f"pregunta {n}"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id=f"c{n}", type="function", function={"name": "t", "arguments": "{}"})],
            ),
            Message(role="tool", tool_call_id=f"c{n}", content=f"estado {n} " + "x" * tool_chars),
            Message(role="assistant", content=f"respuesta {n}"),
        ]
    s._rewrite_persisted()
    return s


def _stored(s: ChatSession) -> list[dict]:
    return [json.loads(line) for line in s._persist_path.read_text().splitlines()]


def test_stubs_old_tool_results_but_keeps_replies(tmp_path):
    s = _session(tmp_path, turns=6)
    stats = s.trim_history(max_user_turns=20, keep_user_turns=12, stale_tool_turns=3)
    assert stats["dropped_messages"] == 0 and stats["stubbed_tool_results"] == 3
    tools = [m for m in s._messages if m.role == "tool"]
    assert all('"omitted"' in m.content for m in tools[:3])
    assert all(m.content.startswith("estado") for m in tools[3:])  # last 3 turns intact
    assert [m.content for m in s._messages if m.role == "assistant" and m.content][:1] == ["respuesta 0"]
    assert _stored(s) == [m.model_dump(exclude_none=True) for m in s._messages]
    # Idempotent: a second pass changes nothing (no needless cache invalidation).
    assert s.trim_history(max_user_turns=20, keep_user_turns=12, stale_tool_turns=3) is None


def test_archives_old_turns_only_past_the_high_mark(tmp_path):
    s = _session(tmp_path, turns=20)
    assert s.trim_history(max_user_turns=20, keep_user_turns=12, stale_tool_turns=0) is None
    s._messages.append(Message(role="user", content="pregunta 20"))
    stats = s.trim_history(max_user_turns=20, keep_user_turns=12, stale_tool_turns=0)
    assert stats["dropped_messages"] == 9 * 4
    assert s._messages[0].role == "user" and s._messages[0].content == "pregunta 9"
    assert sum(m.role == "user" for m in s._messages) == 12
    archived = [json.loads(x) for x in open(stats["archived"])]
    assert archived[0]["content"] == "pregunta 0" and len(archived) == 36


def test_small_tool_results_are_kept(tmp_path):
    s = _session(tmp_path, turns=6, tool_chars=50)
    assert s.trim_history(max_user_turns=20, keep_user_turns=12, stale_tool_turns=3) is None
