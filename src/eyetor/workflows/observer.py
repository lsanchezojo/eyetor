"""WorkerObserver — collects events from a worker session for observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class WorkerEvent:
    """A single event emitted during a worker's execution."""

    type: Literal["iteration", "tool_start", "tool_end", "tool_error", "llm_response", "done"]
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)


class WorkerObserver:
    """Collects events from a ChatSession or ToolAgent execution.

    Attach to a session via ``observer=`` kwarg. The session emits events
    at key points in the agentic loop; this class records them for later
    inspection.

    Usage:
        observer = WorkerObserver()
        session = ChatSession(..., observer=observer)
        await session.send_sync("do something")
        print(observer.get_summary())
    """

    def __init__(self) -> None:
        self._events: list[WorkerEvent] = []
        self._done = False

    # -- Event emitters (called by the session loop) --

    def on_iteration(self, n: int) -> None:
        self._events.append(WorkerEvent(
            type="iteration", timestamp=datetime.now(), data={"n": n},
        ))

    def on_tool_start(self, tool_name: str, args: str) -> None:
        self._events.append(WorkerEvent(
            type="tool_start", timestamp=datetime.now(),
            data={"tool": tool_name, "args": args[:500]},
        ))

    def on_tool_end(self, tool_name: str, result: str) -> None:
        self._events.append(WorkerEvent(
            type="tool_end", timestamp=datetime.now(),
            data={"tool": tool_name, "result_len": len(result)},
        ))

    def on_tool_error(self, tool_name: str, error: str) -> None:
        self._events.append(WorkerEvent(
            type="tool_error", timestamp=datetime.now(),
            data={"tool": tool_name, "error": error[:500]},
        ))

    def on_llm_response(self, content: str, tool_calls: list) -> None:
        self._events.append(WorkerEvent(
            type="llm_response", timestamp=datetime.now(),
            data={
                "content_len": len(content or ""),
                "tool_calls": [tc.function.name for tc in (tool_calls or [])],
            },
        ))

    def on_done(self, final_output: str) -> None:
        self._done = True
        self._events.append(WorkerEvent(
            type="done", timestamp=datetime.now(),
            data={"output_len": len(final_output)},
        ))

    # -- Queries --

    def get_events(self) -> list[WorkerEvent]:
        return list(self._events)

    def is_done(self) -> bool:
        return self._done

    def last_activity(self) -> datetime | None:
        return self._events[-1].timestamp if self._events else None

    def get_summary(self) -> str:
        """Return a human-readable summary of what happened."""
        iterations = sum(1 for e in self._events if e.type == "iteration")
        tools_called = [e.data["tool"] for e in self._events if e.type == "tool_start"]
        errors = [e.data for e in self._events if e.type == "tool_error"]
        done_event = next((e for e in self._events if e.type == "done"), None)

        parts = [f"{iterations} iterations"]
        if tools_called:
            parts.append(f"tools: {', '.join(tools_called)}")
        if errors:
            parts.append(f"{len(errors)} errors")
        if done_event:
            parts.append(f"output: {done_event.data['output_len']} chars")
        else:
            parts.append("not finished")
        return " | ".join(parts)
