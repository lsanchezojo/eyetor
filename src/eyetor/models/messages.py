"""Message models matching OpenAI wire format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FunctionCall(BaseModel):
    """A function call within a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call requested by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """A conversation message in OpenAI-compatible format.

    model_dump(exclude_none=True) produces the exact payload
    for the HTTP POST body's messages array.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For role="tool" responses
    name: str | None = None


class ToolResult(BaseModel):
    """Result of executing a tool call."""

    tool_call_id: str
    content: str
