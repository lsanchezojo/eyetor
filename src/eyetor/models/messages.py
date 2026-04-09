"""Message models matching OpenAI wire format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

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


@dataclass
class TokenUsage:
    """Token counts from an LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass
class StreamingResponse:
    """Wrapper around an async text stream with attached usage metadata.

    Allows callers to consume tokens via `async for` while also accessing
    usage data after the stream is exhausted via the `.usage` property.
    """

    _iterator: AsyncIterator[str] = field(repr=False)
    _usage: TokenUsage | None = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._iterator.__anext__()

    @property
    def usage(self) -> TokenUsage | None:
        """Usage metadata extracted from the final SSE chunk (if available)."""
        return self._usage


@dataclass
class CompletionResult:
    """Wraps a Message with API metadata (usage, model, finish reason).

    This is the return type of BaseProvider.complete(). The Message itself
    stays clean (wire format only); metadata lives here.
    """

    message: Message
    usage: TokenUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
