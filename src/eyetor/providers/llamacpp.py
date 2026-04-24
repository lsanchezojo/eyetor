"""llama.cpp server LLM provider (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

from eyetor.models.messages import CompletionResult, FunctionCall, Message, StreamingResponse, ToolCall
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider, ContextOverflowError
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, parse_sse

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseProvider):
    """Provider adapter for llama.cpp server's OpenAI-compatible API.

    Authentication is optional — only used if api_key is set.

    When ``thinking=True``, each request includes ``chat_template_kwargs``
    with ``enable_thinking: true``, which activates the reasoning channel on
    models that support it (e.g. Gemma-4, QwQ).  The ``<think>`` block is
    stripped from the visible response but logged at DEBUG level.
    """

    def __init__(
        self,
        *args: Any,
        thinking: bool = False,
        request_timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        stream: bool = False,
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, tools, temperature, stream)
        use_thinking = self.thinking if thinking is None else thinking
        payload["chat_template_kwargs"] = {"enable_thinking": bool(use_thinking)}
        return payload

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        thinking: bool | None = None,
    ) -> CompletionResult:
        use_thinking = self.thinking if thinking is None else thinking
        payload = self._build_payload(
            messages, tools, temperature, stream=False, thinking=use_thinking
        )
        async with self._client(timeout=self.request_timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
            if response.status_code >= 400:
                body = response.text[:500]
                logger.error(
                    "llama.cpp %d error: %s",
                    response.status_code,
                    body,
                )
                if response.status_code == 400 and "exceed_context_size" in body:
                    n_prompt = n_ctx = None
                    try:
                        err = response.json().get("error", {})
                        n_prompt = err.get("n_prompt_tokens")
                        n_ctx = err.get("n_ctx")
                    except Exception:
                        pass
                    raise ContextOverflowError(
                        f"llama.cpp context overflow: {n_prompt} > {n_ctx}",
                        n_prompt_tokens=n_prompt,
                        n_ctx=n_ctx,
                    )
            response.raise_for_status()
            data = response.json()
            result = _parse_completion_response(data)
            if use_thinking:
                reasoning = _extract_reasoning(data)
                if reasoning:
                    result.reasoning_content = reasoning
                    logger.debug("llama.cpp reasoning:\n%s", reasoning.strip())
            # Only recover textual tool_calls if tools were actually offered.
            # When tools is None (e.g. forced-final-answer after loop detection),
            # any <tool_call> markup the model emits is noise — promoting it to
            # a structured call would wipe msg.content and leave the session
            # with an empty response.
            if tools:
                _recover_textual_tool_calls(result)
            return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        payload = self._build_payload(messages, tools, temperature, stream=True)
        sr = StreamingResponse(iter([]), None)  # placeholder, replaced below
        reasoning_parts: list[str] = []

        async def _stream_tokens() -> AsyncIterator[str]:
            async with self._client(timeout=self.request_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    response.raise_for_status()
                    async for chunk in parse_sse(response):
                        if self.thinking:
                            r_token = _extract_reasoning_delta(chunk)
                            if r_token:
                                reasoning_parts.append(r_token)
                        text = extract_delta_content(chunk)
                        if text:
                            yield text
            # After stream exhaustion, populate reasoning on the response object
            if reasoning_parts:
                sr.reasoning_content = "".join(reasoning_parts)

        sr._iterator = _stream_tokens()
        return sr


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_reasoning(data: dict[str, Any]) -> str | None:
    """Extract reasoning_content from a non-streaming response."""
    try:
        return data["choices"][0]["message"].get("reasoning_content") or None
    except (KeyError, IndexError):
        return None


def _extract_reasoning_delta(chunk: dict[str, Any]) -> str | None:
    """Extract a reasoning_content delta token from a streaming chunk."""
    try:
        return chunk["choices"][0]["delta"].get("reasoning_content") or None
    except (KeyError, IndexError):
        return None


# Hermes/Qwen/chatml variants that small local models sometimes emit as plain
# text (inside <think> or leaked to content) instead of as structured tool_calls.
# Pattern captures the inner payload so we can parse it.
_TEXTUAL_TOOL_CALL_RE = re.compile(
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


def _parse_textual_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls emitted as plain-text markup (Hermes/chatml XML or
    JSON) and return a list of structured ``ToolCall``\\ s plus the original
    text with those blocks stripped.

    Handles three shapes the local models actually emit:

    * ``<tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>``
    * ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
    * Bare ``<function=NAME>...</function>`` without the outer wrapper.
    """
    if not text:
        return [], text
    lowered = text.lower()
    if (
        "<tool_call" not in lowered
        and "<|tool_call|>" not in lowered
        and "<function=" not in lowered
    ):
        return [], text

    calls: list[ToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _record(name: str, arguments: dict[str, Any] | str) -> None:
        args_json = (
            arguments if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False)
        )
        calls.append(
            ToolCall(
                id=uuid.uuid4().hex[:24],
                function=FunctionCall(name=name, arguments=args_json),
            )
        )

    # 1) <tool_call>...</tool_call> wrapper (with JSON or nested <function=>).
    for m in _TEXTUAL_TOOL_CALL_RE.finditer(text):
        payload = (m.group(1) or m.group(2) or "").strip()
        consumed_spans.append(m.span())
        # Prefer JSON shape; fall back to nested function/parameter blocks.
        parsed_as_json = False
        if payload.startswith("{"):
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict) and "name" in obj:
                    _record(obj["name"], obj.get("arguments", {}))
                    parsed_as_json = True
            except (ValueError, TypeError):
                pass
        if not parsed_as_json:
            for fm in _FUNCTION_BLOCK_RE.finditer(payload):
                name = fm.group(1).strip()
                inner = fm.group(2)
                params = {
                    pm.group(1).strip(): pm.group(2).strip()
                    for pm in _PARAMETER_BLOCK_RE.finditer(inner)
                }
                if name:
                    _record(name, params)

    # 2) Bare <function=...>...</function> blocks outside any wrapper.
    for fm in _FUNCTION_BLOCK_RE.finditer(text):
        # Skip blocks already covered by a <tool_call> span.
        if any(start <= fm.start() < end for start, end in consumed_spans):
            continue
        name = fm.group(1).strip()
        inner = fm.group(2)
        params = {
            pm.group(1).strip(): pm.group(2).strip()
            for pm in _PARAMETER_BLOCK_RE.finditer(inner)
        }
        if name:
            consumed_spans.append(fm.span())
            _record(name, params)

    if not calls:
        return [], text

    # Strip consumed spans in reverse to preserve offsets.
    cleaned = text
    for start, end in sorted(consumed_spans, key=lambda s: s[0], reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return calls, cleaned


def _recover_textual_tool_calls(result: CompletionResult) -> None:
    """Promote textual tool-call markup to structured ``tool_calls``.

    Small thinking-mode models (Qwen/Hermes family served via llama.cpp) often
    emit tool calls as plain-text XML inside the ``<think>`` block, so they
    end up in ``reasoning_content`` (or leak to ``content``) instead of being
    parsed as structured ``tool_calls`` by the server. If the response already
    has structured calls, we leave everything alone.
    """
    msg = result.message
    if msg.tool_calls:
        return

    sources = [
        ("reasoning", result.reasoning_content or ""),
        ("content", msg.content or ""),
    ]
    all_calls: list[ToolCall] = []
    for label, text in sources:
        calls, cleaned = _parse_textual_tool_calls(text)
        if not calls:
            continue
        all_calls.extend(calls)
        if label == "reasoning":
            result.reasoning_content = cleaned or None
        else:
            msg.content = cleaned or None
        logger.warning(
            "llama.cpp: recovered %d textual tool_call(s) from %s: %s",
            len(calls),
            label,
            ", ".join(tc.function.name for tc in calls),
        )

    if all_calls:
        msg.tool_calls = all_calls
