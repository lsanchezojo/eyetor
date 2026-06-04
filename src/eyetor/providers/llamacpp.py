"""llama.cpp server LLM provider (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

from eyetor.models.messages import (
    CompletionResult,
    FunctionCall,
    Message,
    StreamingResponse,
    TokenUsage,
    ToolCall,
)
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, extract_usage, parse_sse

logger = logging.getLogger(__name__)

_NO_THINKING_PHASES = frozenset(
    {
        "compaction",
        "degeneration_recovery",
        "loop_break",
        "chain_synthesize",
    }
)


class LlamaCppProvider(BaseProvider):
    """Provider adapter for llama.cpp server's OpenAI-compatible API.

    Authentication is optional — only used if api_key is set.

    When ``thinking=True``, each request includes ``chat_template_kwargs``
    with ``enable_thinking: true``, which activates the reasoning channel on
    models that support it (e.g. Gemma-4, QwQ).  The ``<think>`` block is
    stripped from the visible response but logged at DEBUG level.

    Local SLMs (Qwen3, etc.) sometimes emit tool calls as TEXT inside
    ``content`` instead of the structured ``tool_calls`` field — either as
    Hermes JSON or Llama-3 pythonic XML. ``_extract_leaked_tool_calls``
    recovers both formats and rewrites the message so the rest of the stack
    sees a normal structured call.
    """

    def __init__(
        self,
        *args: Any,
        thinking: bool = False,
        request_timeout: float = 600.0,
        reasoning_budget: int | None = None,
        max_tokens: int | None = None,
        max_tokens_by_phase: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        self.request_timeout = request_timeout
        self.reasoning_budget = reasoning_budget
        self.max_tokens = max_tokens
        self.max_tokens_by_phase = max_tokens_by_phase or {}

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        stream: bool = False,
        force_no_thinking: bool = False,
    ) -> dict[str, Any]:
        # NOTE: we do NOT inject a tool-format hint here. The GGUF's chat
        # template already teaches the model the exact XML pythonic format
        # llama.cpp doesn't parse (`<tool_call><function=...><parameter=...>`).
        # Adding a contradictory JSON-format hint makes Qwen3 flip-flop
        # between formats and degrades tool-name fidelity.
        payload = super()._build_payload(messages, tools, temperature, stream)
        thinking_enabled = self._thinking_enabled_for_current_phase(force_no_thinking)
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
        if thinking_enabled:
            if self.reasoning_budget is not None and self.reasoning_budget > 0:
                payload["reasoning_budget_tokens"] = self.reasoning_budget
        max_tokens = self._max_tokens_for_current_phase()
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
            # llama.cpp also understands n_predict; setting both keeps the
            # OpenAI-compatible and native knobs aligned.
            payload["n_predict"] = max_tokens
        return payload

    def _thinking_enabled_for_current_phase(
        self, force_no_thinking: bool = False
    ) -> bool:
        if force_no_thinking or not self.thinking:
            return False
        phase = self._current_phase()
        return phase not in _NO_THINKING_PHASES

    def _max_tokens_for_current_phase(self) -> int | None:
        if not self.max_tokens_by_phase:
            return self.max_tokens
        phase = self._current_phase()
        if phase and phase in self.max_tokens_by_phase:
            return int(self.max_tokens_by_phase[phase])
        return self.max_tokens

    def _current_phase(self) -> str:
        try:
            from eyetor.tracking.context import current_phase

            return current_phase.get()
        except Exception:  # pragma: no cover - defensive fallback
            return ""

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        thinking_enabled = self._thinking_enabled_for_current_phase()
        result = await self._complete_once(
            messages, tools, temperature, force_no_thinking=False
        )
        # Think-only degeneration: with the reasoning channel open, Qwen3 Q4
        # sometimes emits a tiny <think> block then EOS with no content and no
        # tool_call. The reasoning is usually too short to hold a leaked
        # <tool_call>, so it would escalate to a remote provider. Retry ONCE
        # locally with thinking disabled before giving up — non-thinking mode
        # is far more reliable at directly emitting a tool_call or an answer.
        if thinking_enabled and _is_degenerate_completion(result):
            logger.info(
                "llama.cpp: empty think-only completion (phase=%s, reasoning=%dch); "
                "retrying once with thinking disabled",
                self._current_phase() or "?",
                len((result.reasoning_content or "").strip()),
            )
            retry = await self._complete_once(
                messages, tools, temperature, force_no_thinking=True
            )
            if _is_degenerate_completion(retry):
                logger.warning(
                    "llama.cpp: no-thinking retry still empty (phase=%s); "
                    "escalating. First-pass reasoning: %s",
                    self._current_phase() or "?",
                    (result.reasoning_content or "").strip()[:200],
                )
            return retry
        return result

    async def _complete_once(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        temperature: float,
        *,
        force_no_thinking: bool,
    ) -> CompletionResult:
        payload = self._build_payload(
            messages, tools, temperature, stream=False,
            force_no_thinking=force_no_thinking,
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
            response.raise_for_status()
            data = response.json()
            result = _parse_completion_response(data)
            if self.thinking and not force_no_thinking:
                reasoning = _extract_reasoning(data)
                if reasoning:
                    result.reasoning_content = reasoning
                    logger.debug("llama.cpp reasoning:\n%s", reasoning.strip())
            # Recover tool calls that the model emitted as text — first in
            # content, then in reasoning_content (Qwen3 thinking often emits
            # the next tool call inside the <think> channel, which llama.cpp
            # does not parse).
            _recover_leaked_tool_calls(result.message, tools)
            if not result.message.tool_calls and result.reasoning_content:
                calls, cleaned = _extract_leaked_tool_calls(
                    result.reasoning_content, tools
                )
                if calls:
                    result.message.tool_calls = calls
                    result.reasoning_content = cleaned or None
                    logger.info(
                        "llama.cpp: extracted %d tool_call(s) from reasoning leak "
                        "(names: %s)",
                        len(calls),
                        ", ".join(c.function.name for c in calls),
                    )
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
            usage: TokenUsage | None = None
            # Buffer all content so a leaked <tool_call> block never reaches
            # the consumer character-by-character — we strip it at the end.
            content_buf: list[str] = []
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
                            content_buf.append(text)
                        extracted = extract_usage(chunk)
                        if extracted:
                            usage = extracted
            full = "".join(content_buf)
            calls, cleaned = _extract_leaked_tool_calls(full, tools)
            if calls:
                logger.info(
                    "llama.cpp stream: extracted %d tool_call(s) from content leak; "
                    "hiding raw XML/JSON from output",
                    len(calls),
                )
            if cleaned:
                yield cleaned
            if reasoning_parts:
                raw_reasoning = "".join(reasoning_parts)
                # If the model emitted its tool call in the reasoning channel,
                # strip it from what gets exposed downstream so it never
                # reaches a content-fallback path.
                r_calls, r_clean = _extract_leaked_tool_calls(raw_reasoning, tools)
                if r_calls:
                    logger.info(
                        "llama.cpp stream: extracted %d tool_call(s) from "
                        "reasoning leak",
                        len(r_calls),
                    )
                sr.reasoning_content = r_clean or None
            if usage is not None:
                sr._usage = usage

        sr._iterator = _stream_tokens()
        return sr


# ------------------------------------------------------------------
# Reasoning helpers
# ------------------------------------------------------------------

def _is_degenerate_completion(result: CompletionResult) -> bool:
    """True when the model returned no usable output (no content, no tool_call).

    Mirrors ``FallbackProvider._is_empty_completion`` but lives here so the
    provider can self-recover before the fallback chain ever escalates.
    """
    message = result.message
    if message.tool_calls:
        return False
    return not (message.content or "").strip()


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


# ------------------------------------------------------------------
# Tool-call leak recovery
# ------------------------------------------------------------------

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_PY_FUNC_RE = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE
)
_PY_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


def _recover_leaked_tool_calls(
    message: Message, tools: list[ToolDefinition] | None
) -> None:
    """Mutate ``message`` in place when content holds a leaked tool call.

    If the model already returned a structured ``tool_calls`` array we trust
    it and do nothing. Otherwise we scan ``content`` for ``<tool_call>``
    blocks; on success the calls are attached and ``content`` is replaced by
    the surrounding text (typically empty).
    """
    if message.tool_calls or not message.content:
        return
    calls, cleaned = _extract_leaked_tool_calls(message.content, tools)
    if not calls:
        return
    message.tool_calls = calls
    message.content = cleaned or None
    logger.info(
        "llama.cpp: extracted %d tool_call(s) from content leak (names: %s)",
        len(calls),
        ", ".join(c.function.name for c in calls),
    )


def _extract_leaked_tool_calls(
    content: str, tools: list[ToolDefinition] | None
) -> tuple[list[ToolCall], str]:
    """Pull ``<tool_call>...</tool_call>`` blocks out of ``content``.

    Accepts both Hermes JSON (``{"name":..., "arguments":...}``) and the
    Llama-3 pythonic XML (``<function=NAME><parameter=KEY>VAL</parameter>``).

    Returns ``(calls, cleaned_content)``. Unparseable blocks are left in
    place so the user can still see the model's intent if it was, e.g., a
    prose ``<tool_call>`` mention.
    """
    if not content or "<tool_call>" not in content.lower():
        return [], content
    calls: list[ToolCall] = []

    def _on_match(m: re.Match) -> str:
        block = m.group(1).strip()
        call = _parse_tool_call_block(block, tools)
        if call is None:
            return m.group(0)
        calls.append(call)
        return ""

    cleaned = _TOOL_CALL_BLOCK_RE.sub(_on_match, content).strip()
    return calls, cleaned


def _parse_tool_call_block(
    block: str, tools: list[ToolDefinition] | None
) -> ToolCall | None:
    """Parse a single ``<tool_call>`` body. Returns None if unrecognized."""
    # 1. Hermes JSON. Strip code fences first — some models wrap it.
    candidate = _CODE_FENCE_RE.sub("", block).strip()
    if candidate.startswith("{"):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "name" in data:
            name = str(data["name"])
            args = data.get("arguments", data.get("parameters", {}))
            args_str = _coerce_arguments(args)
            return _build_tool_call(name, args_str, tools)

    # 2. Llama-3 pythonic XML.
    fm = _PY_FUNC_RE.search(block)
    if fm:
        name = fm.group(1)
        inner = fm.group(2)
        params: dict[str, Any] = {}
        for pm in _PY_PARAM_RE.finditer(inner):
            params[pm.group(1)] = pm.group(2).strip()
        if not params:
            # Some variants put the args raw between the tags.
            raw = inner.strip()
            if raw:
                params = {"args": raw}
        return _build_tool_call(name, json.dumps(params, ensure_ascii=False), tools)

    return None


def _coerce_arguments(args: Any) -> str:
    """Normalize the ``arguments`` field to a JSON string (OpenAI wire format)."""
    if isinstance(args, str):
        # Already a JSON string per OpenAI convention; validate by round-trip
        # to avoid corrupting it on the way through.
        try:
            json.loads(args)
            return args
        except json.JSONDecodeError:
            return json.dumps({"value": args}, ensure_ascii=False)
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return json.dumps({"value": args}, ensure_ascii=False)


def _build_tool_call(
    name: str, args_str: str, tools: list[ToolDefinition] | None
) -> ToolCall:
    resolved = _resolve_tool_name(name, tools)
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:12]}",
        function=FunctionCall(name=resolved, arguments=args_str),
    )


def _resolve_tool_name(name: str, tools: list[ToolDefinition] | None) -> str:
    """Fuzzy-match a model-invented tool name against the registry.

    SLMs frequently strip separators (``skill.web-search`` → ``skillwebsearch``)
    or change them (``fs.read`` → ``fs_read``). We pick the closest match by
    normalized form so the registry can still execute the call. Falls back to
    the original name when there's no plausible match — the registry will
    raise a clear error rather than silently picking the wrong tool.
    """
    if not tools:
        return name
    real = [t.name for t in tools]
    if name in real:
        return name

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    target = _norm(name)
    if not target:
        return name
    # 1. Exact match on normalized form.
    for r in real:
        if _norm(r) == target:
            logger.info("llama.cpp: tool name '%s' resolved to '%s'", name, r)
            return r
    # 2. Substring containment (model truncated or extended).
    for r in real:
        nr = _norm(r)
        if nr and (target in nr or nr in target):
            logger.info(
                "llama.cpp: tool name '%s' fuzzy-matched to '%s'", name, r
            )
            return r
    logger.warning(
        "llama.cpp: tool name '%s' not found in registry (available: %s); "
        "passing through unchanged",
        name,
        ", ".join(real),
    )
    return name
