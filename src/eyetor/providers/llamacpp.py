"""llama.cpp server LLM provider (OpenAI-compatible)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from eyetor.models.messages import CompletionResult, Message, StreamingResponse, ToolCall
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider, ContextOverflowError
from eyetor.providers.openrouter import _parse_completion_response
from eyetor.streaming.parsers import extract_delta_content, parse_sse
from eyetor.utils.tool_calls import offered_tool_names, parse_textual_tool_calls

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
        protected = {"messages", "model", "tools"}
        for key, value in self.extra_body.items():
            if key not in protected:
                payload[key] = value
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
            result = _parse_completion_response(data, tools=tools)
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
                _recover_textual_tool_calls(result, tools)
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


def _recover_textual_tool_calls(
    result: CompletionResult,
    tools: list[ToolDefinition],
) -> None:
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
    tool_names = offered_tool_names(tools)
    for label, text in sources:
        parsed = parse_textual_tool_calls(text, available_tool_names=tool_names)
        if not parsed.had_markup:
            continue
        all_calls.extend(parsed.tool_calls)
        if parsed.tool_calls:
            if label == "reasoning":
                result.reasoning_content = parsed.cleaned_text or None
            else:
                msg.content = parsed.cleaned_text or None
            logger.warning(
                "llama.cpp: recovered %d textual tool_call(s) from %s: %s",
                len(parsed.tool_calls),
                label,
                ", ".join(tc.function.name for tc in parsed.tool_calls),
            )
        if parsed.unresolved_names:
            logger.warning(
                "llama.cpp: ignored unresolved textual tool_call(s) from %s: %s",
                label,
                ", ".join(parsed.unresolved_names),
            )

    if all_calls:
        msg.tool_calls = all_calls
