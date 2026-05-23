"""TrackingProvider — wraps any provider to record usage and enforce limits."""

from __future__ import annotations

import contextvars
import logging
import time
from typing import AsyncIterator

from eyetor.models.messages import CompletionResult, Message, StreamingResponse
from eyetor.models.tools import ToolDefinition
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default="unknown"
)


def _messages_text(messages: list[Message]) -> str:
    """Compact role-tagged serialization of a request, for the prompt digest."""
    return "\n".join(f"{m.role}:{m.content or ''}" for m in messages)


class UsageLimitExceeded(Exception):
    """Raised when a provider's daily usage limit has been reached."""


class TrackingProvider(BaseProvider):
    """Wraps any provider to record usage and enforce daily limits.

    Intercepts complete() calls to:
    1. Check limits before the API call
    2. Measure call duration
    3. Record usage (tokens, cost, speed, finish_reason) after the call
    """

    def __init__(
        self,
        inner: BaseProvider,
        tracker: "UsageTracker",
        provider_name: str,
        cost_estimator: "CostEstimator | None" = None,
    ) -> None:
        super().__init__(
            base_url=inner.base_url,
            model=inner.model,
            api_key=inner.api_key,
            ssl_verify=inner.ssl_verify,
            temperature=inner.temperature,
        )
        self._inner = inner
        self._tracker = tracker
        self._provider_name = provider_name
        self._cost_estimator = cost_estimator

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        from eyetor.tracking import context as ctx

        if not ctx.skip_limit.get() and not self._tracker.check_limits(
            self._provider_name
        ):
            raise UsageLimitExceeded(
                f"Daily usage limit reached for provider '{self._provider_name}'."
            )

        t0 = time.monotonic()
        result = await self._inner.complete(messages, tools, temperature)
        duration_s = time.monotonic() - t0
        duration_ms = int(duration_s * 1000)

        prompt_tokens = result.usage.prompt_tokens if result.usage else 0
        completion_tokens = result.usage.completion_tokens if result.usage else 0
        speed_tps = completion_tokens / duration_s if duration_s > 0 else 0.0

        cost = 0.0
        if result.usage and result.usage.cost > 0:
            cost = result.usage.cost
        elif self._cost_estimator:
            cost = self._cost_estimator.estimate(
                result.model or self._inner.model,
                prompt_tokens,
                completion_tokens,
                provider=self._provider_name,
            )

        self._tracker.record(
            session_id=ctx.current_session_id.get(),
            provider=self._provider_name,
            model=result.model or self._inner.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost,
            duration_ms=duration_ms,
            speed_tps=round(speed_tps, 1),
            finish_reason=result.finish_reason or "",
            agent=ctx.current_agent.get(),
            phase=ctx.current_phase.get(),
            channel=ctx.current_channel.get(),
            trace_id=ctx.current_trace_id.get(),
            tool_count=len(result.message.tool_calls or []),
            msg_count=len(messages),
            prompt_digest=ctx.make_digest(_messages_text(messages)),
            response_digest=ctx.make_digest(result.message.content or ""),
        )

        return result

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        from eyetor.tracking import context as ctx

        if not ctx.skip_limit.get() and not self._tracker.check_limits(
            self._provider_name
        ):
            raise UsageLimitExceeded(
                f"Daily usage limit reached for provider '{self._provider_name}'."
            )

        # Snapshot the tracking context NOW: the recording happens in the
        # generator's finally, which the consumer drives after any wrapping
        # `tracking_context` has already exited and reset the vars.
        snap_session = ctx.current_session_id.get()
        snap_agent = ctx.current_agent.get()
        snap_phase = ctx.current_phase.get()
        snap_channel = ctx.current_channel.get()
        snap_trace = ctx.current_trace_id.get()

        t0 = time.monotonic()
        resp = await self._inner.stream(messages, tools, temperature)
        tokens: list[str] = []
        recorded = False

        async def _stream_and_track() -> AsyncIterator[str]:
            nonlocal recorded
            try:
                async for token in resp:
                    tokens.append(token)
                    yield token
            finally:
                if not recorded:
                    recorded = True
                    duration_s = time.monotonic() - t0
                    duration_ms = int(duration_s * 1000)
                    joined = "".join(tokens)
                    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
                    if resp.usage and resp.usage.completion_tokens:
                        completion_tokens = resp.usage.completion_tokens
                    else:
                        # Last-resort fallback: this is a CHARACTER count, not
                        # a token count — only used when the provider returned
                        # no usage block even with stream_options.include_usage.
                        completion_tokens = len(joined)
                        logger.debug(
                            "stream(): no usage from provider '%s'; "
                            "using char-count fallback (%d)",
                            self._provider_name,
                            completion_tokens,
                        )
                    speed_tps = (
                        completion_tokens / duration_s if duration_s > 0 else 0.0
                    )

                    cost = 0.0
                    if resp.usage and resp.usage.cost > 0:
                        cost = resp.usage.cost
                    elif self._cost_estimator:
                        cost = self._cost_estimator.estimate(
                            self._inner.model,
                            prompt_tokens,
                            completion_tokens,
                            provider=self._provider_name,
                        )

                    self._tracker.record(
                        session_id=snap_session,
                        provider=self._provider_name,
                        model=self._inner.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        estimated_cost=cost,
                        duration_ms=duration_ms,
                        speed_tps=round(speed_tps, 1),
                        finish_reason="",
                        agent=snap_agent,
                        phase=snap_phase,
                        channel=snap_channel,
                        trace_id=snap_trace,
                        tool_count=0,
                        msg_count=len(messages),
                        prompt_digest=ctx.make_digest(_messages_text(messages)),
                        response_digest=ctx.make_digest(joined),
                    )

        return StreamingResponse(_stream_and_track(), resp.usage)

    def __repr__(self) -> str:
        return f"TrackingProvider({self._inner!r})"
