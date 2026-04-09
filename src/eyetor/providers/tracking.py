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
        if not self._tracker.check_limits(self._provider_name):
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
            session_id=current_session_id.get(),
            provider=self._provider_name,
            model=result.model or self._inner.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost,
            duration_ms=duration_ms,
            speed_tps=round(speed_tps, 1),
            finish_reason=result.finish_reason or "",
        )

        return result

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        if not self._tracker.check_limits(self._provider_name):
            raise UsageLimitExceeded(
                f"Daily usage limit reached for provider '{self._provider_name}'."
            )

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
                    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
                    completion_tokens = len("".join(tokens))
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
                        session_id=current_session_id.get(),
                        provider=self._provider_name,
                        model=self._inner.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        estimated_cost=cost,
                        duration_ms=duration_ms,
                        speed_tps=round(speed_tps, 1),
                        finish_reason="",
                    )

        return StreamingResponse(_stream_and_track(), resp.usage)

    def __repr__(self) -> str:
        return f"TrackingProvider({self._inner!r})"
