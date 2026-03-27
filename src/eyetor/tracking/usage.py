"""UsageTracker — records LLM calls and enforces limits."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.config import TrackingConfig
from eyetor.tracking.store import TrackingStore, UsageSummary

logger = logging.getLogger(__name__)


class UsageTracker:
    """Records token usage per LLM call and checks daily limits."""

    def __init__(self, store: TrackingStore, config: TrackingConfig) -> None:
        self._store = store
        self._config = config

    @classmethod
    def from_config(cls, config: TrackingConfig) -> "UsageTracker":
        """Create a UsageTracker from config (expands ~ in db_path)."""
        path = Path(config.db_path).expanduser()
        return cls(TrackingStore(path), config)

    def record(
        self,
        session_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost: float = 0.0,
    ) -> None:
        """Record a completed LLM call."""
        self._store.record(
            session_id=session_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
        )
        logger.debug(
            "Usage recorded: provider=%s model=%s tokens=%d+%d cost=%.4f",
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            estimated_cost,
        )

    def check_limits(self, provider: str) -> bool:
        """Return True if limits are OK, False if any daily limit is exceeded."""
        limits = self._config.limits.get(provider)
        if not limits:
            return True
        totals = self._store.get_daily_totals(provider)
        if limits.daily_tokens and totals["total_tokens"] >= limits.daily_tokens:
            logger.warning(
                "Daily token limit reached for provider '%s': %d >= %d",
                provider,
                totals["total_tokens"],
                limits.daily_tokens,
            )
            return False
        if limits.daily_cost_usd and totals["total_cost"] >= limits.daily_cost_usd:
            logger.warning(
                "Daily cost limit reached for provider '%s': %.4f >= %.4f",
                provider,
                totals["total_cost"],
                limits.daily_cost_usd,
            )
            return False
        return True

    def get_summary(self, period: str = "day", provider: str | None = None) -> list[UsageSummary]:
        """Return aggregated usage summaries."""
        return self._store.get_summary(period=period, provider=provider)
