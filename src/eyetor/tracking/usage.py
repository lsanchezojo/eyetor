"""UsageTracker — records LLM calls and enforces limits."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.config import TrackingConfig
from eyetor.tracking.store import TrackingStore, UsageRecord, UsageSummary

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
        duration_ms: int = 0,
        speed_tps: float = 0.0,
        finish_reason: str = "",
    ) -> None:
        """Record a completed LLM call."""
        self._store.record(
            session_id=session_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
            duration_ms=duration_ms,
            speed_tps=speed_tps,
            finish_reason=finish_reason,
        )
        logger.debug(
            "Usage recorded: provider=%s model=%s tokens=%d+%d cost=%.4f speed=%.1ftps finish=%s",
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            estimated_cost,
            speed_tps,
            finish_reason,
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

    def get_recent(
        self, limit: int = 10, provider: str | None = None
    ) -> list[UsageRecord]:
        """Return the most recent individual usage records."""
        return self._store.get_recent(limit=limit, provider=provider)

    def get_summary(
        self, period: str = "day", provider: str | None = None
    ) -> list[UsageSummary]:
        """Return aggregated usage summaries."""
        return self._store.get_summary(
            period=period,
            provider=provider,
            month_start_day=self._config.month_start_day,
            month_start_hour=self._config.month_start_hour,
        )

    def get_records(
        self, period: str = "day", provider: str | None = None
    ) -> list[UsageRecord]:
        """Return individual usage records for the given period."""
        return self._store.get_records(
            period=period,
            provider=provider,
            month_start_day=self._config.month_start_day,
            month_start_hour=self._config.month_start_hour,
        )

    def clear_session(self, session_id: str) -> None:
        """Clear all tracking data for a session (called on /reset)."""
        deleted = self._store.delete_session(session_id)
        logger.debug("Cleared %d usage records for session '%s'", deleted, session_id)
