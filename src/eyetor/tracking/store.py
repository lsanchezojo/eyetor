"""SQLite-backed store for LLM usage tracking."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class UsageRecord:
    """A single LLM call usage record."""

    id: int
    session_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    timestamp: str
    duration_ms: int = 0
    speed_tps: float = 0.0
    finish_reason: str = ""


@dataclass
class UsageSummary:
    """Aggregated usage statistics."""

    provider: str
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


_DDL = """
CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost    REAL    NOT NULL DEFAULT 0.0,
    timestamp         TEXT    NOT NULL,
    duration_ms       INTEGER NOT NULL DEFAULT 0,
    speed_tps         REAL    NOT NULL DEFAULT 0.0,
    finish_reason     TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_provider  ON usage(provider);
"""

_MIGRATIONS = [
    "ALTER TABLE usage ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE usage ADD COLUMN speed_tps REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE usage ADD COLUMN finish_reason TEXT NOT NULL DEFAULT ''",
]


class TrackingStore:
    """Persistent storage for LLM usage records."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply schema migrations (safe to run multiple times)."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

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
        """Insert a usage record."""
        self._conn.execute(
            """
            INSERT INTO usage (session_id, provider, model, prompt_tokens,
                               completion_tokens, estimated_cost, timestamp,
                               duration_ms, speed_tps, finish_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                estimated_cost,
                datetime.utcnow().isoformat(),
                duration_ms,
                speed_tps,
                finish_reason,
            ),
        )
        self._conn.commit()

    def get_summary(
        self,
        period: str = "day",
        provider: str | None = None,
    ) -> list[UsageSummary]:
        """Aggregate usage by provider+model for the given period."""
        delta = {"day": 1, "week": 7, "month": 30}.get(period, 1)
        since = (datetime.utcnow() - timedelta(days=delta)).isoformat()
        params: list = [since]
        where_provider = ""
        if provider:
            where_provider = "AND provider = ?"
            params.append(provider)

        rows = self._conn.execute(
            f"""
            SELECT provider, model,
                   COUNT(*) as calls,
                   SUM(prompt_tokens) as prompt_tokens,
                   SUM(completion_tokens) as completion_tokens,
                   SUM(prompt_tokens + completion_tokens) as total_tokens,
                   SUM(estimated_cost) as estimated_cost
            FROM usage
            WHERE timestamp >= ? {where_provider}
            GROUP BY provider, model
            ORDER BY total_tokens DESC
            """,
            params,
        ).fetchall()
        return [
            UsageSummary(
                provider=r["provider"],
                model=r["model"],
                calls=r["calls"],
                prompt_tokens=r["prompt_tokens"] or 0,
                completion_tokens=r["completion_tokens"] or 0,
                total_tokens=r["total_tokens"] or 0,
                estimated_cost=r["estimated_cost"] or 0.0,
            )
            for r in rows
        ]

    def get_recent(
        self,
        limit: int = 10,
        provider: str | None = None,
    ) -> list[UsageRecord]:
        """Return the most recent individual usage records."""
        params: list = []
        where = ""
        if provider:
            where = "WHERE provider = ?"
            params.append(provider)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT id, session_id, provider, model, prompt_tokens,
                   completion_tokens, estimated_cost, timestamp,
                   duration_ms, speed_tps, finish_reason
            FROM usage {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            UsageRecord(
                id=r["id"],
                session_id=r["session_id"],
                provider=r["provider"],
                model=r["model"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                estimated_cost=r["estimated_cost"] or 0.0,
                timestamp=r["timestamp"],
                duration_ms=r["duration_ms"] or 0,
                speed_tps=r["speed_tps"] or 0.0,
                finish_reason=r["finish_reason"] or "",
            )
            for r in rows
        ]

    def get_daily_totals(self, provider: str) -> dict:
        """Daily totals for a specific provider (for limit checks)."""
        since = (datetime.utcnow() - timedelta(days=1)).isoformat()
        row = self._conn.execute(
            """
            SELECT SUM(prompt_tokens + completion_tokens) as total_tokens,
                   SUM(estimated_cost) as total_cost
            FROM usage WHERE provider=? AND timestamp >= ?
            """,
            (provider, since),
        ).fetchone()
        return {
            "total_tokens": row["total_tokens"] or 0,
            "total_cost": row["total_cost"] or 0.0,
        }

    def close(self) -> None:
        self._conn.close()
