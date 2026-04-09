"""SQLite-backed store for LLM usage tracking."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


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
    avg_speed_tps: float = 0.0


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
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        """Open (or reopen) the database and ensure the schema exists."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        conn.commit()
        self._apply_migrations(conn)
        return conn

    def _ensure_db(self) -> None:
        """Reconnect and recreate the schema if the database is corrupt."""
        try:
            self._conn.close()
        except Exception:
            pass
        logger.warning("Tracking database corrupt or missing — recreating schema")
        self._conn = self._open()

    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        """Execute SQL with automatic recovery on corrupt/missing schema."""
        try:
            return self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            self._ensure_db()
            return self._conn.execute(sql, params)

    @staticmethod
    def _apply_migrations(conn: sqlite3.Connection) -> None:
        """Apply schema migrations (safe to run multiple times)."""
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
                conn.commit()
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
        params = (
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
        )
        sql = """
            INSERT INTO usage (session_id, provider, model, prompt_tokens,
                               completion_tokens, estimated_cost, timestamp,
                               duration_ms, speed_tps, finish_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._exec(sql, params)
        self._conn.commit()

    def _period_since(
        self,
        period: str,
        month_start_day: int = 1,
        month_start_hour: int = 0,
    ) -> str:
        """Return UTC ISO timestamp for the start of the given period.

        "day"   → start of today in local time (midnight local → UTC)
        "week"  → start of current week (Monday, midnight local → UTC)
        "month" → start of current month (day X at hour Y, local → UTC)
        """
        now_local = datetime.now()
        utc_offset = now_local - datetime.utcnow()

        if period == "day":
            midnight_local = now_local.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return (midnight_local - utc_offset).isoformat()
        if period == "week":
            days_since_monday = now_local.weekday()
            start_of_week_local = now_local.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_since_monday)
            return (start_of_week_local - utc_offset).isoformat()
        if period == "month":
            current_day = now_local.day
            if current_day >= month_start_day:
                start_of_month_local = now_local.replace(
                    day=month_start_day,
                    hour=month_start_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            else:
                prev_month = now_local.month - 1 if now_local.month > 1 else 12
                prev_year = (
                    now_local.year if now_local.month > 1 else now_local.year - 1
                )
                last_day_prev_month = (
                    (datetime(prev_year, prev_month + 1, 1) - timedelta(days=1)).day
                    if prev_month < 12
                    else 31
                )
                day = min(month_start_day, last_day_prev_month)
                start_of_month_local = datetime(
                    prev_year, prev_month, day, month_start_hour, 0, 0
                )
            return (start_of_month_local - utc_offset).isoformat()

        return (datetime.utcnow() - timedelta(days=7)).isoformat()

    def get_summary(
        self,
        period: str = "day",
        provider: str | None = None,
        month_start_day: int = 1,
        month_start_hour: int = 0,
    ) -> list[UsageSummary]:
        """Aggregate usage by provider+model for the given period."""
        since = self._period_since(period, month_start_day, month_start_hour)
        params: list = [since]
        where_provider = ""
        if provider:
            where_provider = "AND provider = ?"
            params.append(provider)

        rows = self._exec(
            f"""
            SELECT provider, model,
                   COUNT(*) as calls,
                   SUM(prompt_tokens) as prompt_tokens,
                   SUM(completion_tokens) as completion_tokens,
                   SUM(prompt_tokens + completion_tokens) as total_tokens,
                   SUM(estimated_cost) as estimated_cost,
                   AVG(NULLIF(speed_tps, 0)) as avg_speed_tps
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
                avg_speed_tps=r["avg_speed_tps"] or 0.0,
            )
            for r in rows
        ]

    def get_records(
        self,
        period: str = "day",
        provider: str | None = None,
        month_start_day: int = 1,
        month_start_hour: int = 0,
    ) -> list[UsageRecord]:
        """Return individual usage records for the given period, newest first."""
        since = self._period_since(period, month_start_day, month_start_hour)
        params: list = [since]
        where_provider = ""
        if provider:
            where_provider = "AND provider = ?"
            params.append(provider)
        rows = self._exec(
            f"""
            SELECT id, session_id, provider, model, prompt_tokens,
                   completion_tokens, estimated_cost, timestamp,
                   duration_ms, speed_tps, finish_reason
            FROM usage
            WHERE timestamp >= ? {where_provider}
            ORDER BY timestamp DESC
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
        rows = self._exec(
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
        row = self._exec(
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

    def delete_session(self, session_id: str) -> int:
        """Delete all usage records for a session. Returns count deleted."""
        cursor = self._exec(
            "DELETE FROM usage WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
