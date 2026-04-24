"""SQLite-backed store for dream proposals and analysis logs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    DISMISSED = "dismissed"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingType(str, Enum):
    ERROR_CRITICAL = "error_critical"
    ERROR_RECOVERED = "error_recovered"
    INEFFICIENCY = "inefficiency"
    MEMORY_MISSING = "memory_missing"
    REASONING_SUBOPTIMAL = "reasoning_suboptimal"


@dataclass
class Finding:
    """A single finding from analysis."""

    type: FindingType
    priority: Priority
    tool_name: str | None
    description: str
    context: str
    evidence: list[str]


@dataclass
class DreamProposal:
    """A single dream proposal."""

    id: int
    date: str
    proposal_index: int
    priority: Priority
    title: str
    description: str
    change_location: str
    change_content: str
    reason: str
    status: ProposalStatus
    created_at: str


@dataclass
class DreamAnalysis:
    """Result of a dream analysis run."""

    date: str
    sessions_count: int
    tool_calls_count: int
    errors_count: int
    total_cost: float
    findings: list[Finding]
    proposals: list[dict[str, Any]]


_DDL = """
CREATE TABLE IF NOT EXISTS dream_analyses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    sessions_count INTEGER NOT NULL DEFAULT 0,
    tool_calls_count INTEGER NOT NULL DEFAULT 0,
    errors_count INTEGER NOT NULL DEFAULT 0,
    total_cost  REAL    NOT NULL DEFAULT 0.0,
    findings   TEXT    NOT NULL,
    proposals  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_analyses_date ON dream_analyses(date);

CREATE TABLE IF NOT EXISTS dream_proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    proposal_index  INTEGER NOT NULL,
    priority        TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    change_location TEXT    NOT NULL,
    change_content  TEXT    NOT NULL,
    reason         TEXT    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'pending',
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_proposals_status ON dream_proposals(status);
CREATE INDEX IF NOT EXISTS idx_dream_proposals_date ON dream_proposals(date);
"""


class DreamsStore:
    """Persistent storage for dream proposals and analyses."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def save_analysis(self, analysis: DreamAnalysis) -> None:
        """Save a dream analysis run."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO dream_analyses
                (date, sessions_count, tool_calls_count, errors_count, total_cost, findings, proposals, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.date,
                analysis.sessions_count,
                analysis.tool_calls_count,
                analysis.errors_count,
                analysis.total_cost,
                json.dumps([f.model_dump() for f in analysis.findings], ensure_ascii=False),
                json.dumps(analysis.proposals, ensure_ascii=False),
                now,
            ),
        )
        self._conn.commit()

    def save_proposal(self, date: str, index: int, proposal: dict) -> int:
        """Save a proposal and return its ID."""
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO dream_proposals
                (date, proposal_index, priority, title, description, change_location, change_content, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                index,
                proposal.get("priority", "medium"),
                proposal.get("title", ""),
                proposal.get("description", ""),
                proposal.get("change_location", ""),
                proposal.get("change_content", ""),
                proposal.get("reason", ""),
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_pending_proposals(self, date: str | None = None) -> list[DreamProposal]:
        """Get all pending proposals, optionally filtered by date."""
        if date:
            rows = self._conn.execute(
                "SELECT * FROM dream_proposals WHERE status = ? AND date = ? ORDER BY priority, id",
                (ProposalStatus.PENDING.value, date),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM dream_proposals WHERE status = ? ORDER BY priority, id",
                (ProposalStatus.PENDING.value,),
            ).fetchall()
        return [DreamProposal(**dict(row)) for row in rows]

    def get_proposals_by_date(self, date: str) -> list[DreamProposal]:
        """Get all proposals for a specific date."""
        rows = self._conn.execute(
            "SELECT * FROM dream_proposals WHERE date = ? ORDER BY proposal_index",
            (date,),
        ).fetchall()
        return [DreamProposal(**dict(row)) for row in rows]

    def get_analysis_by_date(self, date: str) -> DreamAnalysis | None:
        """Get analysis for a specific date."""
        row = self._conn.execute(
            "SELECT * FROM dream_analyses WHERE date = ?",
            (date,),
        ).fetchone()
        if not row:
            return None
        findings = [Finding(**f) for f in json.loads(row["findings"])]
        return DreamAnalysis(
            date=row["date"],
            sessions_count=row["sessions_count"],
            tool_calls_count=row["tool_calls_count"],
            errors_count=row["errors_count"],
            total_cost=row["total_cost"],
            findings=findings,
            proposals=json.loads(row["proposals"]),
        )

    def update_proposal_status(self, proposal_id: int, status: ProposalStatus) -> None:
        """Update proposal status."""
        self._conn.execute(
            "UPDATE dream_proposals SET status = ? WHERE id = ?",
            (status.value, proposal_id),
        )
        self._conn.commit()

    def get_all_proposals(self, limit: int = 30) -> list[DreamProposal]:
        """Get all proposals (for history), most recent first."""
        rows = self._conn.execute(
            "SELECT * FROM dream_proposals ORDER BY date DESC, proposal_index LIMIT ?",
            (limit,),
        ).fetchall()
        return [DreamProposal(**dict(row)) for row in rows]