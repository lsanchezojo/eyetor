"""Dreams analyzer — analyzes sessions, tracking, and reasoning to find patterns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from eyetor.dreams.config import DreamConfig
from eyetor.dreams.store import DreamAnalysis, DreamsStore, Finding, FindingType, Priority

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Summary of a single session."""

    session_id: str
    messages_count: int
    tool_calls: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    reasoning_snippets: list[str]


class DreamsAnalyzer:
    """Analyzes sessions to find patterns and potential improvements."""

    def __init__(
        self,
        store: DreamsStore,
        sessions_dir: Path,
        tracking_db: Path,
        memory_db: Path,
        config: DreamConfig,
    ) -> None:
        self._store = store
        self._sessions_dir = sessions_dir
        self._tracking_db = tracking_db
        self._memory_db = memory_db
        self._config = config
        self._findings: list[Finding] = []

    async def run_analysis(self) -> DreamAnalysis:
        """Run the full dream analysis."""
        date = datetime.utcnow().strftime("%Y-%m-%d")

        logger.info("Starting dream analysis for %s", date)

        self._findings = []

        sessions = self._load_recent_sessions()
        logger.info("Loaded %d sessions", len(sessions))

        tracking_summary = self._analyze_tracking()
        logger.info("Tracking: %d errors, $%.4f cost", tracking_summary["errors"], tracking_summary["cost"])

        self._find_errors(sessions, tracking_summary)
        self._find_inefficiencies(sessions, tracking_summary)
        self._analyze_reasoning(sessions)

        proposals = self._generate_proposal_list()

        analysis = DreamAnalysis(
            date=date,
            sessions_count=len(sessions),
            tool_calls_count=sum(len(s.tool_calls) for s in sessions),
            errors_count=tracking_summary["errors"],
            total_cost=tracking_summary["cost"],
            findings=self._findings,
            proposals=proposals,
        )

        return analysis

    def _load_recent_sessions(self) -> list[SessionSummary]:
        """Load sessions from the last N days."""
        sessions_dir = Path(self._sessions_dir).expanduser()
        days = self._config.days_to_analyze
        cutoff = datetime.utcnow() - timedelta(days=days)

        summaries = []

        if not sessions_dir.exists():
            return summaries

        for jsonl_file in sessions_dir.glob("*.jsonl"):
            try:
                mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
                if mtime < cutoff:
                    continue

                messages = []
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            messages.append(data)
                        except json.JSONDecodeError:
                            continue

                tool_calls = []
                errors = []
                reasoning_snippets = []

                for msg in messages:
                    if msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            tool_calls.append({
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments"),
                            })
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "")
                        if "error" in content.lower() or "failed" in content.lower():
                            errors.append({
                                "message": content[:500],
                                "session": jsonl_file.stem,
                            })
                    if msg.get("reasoning_content"):
                        reasoning_snippets.append(msg["reasoning_content"])

                summaries.append(SessionSummary(
                    session_id=jsonl_file.stem,
                    messages_count=len(messages),
                    tool_calls=tool_calls,
                    errors=errors,
                    reasoning_snippets=reasoning_snippets,
                ))
            except Exception as e:
                logger.warning("Failed to load session %s: %s", jsonl_file, e)
                continue

        return summaries

    def _analyze_tracking(self) -> dict[str, Any]:
        """Analyze tracking store for errors and costs."""
        import sqlite3

        result = {"errors": 0, "cost": 0.0, "calls": 0, "slow_calls": 0}

        try:
            conn = sqlite3.connect(str(self._tracking_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as calls,
                    SUM(CASE WHEN finish_reason IN ('error', 'timeout') THEN 1 ELSE 0 END) as errors,
                    COALESCE(SUM(estimated_cost), 0) as cost,
                    SUM(CASE WHEN duration_ms > ? THEN 1 ELSE 0 END) as slow_calls
                FROM usage
                WHERE timestamp >= datetime('now', '-{} days')
                """,
                (self._config.thresholds.slow_tool_ms, self._config.days_to_analyze),
            )
            row = cursor.fetchone()
            if row:
                result["errors"] = row["errors"] or 0
                result["cost"] = row["cost"] or 0.0
                result["calls"] = row["calls"] or 0
                result["slow_calls"] = row["slow_calls"] or 0
            conn.close()
        except Exception as e:
            logger.warning("Failed to analyze tracking: %s", e)

        return result

    def _find_errors(self, sessions: list[SessionSummary], tracking: dict) -> None:
        """Identify error patterns."""
        threshold = self._config.thresholds

        if threshold.critical_error and tracking["errors"] > 0:
            error_tools = {}
            for session in sessions:
                for error in session.errors:
                    for tc in session.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        if tool_name not in error_tools:
                            error_tools[tool_name] = []
                        error_tools[tool_name].append(error.get("message", "")[:200])

            for tool_name, messages in error_tools.items():
                if len(messages) >= 1:
                    context = f"{len(messages)} occurrence(s)"
                    finding = Finding(
                        type=FindingType.ERROR_CRITICAL if len(messages) > 1 else FindingType.ERROR_RECOVERED,
                        priority=Priority.CRITICAL if len(messages) > 1 else Priority.HIGH,
                        tool_name=tool_name,
                        description=f"Errors in {tool_name}",
                        context=context,
                        evidence=messages[:5],
                    )
                    self._findings.append(finding)

    def _find_inefficiencies(self, sessions: list[SessionSummary], tracking: dict) -> None:
        """Identify inefficiency patterns."""
        threshold = self._config.thresholds

        if tracking["slow_calls"] > 0:
            tool_durations = {}
            for session in sessions:
                for tc in session.tool_calls:
                    name = tc.get("name", "unknown")
                    if name not in tool_durations:
                        tool_durations[name] = 0
                    tool_durations[name] += 1

            for tool_name, count in tool_durations.items():
                if count >= 3 and tool_name not in ["bash", "read", "grep"]:
                    finding = Finding(
                        type=FindingType.INEFFICIENCY,
                        priority=Priority.MEDIUM,
                        tool_name=tool_name,
                        description=f"High usage of {tool_name}",
                        context=f"{count} calls in {len(sessions)} sessions",
                        evidence=[f"{count} tool calls detected"],
                    )
                    self._findings.append(finding)

    def _analyze_reasoning(self, sessions: list[SessionSummary]) -> None:
        """Analyze reasoning patterns for potential improvements."""
        reasoning_by_pattern: dict[str, list[str]] = {}

        for session in sessions:
            for snippet in session.reasoning_snippets:
                snippet_lower = snippet.lower()

                if "intent" in snippet_lower and "tool" in snippet_lower:
                    key = "intent_tool_mismatch"
                elif "retry" in snippet_lower or "reintentar" in snippet_lower:
                    key = "retry_pattern"
                elif "assume" in snippet_lower or "asumir" in snippet_lower or "probably" in snippet_lower:
                    key = "assumption_risk"
                else:
                    continue

                if key not in reasoning_by_pattern:
                    reasoning_by_pattern[key] = []
                reasoning_by_pattern[key].append(snippet[:300])

        for pattern, snippets in reasoning_by_pattern.items():
            if len(snippets) >= 2:
                description_map = {
                    "intent_tool_mismatch": "Potential intent/tool mismatch detected",
                    "retry_pattern": "Repeated retry patterns suggest inefficient tool selection",
                    "assumption_risk": "Assumptions in reasoning may lead to errors",
                }
                description = description_map.get(pattern, "Reasoning pattern detected")

                finding = Finding(
                    type=FindingType.REASONING_SUBOPTIMAL,
                    priority=Priority.LOW,
                    tool_name=None,
                    description=description,
                    context=f"{len(snippets)} occurrences",
                    evidence=snippets[:3],
                )
                self._findings.append(finding)

    def _generate_proposal_list(self) -> list[dict[str, Any]]:
        """Generate proposals from findings."""
        max_proposals = self._config.max_proposals
        proposals = []

        priority_map = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }

        sorted_findings = sorted(
            self._findings,
            key=lambda f: (priority_map.get(f.priority, 4), f.type.value),
        )

        for i, finding in enumerate(sorted_findings[:max_proposals]):
            proposal = self._finding_to_proposal(i + 1, finding)
            proposals.append(proposal)

        return proposals

    def _finding_to_proposal(self, index: int, finding: Finding) -> dict[str, Any]:
        """Convert a finding to a proposal."""
        title = f"Proposal {index}: {finding.description}"

        change_location, change_content = self._generate_change(finding)

        reason = f"Prioridad {finding.priority.value}: {finding.context}"

        return {
            "priority": finding.priority.value,
            "title": title,
            "description": finding.description,
            "change_location": change_location,
            "change_content": change_content,
            "reason": reason,
            "finding_type": finding.type.value,
        }

    def _generate_change(self, finding: Finding) -> tuple[str, str]:
        """Generate the suggested change for a finding."""
        if finding.type == FindingType.ERROR_CRITICAL:
            return (
                "models/tools.py",
                f"# Review timeout configuration for {finding.tool_name}",
            )
        elif finding.type == FindingType.ERROR_RECOVERED:
            return (
                "models/tools.py",
                f"# Consider adding validation for {finding.tool_name}",
            )
        elif finding.type == FindingType.INEFFICIENCY:
            return (
                "Consider caching frequent results",
                f"# {finding.tool_name} is called frequently - consider caching",
            )
        elif finding.type == FindingType.MEMORY_MISSING:
            return (
                "memory/store.py",
                "# Add user preference capture logic",
            )
        elif finding.type == FindingType.REASONING_SUBOPTIMAL:
            return (
                "agents/tool_agent.py",
                "# Improve prompt for better tool selection",
            )
        else:
            return (
                "Review and adjust",
                "# Manual review needed for this finding",
            )