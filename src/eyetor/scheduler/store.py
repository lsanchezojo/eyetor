"""SQLite-backed persistence for scheduled tasks."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ScheduledTask:
    name: str
    prompt: str
    schedule: str          # cron "0 9 * * *" or interval "every 30m"
    session_id: str        # session used for execution
    notify: str            # "telegram" | "log" | "none"
    timezone: str = "Europe/Madrid"
    notify_target: str | None = None   # chat_id or log path
    enabled: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run: str | None = None


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    schedule TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Europe/Madrid',
    session_id TEXT NOT NULL,
    notify TEXT NOT NULL DEFAULT 'telegram',
    notify_target TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_run TEXT
)
"""


class SchedulerStore:
    """Persistent store for ScheduledTask objects backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def add(self, task: ScheduledTask) -> ScheduledTask:
        self._conn.execute(
            """INSERT INTO scheduled_tasks
               (id, name, prompt, schedule, timezone, session_id, notify, notify_target,
                enabled, created_at, last_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task.id, task.name, task.prompt, task.schedule, task.timezone,
             task.session_id, task.notify, task.notify_target,
             int(task.enabled), task.created_at, task.last_run),
        )
        self._conn.commit()
        return task

    def get(self, task_id: str) -> ScheduledTask | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _row_to_task(row) if row else None

    def list_all(self) -> list[ScheduledTask]:
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at"
        ).fetchall()
        return [_row_to_task(r) for r in rows]

    def list_enabled(self) -> list[ScheduledTask]:
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY created_at"
        ).fetchall()
        return [_row_to_task(r) for r in rows]

    def delete(self, task_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_last_run(self, task_id: str) -> None:
        self._conn.execute(
            "UPDATE scheduled_tasks SET last_run = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), task_id),
        )
        self._conn.commit()

    def set_enabled(self, task_id: str, enabled: bool) -> bool:
        cur = self._conn.execute(
            "UPDATE scheduled_tasks SET enabled = ? WHERE id = ?",
            (int(enabled), task_id),
        )
        self._conn.commit()
        return cur.rowcount > 0


def _row_to_task(row: sqlite3.Row) -> ScheduledTask:
    return ScheduledTask(
        id=row["id"],
        name=row["name"],
        prompt=row["prompt"],
        schedule=row["schedule"],
        timezone=row["timezone"],
        session_id=row["session_id"],
        notify=row["notify"],
        notify_target=row["notify_target"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        last_run=row["last_run"],
    )
