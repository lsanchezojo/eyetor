"""SchedulerChannel — runs APScheduler alongside other channels and executes periodic tasks."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from eyetor.channels.base import BaseChannel
from eyetor.scheduler.store import ScheduledTask, SchedulerStore

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*(m|min|minutes?|h|hours?|d|days?)$", re.IGNORECASE)


def _parse_trigger(schedule: str, tz: str):
    """Parse a schedule string into an APScheduler trigger."""
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    m = _INTERVAL_RE.match(schedule.strip())
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("m"):
            return IntervalTrigger(minutes=value)
        elif unit.startswith("h"):
            return IntervalTrigger(hours=value)
        elif unit.startswith("d"):
            return IntervalTrigger(days=value)

    # Assume cron expression
    return CronTrigger.from_crontab(schedule, timezone=tz)


def _write_task_log(task: ScheduledTask, response: str, log_path: str) -> None:
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"[{ts}] [{task.name}]\n{response}\n{'─' * 60}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)


class SchedulerChannel(BaseChannel):
    """Background channel that fires scheduled tasks via APScheduler.

    When a task fires:
    - Runs session.send(prompt) to get the agent's response
    - Delivers the response according to task.notify:
        "telegram" → sends to task.notify_target chat_id
        "log"      → appends to log file
        "none"     → silent execution (no output)
    """

    def __init__(
        self,
        store: SchedulerStore,
        session_manager,
        bot_token: str | None,
        default_timezone: str = "Europe/Madrid",
    ) -> None:
        self._store = store
        self._session_mgr = session_manager
        self._bot_token = bot_token
        self._default_tz = default_timezone
        self._scheduler = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._scheduler = AsyncIOScheduler(timezone=self._default_tz)

        for task in self._store.list_enabled():
            self._add_job(task)
            logger.info("Scheduled task '%s' (%s)", task.name, task.schedule)

        self._scheduler.start()
        logger.info("Scheduler started with %d task(s)", len(self._store.list_enabled()))
        await self._stop_event.wait()

    async def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Public API used by scheduler tools
    # ------------------------------------------------------------------

    def add_task(self, task: ScheduledTask) -> ScheduledTask:
        """Persist and schedule a new task."""
        self._store.add(task)
        if self._scheduler and task.enabled:
            self._add_job(task)
        logger.info("Added task '%s' (id=%s, schedule=%s)", task.name, task.id, task.schedule)
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Remove a task from the store and unschedule it."""
        deleted = self._store.delete(task_id)
        if deleted and self._scheduler:
            try:
                self._scheduler.remove_job(task_id)
            except Exception:
                pass
        return deleted

    def list_tasks(self) -> list[dict]:
        tasks = self._store.list_all()
        result = []
        for task in tasks:
            next_run = None
            if self._scheduler:
                job = self._scheduler.get_job(task.id)
                if job and job.next_run_time:
                    next_run = job.next_run_time.isoformat()
            result.append({
                "id": task.id,
                "name": task.name,
                "schedule": task.schedule,
                "notify": task.notify,
                "enabled": task.enabled,
                "last_run": task.last_run,
                "next_run": next_run,
            })
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_job(self, task: ScheduledTask) -> None:
        try:
            trigger = _parse_trigger(task.schedule, task.timezone or self._default_tz)
            self._scheduler.add_job(
                self._run_task,
                trigger,
                id=task.id,
                args=[task.id],
                replace_existing=True,
                misfire_grace_time=300,
            )
        except Exception as exc:
            logger.error("Failed to schedule task '%s': %s", task.name, exc)

    async def _run_task(self, task_id: str) -> None:
        task = self._store.get(task_id)
        if not task or not task.enabled:
            return

        logger.info("Running scheduled task '%s' (id=%s)", task.name, task_id)
        try:
            session = self._session_mgr.get_or_create(task.session_id)
            response = await session.send_sync(task.prompt)

            await self._deliver(task, response)
            self._store.update_last_run(task_id)
        except Exception as exc:
            logger.error("Error running task '%s': %s", task.name, exc)

    async def _deliver(self, task: ScheduledTask, response: str) -> None:
        if task.notify == "telegram":
            await self._send_telegram(task, response)
        elif task.notify == "log":
            log_path = task.notify_target or "~/.eyetor/scheduler.log"
            _write_task_log(task, response, log_path)
        # notify == "none": intentionally silent

    async def _send_telegram(self, task: ScheduledTask, response: str) -> None:
        if not self._bot_token or not task.notify_target:
            logger.warning("Task '%s': Telegram notify configured but no bot_token or chat_id", task.name)
            return
        try:
            from aiogram import Bot
            from eyetor.channels.telegram import _md_to_html
            bot = Bot(token=self._bot_token)
            text = f"⏰ <b>{task.name}</b>\n\n{_md_to_html(response)}"
            await bot.send_message(task.notify_target, text, parse_mode="HTML")
            await bot.session.close()
        except Exception as exc:
            logger.error("Task '%s': Telegram delivery failed: %s", task.name, exc)
