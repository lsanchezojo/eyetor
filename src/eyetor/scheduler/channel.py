"""SchedulerChannel — runs APScheduler alongside other channels and executes periodic tasks."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from eyetor.channels.base import BaseChannel
from eyetor.scheduler.store import ScheduledTask, SchedulerStore

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*(m|min|minutes?|h|hours?|d|days?)$", re.IGNORECASE)

# Absolute datetime: '2026-04-16 09:00', '2026-04-16T09:00:00', optionally prefixed by 'at '
_ABS_DATE_RE = re.compile(
    r"^(?:at\s+)?(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$",
    re.IGNORECASE,
)

# Relative weekday: 'next thursday at 9', 'next monday at 09:30'
_REL_WEEKDAY_RE = re.compile(
    r"^next\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
    r"(?:\s+at\s+(\d{1,2})(?::(\d{2}))?)?$",
    re.IGNORECASE,
)

# Tomorrow: 'tomorrow at 18:00'
_TOMORROW_RE = re.compile(
    r"^tomorrow(?:\s+at\s+(\d{1,2})(?::(\d{2}))?)?$",
    re.IGNORECASE,
)

_WEEKDAY_INDEX = {
    "monday": 0, "lunes": 0,
    "tuesday": 1, "martes": 1,
    "wednesday": 2, "miercoles": 2, "miércoles": 2,
    "thursday": 3, "jueves": 3,
    "friday": 4, "viernes": 4,
    "saturday": 5, "sabado": 5, "sábado": 5,
    "sunday": 6, "domingo": 6,
}


def _resolve_relative_date(expr: str, tz: str) -> datetime | None:
    """Resolve relative date expressions ('next thursday at 9', 'tomorrow at 18:00')
    into an absolute timezone-aware datetime. Returns None if not recognized."""
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")
    now = datetime.now(zone)

    m = _REL_WEEKDAY_RE.match(expr.strip())
    if m:
        weekday_name = m.group(1).lower()
        target_weekday = _WEEKDAY_INDEX.get(weekday_name)
        if target_weekday is None:
            return None
        hour = int(m.group(2)) if m.group(2) else None
        minute = int(m.group(3)) if m.group(3) else 0
        if hour is None:
            return None  # caller should refuse: hour required
        days_ahead = (target_weekday - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # 'next' implies the upcoming occurrence, not today
        target_date = (now + timedelta(days=days_ahead)).date()
        return datetime(
            target_date.year, target_date.month, target_date.day,
            hour, minute, tzinfo=zone,
        )

    m = _TOMORROW_RE.match(expr.strip())
    if m:
        hour = int(m.group(1)) if m.group(1) else None
        minute = int(m.group(2)) if m.group(2) else 0
        if hour is None:
            return None
        target_date = (now + timedelta(days=1)).date()
        return datetime(
            target_date.year, target_date.month, target_date.day,
            hour, minute, tzinfo=zone,
        )

    return None


def _parse_trigger(schedule: str, tz: str):
    """Parse a schedule string into an APScheduler trigger.

    Supports, in order:
      1. Interval: 'every 30m', 'every 2h', 'every 1d'
      2. Absolute datetime (one-shot): '2026-04-16 09:00', 'at 2026-04-16T09:00:00'
      3. Relative date (one-shot): 'next thursday at 9', 'tomorrow at 18:00'
      4. Cron (5 fields, recurring): '0 9 * * *'
    """
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    expr = schedule.strip()

    # 1. Interval
    m = _INTERVAL_RE.match(expr)
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("m"):
            return IntervalTrigger(minutes=value)
        elif unit.startswith("h"):
            return IntervalTrigger(hours=value)
        elif unit.startswith("d"):
            return IntervalTrigger(days=value)

    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")

    # 2. Absolute datetime
    m = _ABS_DATE_RE.match(expr)
    if m:
        date_str, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4) or "0"
        y, mo, d = (int(x) for x in date_str.split("-"))
        run_date = datetime(y, mo, d, int(hh), int(mm), int(ss), tzinfo=zone)
        return DateTrigger(run_date=run_date, timezone=zone)

    # 3. Relative date
    rel = _resolve_relative_date(expr, tz)
    if rel is not None:
        return DateTrigger(run_date=rel, timezone=zone)

    # 4. Cron fallback
    return CronTrigger.from_crontab(expr, timezone=tz)


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
        # Jobs registered before start() lands here and is drained on start.
        self._pending_callable_jobs: list[tuple] = []

    async def start(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._scheduler = AsyncIOScheduler(timezone=self._default_tz)

        for task in self._store.list_enabled():
            self._add_job(task)
            logger.info("Scheduled task '%s' (%s)", task.name, task.schedule)

        for job_id, func, trigger, name in self._pending_callable_jobs:
            try:
                self._scheduler.add_job(
                    func,
                    trigger,
                    id=job_id,
                    name=name,
                    replace_existing=True,
                    misfire_grace_time=3700,
                )
                logger.info("Scheduled callable job '%s' (id=%s)", name, job_id)
            except Exception as exc:
                logger.error("Failed to schedule callable job '%s': %s", name, exc)
        self._pending_callable_jobs.clear()

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

    def add_callable_job(self, job_id: str, func, trigger, name: str) -> None:
        """Register a Python callable (sync or async) to run on a trigger.

        Unlike add_task(), this bypasses the prompt-based pipeline and executes
        the callable directly. Used for internal system jobs (e.g. dreams analysis)
        that aren't agent turns. If the scheduler hasn't started yet, the job is
        queued and registered when start() runs.
        """
        if self._scheduler and self._scheduler.running:
            try:
                self._scheduler.add_job(
                    func,
                    trigger,
                    id=job_id,
                    name=name,
                    replace_existing=True,
                    misfire_grace_time=3700,
                )
                logger.info("Scheduled callable job '%s' (id=%s)", name, job_id)
            except Exception as exc:
                logger.error("Failed to schedule callable job '%s': %s", name, exc)
        else:
            self._pending_callable_jobs.append((job_id, func, trigger, name))

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
                misfire_grace_time=3700,  # >1h to survive DST transitions
            )
        except Exception as exc:
            logger.error("Failed to schedule task '%s': %s", task.name, exc)

    async def _run_task(self, task_id: str) -> None:
        task = self._store.get(task_id)
        if not task or not task.enabled:
            return

        logger.info("Running scheduled task '%s' (id=%s)", task.name, task_id)
        try:
            # Scheduled runs are independent: wipe history so every run starts
            # clean. Without this the session accumulates 40+ messages over
            # weeks and drifts the prompt past the model's context window.
            session = self._session_mgr.get_or_create(task.session_id)
            session.reset()
            response = await session.send_sync(task.prompt)

            await self._deliver(task, response)
            self._store.update_last_run(task_id)
        except Exception as exc:
            logger.exception("Error running task '%s': %s", task.name, exc)

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
