"""schedule_task / list_tasks / cancel_task tools (per session)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from eyetor.models.tools import ToolDefinition, ToolRegistry

if TYPE_CHECKING:
    from eyetor.scheduler.channel import SchedulerChannel


def register_scheduler_tools(
    tool_registry: ToolRegistry, scheduler: "SchedulerChannel", session_id: str
) -> None:
    """Register schedule_task / list_tasks / cancel_task tools."""
    from eyetor.scheduler.store import ScheduledTask

    # Auto-derive Telegram chat_id from session_id (e.g. "telegram-123456" → "123456")
    notify_target = (
        session_id.split("-", 1)[1] if session_id.startswith("telegram-") else None
    )

    async def handle_schedule_task(
        name: str,
        prompt: str,
        schedule: str,
        notify: str = "telegram",
        timezone: str = "Europe/Madrid",
        notify_target_override: str | None = None,
        user_confirmed_midnight: bool = False,
    ) -> str:
        # Defensive validation: refuse cron with hour=0 unless explicitly confirmed.
        # Cron has 5 fields: 'minute hour day month dow'. If hour field is exactly '0'
        # and the user has not confirmed midnight, ask for clarification instead of
        # silently scheduling at 00:00 (a common LLM failure mode).
        schedule_clean = schedule.strip()
        cron_parts = schedule_clean.split()
        if (
            len(cron_parts) == 5
            and cron_parts[1] == "0"
            and not user_confirmed_midnight
        ):
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "El cron tiene hora 00:00. ¿El usuario ha pedido medianoche "
                        "explícitamente? Si no, pregúntale a qué hora quiere el "
                        "recordatorio antes de volver a llamar a esta herramienta. "
                        "Si realmente quiere medianoche, vuelve a llamar con "
                        "user_confirmed_midnight=true."
                    ),
                }
            )

        target = notify_target_override or (
            notify_target if notify == "telegram" else None
        )
        try:
            task = ScheduledTask(
                name=name,
                prompt=prompt,
                schedule=schedule,
                timezone=timezone,
                session_id=f"scheduler-{session_id}",
                notify=notify,
                notify_target=target,
            )
            added = scheduler.add_task(task)
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"No se pudo crear la tarea: {exc}",
                }
            )
        tasks = scheduler.list_tasks()
        next_run = next((t["next_run"] for t in tasks if t["id"] == added.id), None)
        return json.dumps(
            {
                "ok": True,
                "task_id": added.id,
                "name": added.name,
                "schedule": added.schedule,
                "notify": added.notify,
                "next_run": next_run,
            }
        )

    async def handle_list_tasks() -> str:
        return json.dumps({"ok": True, "tasks": scheduler.list_tasks()})

    async def handle_cancel_task(
        task_id: str | None = None,
        name: str | None = None,
    ) -> str:
        if not task_id and not name:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Indica task_id o name. Si no conoces el ID, llama "
                        "primero a list_tasks o pasa name con el nombre de la tarea."
                    ),
                }
            )
        if name and not task_id:
            needle = name.lower().strip()
            all_tasks = scheduler.list_tasks()
            matches = [t for t in all_tasks if needle in t["name"].lower()]
            if not matches:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"Ninguna tarea coincide con '{name}'.",
                        "available": [
                            {"id": t["id"], "name": t["name"]} for t in all_tasks
                        ],
                    }
                )
            if len(matches) > 1:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "ambiguous",
                        "message": (
                            f"Varias tareas coinciden con '{name}'. "
                            "Pide al usuario cuál cancelar y vuelve a llamar con task_id."
                        ),
                        "matches": [
                            {"id": t["id"], "name": t["name"], "schedule": t["schedule"]}
                            for t in matches
                        ],
                    }
                )
            task_id = matches[0]["id"]
        deleted = scheduler.cancel_task(task_id)
        if deleted:
            return json.dumps({"ok": True, "cancelled": task_id})
        return json.dumps({"ok": False, "error": f"Task '{task_id}' not found."})

    tool_registry.register(
        ToolDefinition(
            name="schedule_task",
            description=(
                "Schedule an automatic task. 3 'schedule' modes: "
                "ONE-SHOT ('2026-04-16 09:00', 'next thursday at 9', 'tomorrow at 8', "
                "'in 30 seconds', 'in 10 minutes', 'in 2 hours'); "
                "RECURRING 5-field cron m h dom mon dow ('0 9 * * 4' = Thursday 9:00, "
                "dow 0=Sun..6=Sat); INTERVAL ('every 30m', 'every 2h', 'every 1d'). "
                "Spanish input rules: 'el jueves' (singular) = ONE-SHOT next Thursday, "
                "NOT cron; 'los jueves'/'cada' = recurring. If no time is given, ask for "
                "it (do not invent 00:00 or 09:00); for midnight, confirm and pass "
                "user_confirmed_midnight=true. After creating, confirm the mode and 'next_run'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short descriptive name (e.g. 'Comprar pan')",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Instruction for the agent when the task fires, written "
                            "TO the agent (e.g. 'Recuérdale al usuario que saque la "
                            "basura'), not a message to the user."
                        ),
                    },
                    "schedule": {
                        "type": "string",
                        "description": (
                            "When to run. One-shot: '2026-04-16 09:00', "
                            "'next thursday at 9', 'tomorrow at 18:00', "
                            "'in 30 seconds', 'in 10 minutes'. "
                            "Recurring: 5-field cron '0 9 * * 4'. "
                            "Interval: 'every 30m', 'every 2h', 'every 1d'."
                        ),
                    },
                    "notify": {
                        "type": "string",
                        "enum": ["telegram", "log", "none"],
                        "description": "Where to deliver the result. Default: 'telegram'",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone for crons and relative dates (e.g. 'Europe/Madrid'). Default: 'Europe/Madrid'",
                    },
                    "notify_target_override": {
                        "type": "string",
                        "description": "Overrides the log path (only if notify='log'). Leave empty for the default.",
                    },
                    "user_confirmed_midnight": {
                        "type": "boolean",
                        "description": "Pass true ONLY if the user explicitly asked for midnight. Default false.",
                    },
                },
                "required": ["name", "prompt", "schedule"],
            },
            handler=handle_schedule_task,
            group="scheduler",
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="list_tasks",
            description="List all scheduled tasks with their next run time and status.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handle_list_tasks,
            group="scheduler",
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="cancel_task",
            description=(
                "Cancel and delete a scheduled task. Pass task_id (preferred, exact) "
                "or name (case-insensitive substring search). If name matches multiple "
                "tasks, the tool returns 'ambiguous' with the candidate list so you ask "
                "the user to confirm before calling again with the right task_id. When "
                "the user says 'borra esa tarea', 'cancela el recordatorio del pan' or "
                "similar, call list_tasks first or use name directly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The exact task ID (returned by list_tasks or schedule_task).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Task name or name fragment (substring search).",
                    },
                },
                "required": [],
            },
            handler=handle_cancel_task,
            group="scheduler",
        )
    )
