"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, TYPE_CHECKING

from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.tracking import current_session_id

if TYPE_CHECKING:
    from eyetor.memory.manager import MemoryManager
    from eyetor.scheduler.channel import SchedulerChannel

logger = logging.getLogger(__name__)


class ChatSession:
    """A single ongoing conversation with an agent.

    Manages message history and runs the tool-calling loop transparently.
    Both streaming and non-streaming interfaces are provided.
    """

    def __init__(
        self,
        session_id: str,
        config: AgentConfig,
        provider: BaseProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt_suffix: str = "",
        memory_manager: "MemoryManager | None" = None,
        scheduler: "SchedulerChannel | None" = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.provider = provider
        self._messages: list[Message] = []
        self._system_prompt_suffix = system_prompt_suffix
        self._memory = memory_manager
        self._scheduler = scheduler

        if memory_manager is not None or scheduler is not None:
            # Copy the shared registry so per-session tools don't pollute other sessions
            shared = tool_registry or ToolRegistry()
            self.tool_registry = ToolRegistry()
            for tool in shared._tools.values():
                self.tool_registry.register(tool)
            if memory_manager is not None:
                self._register_memory_tools(memory_manager)
            if scheduler is not None:
                self._register_scheduler_tools(scheduler)
        else:
            self.tool_registry = tool_registry or ToolRegistry()

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def get_history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def send(self, user_input: str) -> AsyncIterator[str]:
        """Send a user message; yield streaming tokens from the assistant.

        Tool calls are executed silently. The final response is streamed.
        """
        self._messages.append(Message(role="user", content=user_input))
        tool_defs = list(self.tool_registry._tools.values()) if self.tool_registry._tools else None
        current_session_id.set(self.session_id)

        full_messages = self._get_full_messages()
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1
            # Non-streaming call to detect tool calls
            result = await self.provider.complete(
                messages=full_messages,
                tools=tool_defs,
                temperature=self.config.temperature,
            )
            response = result.message
            self._messages.append(response)
            full_messages.append(response)

            if not response.tool_calls:
                # Final answer — yield it token by token (character-level)
                content = response.content or ""
                yield content
                return

            # Execute tool calls
            for tc in response.tool_calls:
                result = await self.tool_registry.execute(tc.function.name, tc.function.arguments)
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=result)
                self._messages.append(tool_msg)
                full_messages.append(tool_msg)
                logger.debug("Tool '%s' → %d chars", tc.function.name, len(result))

        # Max iterations reached
        yield "I reached the maximum number of reasoning steps. Please try a simpler question."

    async def send_sync(self, user_input: str) -> str:
        """Send a user message and return the complete response (non-streaming)."""
        result = ""
        async for chunk in self.send(user_input):
            result += chunk
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_full_messages(self) -> list[Message]:
        """Build the full messages list including system prompt."""
        system_content = self.config.system_prompt
        if self._system_prompt_suffix:
            system_content = f"{system_content}\n\n{self._system_prompt_suffix}"
        if self._memory:
            memory_context = self._memory.build_context(self.session_id)
            if memory_context:
                system_content = f"{system_content}\n\n{memory_context}"
        messages: list[Message] = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.extend(self._messages)
        return messages

    def _register_memory_tools(self, memory_manager: "MemoryManager") -> None:
        """Register remember/forget tools backed by persistent memory."""
        session_id = self.session_id

        async def remember_handler(key: str, value: str, type: str = "fact") -> str:
            memory_manager.remember(session_id, key, value, type)
            return json.dumps({"status": "ok", "key": key, "type": type})

        async def forget_handler(key: str, type: str = "fact") -> str:
            memory_manager.forget(session_id, key, type)
            return json.dumps({"status": "ok", "key": key})

        self.tool_registry.register(ToolDefinition(
            name="remember",
            description=(
                "Save a fact, preference or note to persistent memory so it is available "
                "in future conversations. Use whenever the user shares something important "
                "about themselves, their preferences, or context you should not forget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier (e.g. 'user_name', 'preferred_language', 'project')"},
                    "value": {"type": "string", "description": "The value to remember"},
                    "type": {"type": "string", "enum": ["fact", "preference", "note"], "default": "fact"},
                },
                "required": ["key", "value"],
            },
            handler=remember_handler,
        ))

        self.tool_registry.register(ToolDefinition(
            name="forget",
            description="Delete a previously saved memory entry by key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key of the memory to delete"},
                    "type": {"type": "string", "enum": ["fact", "preference", "note"], "default": "fact"},
                },
                "required": ["key"],
            },
            handler=forget_handler,
        ))

    def _register_scheduler_tools(self, scheduler: "SchedulerChannel") -> None:
        """Register schedule_task / list_tasks / cancel_task tools."""
        from eyetor.scheduler.store import ScheduledTask

        session_id = self.session_id
        # Auto-derive Telegram chat_id from session_id (e.g. "telegram-123456" → "123456")
        notify_target = session_id.split("-", 1)[1] if session_id.startswith("telegram-") else None

        async def handle_schedule_task(
            name: str,
            prompt: str,
            schedule: str,
            notify: str = "telegram",
            timezone: str = "Europe/Madrid",
            notify_target_override: str | None = None,
        ) -> str:
            target = notify_target_override or (notify_target if notify == "telegram" else None)
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
            tasks = scheduler.list_tasks()
            next_run = next((t["next_run"] for t in tasks if t["id"] == added.id), None)
            return json.dumps({
                "ok": True,
                "task_id": added.id,
                "name": added.name,
                "schedule": added.schedule,
                "notify": added.notify,
                "next_run": next_run,
            })

        async def handle_list_tasks() -> str:
            return json.dumps({"ok": True, "tasks": scheduler.list_tasks()})

        async def handle_cancel_task(task_id: str) -> str:
            deleted = scheduler.cancel_task(task_id)
            if deleted:
                return json.dumps({"ok": True, "cancelled": task_id})
            return json.dumps({"ok": False, "error": f"Task '{task_id}' not found."})

        self.tool_registry.register(ToolDefinition(
            name="schedule_task",
            description=(
                "Create a scheduled task that runs automatically at a given time or interval. "
                "The task sends a prompt to the agent and optionally delivers the response. "
                "Notify options: 'telegram' (send to this chat), 'log' (write to file), 'none' (silent)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short descriptive name for the task"},
                    "prompt": {"type": "string", "description": "The message to send to the agent when the task fires"},
                    "schedule": {
                        "type": "string",
                        "description": (
                            "When to run. Cron (5 fields): '0 9 * * *' (daily at 9am), '0 8 * * 1' (Mondays at 8am). "
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
                        "description": "Timezone for cron schedules (e.g. 'Europe/Madrid'). Default: 'Europe/Madrid'",
                    },
                    "notify_target_override": {
                        "type": "string",
                        "description": "Override log file path (only for notify='log'). Leave empty for default.",
                    },
                },
                "required": ["name", "prompt", "schedule"],
            },
            handler=handle_schedule_task,
        ))

        self.tool_registry.register(ToolDefinition(
            name="list_tasks",
            description="List all scheduled tasks with their next run time and status.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handle_list_tasks,
        ))

        self.tool_registry.register(ToolDefinition(
            name="cancel_task",
            description="Cancel and delete a scheduled task by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task ID to cancel (from list_tasks)"},
                },
                "required": ["task_id"],
            },
            handler=handle_cancel_task,
        ))
