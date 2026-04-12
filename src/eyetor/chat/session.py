"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.tracking import current_session_id

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.chat.compactor import ConversationCompactor
    from eyetor.knowledge.manager import KnowledgeManager
    from eyetor.memory.manager import MemoryManager
    from eyetor.scheduler.channel import SchedulerChannel
    from eyetor.tracking.usage import UsageTracker
    from eyetor.tracking.pricing import CostEstimator
    from eyetor.workflows.observer import WorkerObserver

logger = logging.getLogger(__name__)


# Phrases a model may emit when it *announces* a tool call but forgets to
# emit the structured tool_call in the same turn. Used to decide whether
# to nudge it once before accepting the text as the final answer.
_TOOL_INTENT_RE = re.compile(
    r"(voy a (intent\w*|llamar|ejecutar|probar|reintent\w*|usar|lanzar)"
    r"|procedo a (ejecutar|llamar|usar|lanzar|invocar|probar)"
    r"|paso a (ejecutar|llamar|usar|lanzar|invocar|probar)"
    r"|(ejecutar|llamar|usar|lanzar|invocar)(é| ahora| la herramienta)"
    r"|intentar(é| de nuevo| otra vez| nuevamente)"
    r"|reintent\w+"
    r"|probar(é)? (de nuevo|otra vez)"
    r"|let me (try|call|retry|invoke|use|run|execute)"
    r"|i'?ll (try|call|retry|invoke|use|run|execute)"
    r"|i will (try|retry|call|invoke|run|execute)"
    r"|retrying|trying again"
    r"|now (i'?ll|let me|i will) (call|use|run|execute|invoke))",
    re.IGNORECASE,
)


def _mentions_tool_name(text: str, tool_defs: list) -> bool:
    """Check if the text mentions any available tool name."""
    text_lower = text.lower()
    return any(td.name.lower() in text_lower for td in tool_defs)


# Markers that indicate the model is requesting information from the user
# rather than forgetting to emit a tool_call. A message with any of these
# should NOT trigger the announce-without-call nudge: the model is legitimately
# blocked on missing parameters and correctly chose not to emit a call.
_ASK_USER_MARKERS = (
    "indícame",
    "indicame",
    "dime la",
    "dime el",
    "dime dónde",
    "dime donde",
    "proporcióname",
    "proporcioname",
    "necesito que me",
    "por favor indica",
    "por favor dime",
    "por favor proporciona",
    "please provide",
    "please tell me",
    "could you tell",
    "could you provide",
    "let me know",
)


def _is_asking_user(text: str) -> bool:
    """True if the model's message is a request for user input."""
    if not text:
        return False
    if "?" in text or "¿" in text:
        return True
    lowered = text.lower()
    return any(m in lowered for m in _ASK_USER_MARKERS)


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string for log output, collapsing whitespace."""
    if s is None:
        return ""
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(+{len(s) - max_len})"


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
        knowledge: "KnowledgeManager | None" = None,
        root_config: "VectorConfig | None" = None,
        tracker: "UsageTracker | None" = None,
        cost_estimator: "CostEstimator | None" = None,
        observer: "WorkerObserver | None" = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.provider = provider
        self._messages: list[Message] = []
        self._system_prompt_suffix = system_prompt_suffix
        self._memory = memory_manager
        self._scheduler = scheduler
        self._knowledge = knowledge
        self._root_config = root_config
        self._tracker = tracker
        self._cost_estimator = cost_estimator
        self._observer = observer

        self._compactor: ConversationCompactor | None = None
        if root_config and root_config.sessions.compaction.enabled:
            from eyetor.chat.compactor import ConversationCompactor

            self._compactor = ConversationCompactor(root_config.sessions.compaction)
            logger.info(
                "Compactor enabled: context_window=%d, trigger_at_percent=%.2f",
                root_config.sessions.compaction.context_window,
                root_config.sessions.compaction.trigger_at_percent,
            )

        # Session persistence (JSONL)
        self._persist_path: Path | None = None
        if root_config and root_config.sessions.persist:
            sessions_dir = Path(root_config.sessions.dir).expanduser()
            sessions_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[/:@\\]", "_", session_id)
            self._persist_path = sessions_dir / f"{safe_id}.jsonl"
            self._max_messages = root_config.sessions.max_messages
            self._load_history()

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
        """Clear conversation history (and persistent file)."""
        self._messages.clear()
        if self._persist_path and self._persist_path.exists():
            self._persist_path.unlink()

    def get_history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._messages)

    def change_provider(self, name: str, model_override: str | None = None) -> str:
        """Switch the active LLM provider for this session.

        Returns a human-readable confirmation string.
        """
        if not self._root_config:
            return "Error: no root config available — cannot change provider."
        from eyetor.providers import get_provider

        new_prov = get_provider(
            self._root_config,
            name,
            tracker=self._tracker,
            cost_estimator=self._cost_estimator,
        )
        if model_override:
            new_prov.model = model_override
            if hasattr(new_prov, "_inner"):
                new_prov._inner.model = model_override
        self.provider = new_prov
        active_model = model_override or new_prov.model
        return f"Provider cambiado a {name} (modelo: {active_model})"

    # ------------------------------------------------------------------
    # Session persistence (JSONL)
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        """Restore conversation history from the JSONL file on disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            lines = self._persist_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                data = json.loads(line)
                self._messages.append(Message(**data))
            logger.info(
                "Loaded %d messages from %s", len(self._messages), self._persist_path
            )
        except Exception as exc:
            logger.warning(
                "Failed to load session history from %s: %s", self._persist_path, exc
            )

    def _persist_message(self, msg: Message) -> None:
        """Append a single message to the JSONL file."""
        if not self._persist_path:
            return
        try:
            data = msg.model_dump(exclude_none=True)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            self._maybe_rotate()
        except Exception as exc:
            logger.warning("Failed to persist message: %s", exc)

    def _maybe_rotate(self) -> None:
        """Truncate the JSONL file to max_messages if it grows too large."""
        if not self._persist_path or not self._persist_path.exists():
            return
        max_msgs = getattr(self, "_max_messages", 200)
        lines = self._persist_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > max_msgs:
            keep = lines[-max_msgs:]
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
            tmp.replace(self._persist_path)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def send(self, user_input: str) -> AsyncIterator[str]:
        """Send a user message; yield streaming tokens from the assistant.

        Tool calls are executed silently. The final response is streamed.
        """
        user_msg = Message(role="user", content=user_input)
        self._messages.append(user_msg)
        self._persist_message(user_msg)
        tool_defs = (
            list(self.tool_registry._tools.values())
            if self.tool_registry._tools
            else None
        )
        current_session_id.set(self.session_id)

        if self._compactor:
            system_content = self._build_system_content()
            if self._compactor.should_compact(self._messages, system_content):
                result = await self._compactor.compact(
                    self._messages, system_content, self.provider, self.session_id
                )
                if result.compacted:
                    self._apply_compaction(result)

        full_messages = self._get_full_messages()
        iterations = 0
        recent_calls: list[str] = []  # track "name:args" for loop detection
        max_repeat = 3  # max consecutive identical tool calls before forcing answer
        nudged = False  # allow at most one "announce-without-call" nudge per turn

        while iterations < self.config.max_iterations:
            iterations += 1
            if self._observer:
                self._observer.on_iteration(iterations)
            # Non-streaming call to detect tool calls
            result = await self.provider.complete(
                messages=full_messages,
                tools=tool_defs,
                temperature=self.config.temperature,
            )
            response = result.message
            self._messages.append(response)
            self._persist_message(response)
            full_messages.append(response)
            if self._observer:
                self._observer.on_llm_response(
                    response.content or "", response.tool_calls or []
                )

            if not response.tool_calls:
                content = response.content or ""
                # Some small local models announce "voy a reintentar / I'll call X"
                # in plain text without emitting the structured tool_call. Nudge
                # once; if it still refuses, accept the text as final.
                if (
                    not nudged
                    and content
                    and tool_defs
                    and not _is_asking_user(content)
                    and (
                        _TOOL_INTENT_RE.search(content)
                        or _mentions_tool_name(content, tool_defs)
                    )
                ):
                    nudged = True
                    logger.info(
                        "Session '%s' — model announced a tool call without emitting it; nudging once",
                        self.session_id,
                    )
                    full_messages.append(
                        Message(
                            role="user",
                            content=(
                                "Has anunciado que ibas a llamar a una herramienta pero no has emitido "
                                "la tool_call estructurada. Emítela ahora con los parámetros correctos. "
                                "Si ya no necesitas más herramientas, responde al usuario directamente."
                            ),
                        )
                    )
                    continue
                # Final answer — yield it token by token (character-level)
                if self._observer:
                    self._observer.on_done(content)
                yield content
                return

            # Log tool calls at INFO level for observability
            call_names = [tc.function.name for tc in response.tool_calls]
            call_details = [
                f"{tc.function.name}({_truncate(tc.function.arguments, 200)})"
                for tc in response.tool_calls
            ]
            logger.info(
                "Session '%s' iter %d/%d — tool calls: %s",
                self.session_id,
                iterations,
                self.config.max_iterations,
                " | ".join(call_details),
            )

            # Loop detection: track "name:args" signatures
            call_signatures = [
                f"{tc.function.name}:{tc.function.arguments}"
                for tc in response.tool_calls
            ]
            current_sig = "|".join(sorted(call_signatures))
            recent_calls.append(current_sig)
            if len(recent_calls) > max_repeat:
                recent_calls = recent_calls[-max_repeat:]

            if len(recent_calls) == max_repeat and len(set(recent_calls)) == 1:
                logger.warning(
                    "Session '%s' — loop detected: same tool call(s) repeated %d times: %s. "
                    "Forcing final answer.",
                    self.session_id,
                    max_repeat,
                    call_names,
                )
                # Ask the model to answer without tools
                full_messages.append(
                    Message(
                        role="user",
                        content=(
                            "IMPORTANT: You seem to be stuck in a loop calling the same tools repeatedly. "
                            "Please provide your final answer now using the information you already have. "
                            "Do NOT call any more tools."
                        ),
                    )
                )
                result = await self.provider.complete(
                    messages=full_messages,
                    tools=None,  # no tools — force text response
                    temperature=self.config.temperature,
                )
                forced = result.message
                self._messages.append(forced)
                self._persist_message(forced)
                content = forced.content or ""
                yield content
                return

            # Execute tool calls in parallel
            async def _exec_tool(tc: ToolCall) -> tuple[ToolCall, str]:
                if self._observer:
                    self._observer.on_tool_start(
                        tc.function.name, tc.function.arguments
                    )
                r = await self.tool_registry.execute(
                    tc.function.name, tc.function.arguments
                )
                if self._observer:
                    self._observer.on_tool_end(tc.function.name, r)
                return tc, r

            exec_results = await asyncio.gather(
                *[_exec_tool(tc) for tc in response.tool_calls],
                return_exceptions=True,
            )
            for entry in exec_results:
                if isinstance(entry, BaseException):
                    logger.error(
                        "Session '%s' — tool error: %s", self.session_id, entry
                    )
                    if self._observer:
                        self._observer.on_tool_error("?", str(entry))
                    continue
                tc, result = entry
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=result)
                self._messages.append(tool_msg)
                self._persist_message(tool_msg)
                full_messages.append(tool_msg)
                logger.info(
                    "Session '%s' — tool '%s'(%s) → %d chars: %s",
                    self.session_id,
                    tc.function.name,
                    _truncate(tc.function.arguments, 200),
                    len(result),
                    _truncate(result, 200),
                )

        # Max iterations reached
        logger.warning(
            "Session '%s' — max_iterations (%d) reached. Tool calls made: %s",
            self.session_id,
            self.config.max_iterations,
            ", ".join(
                tc.function.name
                for m in self._messages
                if m.tool_calls
                for tc in m.tool_calls
            ),
        )
        msg = "I reached the maximum number of reasoning steps. Please try a simpler question."
        max_msg = Message(role="assistant", content=msg)
        self._messages.append(max_msg)
        self._persist_message(max_msg)
        yield msg

    async def send_sync(self, user_input: str) -> str:
        """Send a user message and return the complete response (non-streaming)."""
        result = ""
        async for chunk in self.send(user_input):
            result += chunk
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_content(self) -> str:
        """Build system content string from config and memory."""
        system_content = self.config.system_prompt
        if self._system_prompt_suffix:
            system_content = f"{system_content}\n\n{self._system_prompt_suffix}"
        if self._memory:
            memory_context = self._memory.build_context(self.session_id)
            if memory_context:
                system_content = f"{system_content}\n\n{memory_context}"
        if self._knowledge:
            kb_context = self._knowledge.build_context()
            if kb_context:
                system_content = f"{system_content}\n\n{kb_context}"
        return system_content

    def _get_full_messages(self) -> list[Message]:
        """Build the full messages list including system prompt."""
        system_content = self._build_system_content()
        messages: list[Message] = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.extend(self._messages)
        return messages

    def _apply_compaction(self, result) -> None:
        """Apply compaction result: archive, rewrite JSONL, update messages."""
        if result.archived_path:
            logger.info("Archived pre-compaction messages to %s", result.archived_path)

        self._messages = result.new_messages

        if self._persist_path:
            try:
                with open(self._persist_path, "w", encoding="utf-8") as f:
                    for msg in self._messages:
                        f.write(
                            json.dumps(
                                msg.model_dump(exclude_none=True), ensure_ascii=False
                            )
                            + "\n"
                        )
                logger.info(
                    "Rewrote session JSONL after compaction (%d messages)",
                    len(self._messages),
                )
            except Exception as e:
                logger.warning("Failed to rewrite JSONL after compaction: %s", e)

    def _register_memory_tools(self, memory_manager: "MemoryManager") -> None:
        """Register remember/forget tools backed by persistent memory."""
        session_id = self.session_id

        async def remember_handler(key: str, value: str, type: str = "fact") -> str:
            memory_manager.remember(session_id, key, value, type)
            return json.dumps({"status": "ok", "key": key, "type": type})

        async def forget_handler(key: str, type: str = "fact") -> str:
            memory_manager.forget(session_id, key, type)
            return json.dumps({"status": "ok", "key": key})

        self.tool_registry.register(
            ToolDefinition(
                name="remember",
                description=(
                    "Save a fact, preference or note to persistent memory so it is available "
                    "in future conversations. Use whenever the user shares something important "
                    "about themselves, their preferences, or context you should not forget."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Short identifier (e.g. 'user_name', 'preferred_language', 'project')",
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to remember",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["fact", "preference", "note"],
                            "default": "fact",
                        },
                    },
                    "required": ["key", "value"],
                },
                handler=remember_handler,
            )
        )

        self.tool_registry.register(
            ToolDefinition(
                name="forget",
                description="Delete a previously saved memory entry by key.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key of the memory to delete",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["fact", "preference", "note"],
                            "default": "fact",
                        },
                    },
                    "required": ["key"],
                },
                handler=forget_handler,
            )
        )

    def _register_scheduler_tools(self, scheduler: "SchedulerChannel") -> None:
        """Register schedule_task / list_tasks / cancel_task tools."""
        from eyetor.scheduler.store import ScheduledTask

        session_id = self.session_id
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

        self.tool_registry.register(
            ToolDefinition(
                name="schedule_task",
                description=(
                    "Programa una tarea que se ejecuta automáticamente. Tres modos:\n"
                    "\n"
                    "1) ONE-SHOT (un solo disparo) — usa una fecha-hora absoluta o relativa:\n"
                    "   - Absoluta: '2026-04-16 09:00' o 'at 2026-04-16T09:00:00'\n"
                    "   - Relativa: 'next thursday at 9', 'next monday at 18:30', 'tomorrow at 8'\n"
                    "\n"
                    "2) RECURRENTE por cron de 5 campos ('m h dom mon dow'):\n"
                    "   - '0 9 * * *' = cada día a las 9:00\n"
                    "   - '0 9 * * 4' = todos los jueves a las 9:00 (dow: 0=domingo, 4=jueves)\n"
                    "   - '30 18 * * 1-5' = lunes a viernes a las 18:30\n"
                    "\n"
                    "3) INTERVALO: 'every 30m', 'every 2h', 'every 1d'.\n"
                    "\n"
                    "REGLAS OBLIGATORIAS — léelas antes de llamar:\n"
                    "- Si el usuario dice 'el jueves', 'el lunes', 'el día X' (singular, sin "
                    "  'cada' ni 'todos los'), interprétalo como ONE-SHOT del próximo jueves/"
                    "  lunes/etc. NO crees un cron recurrente.\n"
                    "- Si el usuario dice 'los jueves', 'cada lunes', 'todos los días', es "
                    "  RECURRENTE (cron o intervalo).\n"
                    "- Si el usuario NO especifica hora, NO inventes una. PREGÚNTASELA "
                    "  primero y vuelve a llamar a esta herramienta cuando la sepas. "
                    "  Nunca uses 00:00 ni 09:00 por defecto.\n"
                    "- Si la hora es 00:00 (medianoche), debe ser porque el usuario lo pidió "
                    "  explícitamente. En ese caso pasa user_confirmed_midnight=true.\n"
                    "- Tras crear la tarea, confirma al usuario el modo (one-shot o "
                    "  recurrente) y la próxima ejecución exacta que devuelve la herramienta "
                    "  en el campo 'next_run'.\n"
                    "\n"
                    "Notify: 'telegram' (envía a este chat), 'log' (escribe a fichero), "
                    "'none' (silencioso)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Nombre corto y descriptivo (p. ej. 'Comprar pan')",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "El mensaje que se enviará al agente cuando la tarea se dispare",
                        },
                        "schedule": {
                            "type": "string",
                            "description": (
                                "Cuándo ejecutar. One-shot: '2026-04-16 09:00', "
                                "'next thursday at 9', 'tomorrow at 18:00'. "
                                "Recurrente: cron 5 campos '0 9 * * 4'. "
                                "Intervalo: 'every 30m', 'every 2h', 'every 1d'."
                            ),
                        },
                        "notify": {
                            "type": "string",
                            "enum": ["telegram", "log", "none"],
                            "description": "Dónde entregar el resultado. Default: 'telegram'",
                        },
                        "timezone": {
                            "type": "string",
                            "description": "Zona horaria para crons y fechas relativas (p. ej. 'Europe/Madrid'). Default: 'Europe/Madrid'",
                        },
                        "notify_target_override": {
                            "type": "string",
                            "description": "Sobrescribe la ruta del log (solo si notify='log'). Déjalo vacío para el default.",
                        },
                        "user_confirmed_midnight": {
                            "type": "boolean",
                            "description": "Pasa true SOLO si el usuario ha pedido medianoche explícitamente. Por defecto false.",
                        },
                    },
                    "required": ["name", "prompt", "schedule"],
                },
                handler=handle_schedule_task,
            )
        )

        self.tool_registry.register(
            ToolDefinition(
                name="list_tasks",
                description="List all scheduled tasks with their next run time and status.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handle_list_tasks,
            )
        )

        self.tool_registry.register(
            ToolDefinition(
                name="cancel_task",
                description=(
                    "Cancela y elimina una tarea programada. Puedes pasar task_id "
                    "(preferido, exacto) o name (búsqueda por subcadena, case-insensitive). "
                    "Si pasas name y hay múltiples coincidencias, la herramienta devuelve "
                    "'ambiguous' con la lista de candidatos para que pidas confirmación al "
                    "usuario antes de volver a llamar con el task_id correcto. Cuando el "
                    "usuario te diga 'borra esa tarea', 'cancela el recordatorio del pan' "
                    "o similar, llama primero a list_tasks o usa directamente name."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "El ID exacto de la tarea (devuelto por list_tasks o schedule_task).",
                        },
                        "name": {
                            "type": "string",
                            "description": "Nombre o fragmento del nombre de la tarea (búsqueda por subcadena).",
                        },
                    },
                    "required": [],
                },
                handler=handle_cancel_task,
            )
        )
