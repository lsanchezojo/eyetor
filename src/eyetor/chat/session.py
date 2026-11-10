"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

from eyetor.access import Access
from eyetor.chat.compactor import ConversationCompactor
from eyetor.chat.tool_gating import all_groups, select_groups
from eyetor.chat.guards import (
    _CLAIM_GUARDS,
    _FORCE_BREAK_LOOP,
    _FORCE_CLEAN_ANSWER,
    _FORCE_SYNTHESIS,
    _NUDGE_ANNOUNCED_CALL,
    _NUDGE_CONFIRMED_ACTION,
    _NUDGE_EMPTY_REPLY,
    _NUDGE_INVALID_TOOL_ARGS,
    _TOOL_INTENT_RE,
    _is_asking_user,
    _is_scaffolding_leak,
    _is_user_confirmation,
    _last_assistant_proposed_action,
    _tool_failed,
    _unbacked_claim,
)
from eyetor.chat.history import (
    _clean_loaded_history,
    _final_text,
    _invalid_tool_call_arguments,
    _is_ghost_assistant,
    _truncate,
)
from eyetor.chat.loop_detector import _LoopDetector
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.tools.memory import register_memory_tools
from eyetor.tools.scheduler import register_scheduler_tools
from eyetor.tracking.context import (
    current_conversation_id,
    current_session_id,
    current_trace_id,
    new_trace_id,
    tracking_context,
)

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.knowledge.manager import KnowledgeManager
    from eyetor.memory.manager import MemoryManager
    from eyetor.scheduler.channel import SchedulerChannel
    from eyetor.tracking.usage import UsageTracker
    from eyetor.tracking.pricing import CostEstimator
    from eyetor.workflows.observer import WorkerObserver

logger = logging.getLogger(__name__)

# Start of the note that replaces stale tool results (see trim_history).
_STALE_TOOL_PREFIX = '{"omitted": '


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
        access: "Access | None" = None,
        conversation_id: str | None = None,
        conversation_index=None,
    ) -> None:
        self.session_id = session_id
        # ChatGPT-style conversations: session_id stays the stable chat
        # identity (access/memory/scheduler scoping); conversation_id only
        # selects which history file this session reads/writes.
        self.conversation_id = conversation_id
        self._conv_index = conversation_index
        self.config = config
        self.provider = provider
        self._messages: list[Message] = []
        self._system_prompt_suffix = system_prompt_suffix
        self._access = access
        self._memory = memory_manager
        self._scheduler = scheduler
        self._knowledge = knowledge
        self._root_config = root_config
        self._tracker = tracker
        self._cost_estimator = cost_estimator
        self._observer = observer
        self.last_reasoning: str | None = None  # Accumulated reasoning from the latest send() turn
        self._force_compact_next = False

        # Conditional tool loading (see eyetor.chat.tool_gating). Token-heavy
        # tool groups are only sent on turns whose text triggers them, plus a
        # short sticky window so follow-ups ("cancélalo", "sí") still see them.
        self._gating_enabled = False
        self._sticky_turns = 2
        self._always_on_groups: set[str] = set()
        if root_config is not None:
            gating_cfg = root_config.sessions.tool_gating
            self._gating_enabled = gating_cfg.enabled
            self._sticky_turns = gating_cfg.sticky_turns
            self._always_on_groups = set(getattr(gating_cfg, "always_on_groups", []))
        self._sticky_groups: dict[str, int] = {}  # group → turns it stays active
        self._last_active_groups: set[str] = set()  # for confirmation inheritance

        self._compactor: ConversationCompactor | None = None
        if root_config and root_config.sessions.compaction.enabled:
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
            from eyetor.chat.conversations import sanitize_id

            safe_id = sanitize_id(session_id)
            stem = f"{safe_id}.{conversation_id}" if conversation_id else safe_id
            self._persist_path = sessions_dir / f"{stem}.jsonl"
            self._max_messages = root_config.sessions.max_messages
            self._load_history()

        if memory_manager is not None or scheduler is not None:
            # Copy the shared registry so per-session tools don't pollute other sessions
            shared = tool_registry or ToolRegistry()
            self.tool_registry = ToolRegistry()
            for tool in shared._tools.values():
                self.tool_registry.register(tool)
            if memory_manager is not None:
                register_memory_tools(self.tool_registry, memory_manager, self.session_id)
            if scheduler is not None:
                register_scheduler_tools(self.tool_registry, scheduler, self.session_id)
        else:
            self.tool_registry = tool_registry or ToolRegistry()

        # Per-chat access control: prune tools this chat may not use. Always
        # operates on a fresh registry so the shared one is never mutated.
        if access is not None and not access.unrestricted:
            allowed = ToolRegistry()
            removed: list[str] = []
            for tool in self.tool_registry._tools.values():
                if access.allows(tool):
                    allowed.register(tool)
                else:
                    removed.append(tool.name)
            self.tool_registry = allowed
            if removed:
                logger.info(
                    "Session '%s' — access policy: %d tool(s) blocked (%s)",
                    session_id,
                    len(removed),
                    ", ".join(sorted(removed)),
                )

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
        """Restore conversation history from the JSONL file on disk.

        Any degenerated assistant messages (empty content AND no
        ``tool_calls``) — at any position — are dropped on load. They are
        ghost turns the model produced when it collapsed; feeding them back
        into a future prompt teaches the model that empty assistant replies
        are acceptable and induces the same failure again.
        """
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            lines = self._persist_path.read_text(encoding="utf-8").strip().splitlines()
            raw: list[Message] = []
            for line in lines:
                data = json.loads(line)
                raw.append(Message(**data))
            kept, dropped = _clean_loaded_history(raw)
            self._messages.extend(kept)
            if dropped:
                logger.warning(
                    "Dropped %d invalid/ghost message(s) from %s. Rewriting JSONL.",
                    dropped,
                    self._persist_path,
                )
                # Rewrite JSONL so the contamination is gone from disk too.
                try:
                    with open(self._persist_path, "w", encoding="utf-8") as f:
                        for msg in self._messages:
                            f.write(
                                json.dumps(
                                    msg.model_dump(exclude_none=True),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                except Exception as exc:  # non-fatal: in-memory cleanup still applies
                    logger.warning("Could not rewrite cleaned JSONL: %s", exc)
            logger.info(
                "Loaded %d messages from %s", len(self._messages), self._persist_path
            )
        except Exception as exc:
            logger.warning(
                "Failed to load session history from %s: %s", self._persist_path, exc
            )

    def _persist_message(self, msg: Message, *, reasoning: str | None = None) -> None:
        """Append a single message to the JSONL file.

        If *reasoning* is provided (thinking/reasoning from the LLM), it is
        stored alongside the message under the ``reasoning_content`` key for
        auditing purposes.
        """
        if not self._persist_path:
            return
        try:
            data = msg.model_dump(exclude_none=True)
            if reasoning:
                data["reasoning_content"] = reasoning
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            self._maybe_rotate()
        except Exception as exc:
            logger.warning("Failed to persist message: %s", exc)

    def _remember_assistant(
        self,
        msg: Message,
        *,
        reasoning: str | None = None,
        context: str = "",
    ) -> bool:
        """Persist an assistant turn unless it is an empty ghost response."""
        if not _is_ghost_assistant(msg):
            self._messages.append(msg)
            self._persist_message(msg, reasoning=reasoning)
            return True

        suffix = f" during {context}" if context else ""
        logger.warning(
            "Session '%s' — ghost assistant turn%s (empty content, no tool_calls, "
            "reasoning=%dch); not persisting to history",
            self.session_id,
            suffix,
            len((reasoning or "").strip()),
        )
        return False

    def _dedupe_tool_calls(self, tool_calls: list[ToolCall] | None) -> list[ToolCall] | None:
        """Drop duplicate tool calls emitted in the same model response."""
        if not tool_calls:
            return tool_calls

        unique: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key in seen:
                logger.warning(
                    "Session '%s' — duplicate tool_call in same turn ignored: %s(%s)",
                    self.session_id,
                    tc.function.name,
                    _truncate(tc.function.arguments, 200),
                )
                continue
            seen.add(key)
            unique.append(tc)
        return unique

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
        When chain mode is active and the query is complex, delegates to
        send_chained() automatically.
        """
        # Conversation bookkeeping (title from first user message, recency).
        # Done here — NOT in send_chained, whose persisted prompt is synthetic.
        if self._conv_index is not None and self.conversation_id:
            try:
                self._conv_index.touch(self.session_id, self.conversation_id, user_input)
            except Exception as exc:  # index must never break a turn
                logger.warning("Conversation index touch failed: %s", exc)

        # Check if chain mode should be used for this query
        if self._is_complex_query(user_input):
            async for chunk in self.send_chained(user_input):
                yield chunk
            return

        self.last_reasoning = None
        user_msg = Message(role="user", content=user_input)
        self._messages.append(user_msg)
        self._persist_message(user_msg)
        tool_defs, active_groups = self._turn_tool_defs(user_input)
        if self._gating_enabled and tool_defs is not None:
            logger.info(
                "Session '%s' — tool gating: %d/%d tools, active groups: %s",
                self.session_id,
                len(tool_defs),
                len(self.tool_registry._tools),
                ", ".join(sorted(active_groups)) or "none",
            )
        current_session_id.set(self.session_id)
        current_conversation_id.set(self.conversation_id or "")
        current_trace_id.set(new_trace_id())

        full_messages = self._get_full_messages()
        if (
            tool_defs
            and _is_user_confirmation(user_input)
            and _last_assistant_proposed_action(self._messages[:-1])
        ):
            logger.info(
                "Session '%s' — user confirmed a pending action; nudging tool execution",
                self.session_id,
            )
            full_messages.append(Message(role="system", content=_NUDGE_CONFIRMED_ACTION))
        iterations = 0
        loops = _LoopDetector()
        nudged = False  # allow at most one "announce-without-call" nudge per turn
        empty_nudged = False  # allow one retry for immediate empty/no-tool replies
        degeneration_recovered = False  # one-shot post-tool synthesis fallback
        scaffold_recovered = False  # one-shot clean-synthesis fallback for leaks
        tools_executed = 0
        backed_claims: set[str] = set()  # _ClaimGuard names whose tool ran OK this turn
        claim_nudged = False  # one retry when an action is claimed but not executed

        while iterations < self.config.max_iterations:
            iterations += 1
            if self._observer:
                self._observer.on_iteration(iterations)
            full_messages = await self._compact_before_llm(
                full_messages, reason=f"main iter {iterations}"
            )
            # Non-streaming call to detect tool calls
            with tracking_context(phase="main"):
                result = await self.provider.complete(
                    messages=full_messages,
                    tools=tool_defs,
                    temperature=self.config.temperature,
                )
            self._mark_force_compact_after_fallback("main")
            response = result.message
            response.tool_calls = self._dedupe_tool_calls(response.tool_calls)
            invalid_tool_args = _invalid_tool_call_arguments(response)
            if invalid_tool_args:
                logger.warning(
                    "Session '%s' iter %d/%d — invalid tool_call arguments: %s",
                    self.session_id,
                    iterations,
                    self.config.max_iterations,
                    "; ".join(invalid_tool_args),
                )
                full_messages.append(Message(role="assistant", content=response.content))
                full_messages.append(Message(role="user", content=_NUDGE_INVALID_TOOL_ARGS))
                continue
            self._append_reasoning(result.reasoning_content)
            # Ghost turns (empty content, no tool_calls) are NOT added to the
            # cross-turn history: persisting them teaches the model that
            # empty replies are normal. We still append to ``full_messages``
            # so the in-flight loop sees a coherent trace.
            self._remember_assistant(
                response,
                reasoning=result.reasoning_content,
                context=f"iter {iterations}",
            )
            full_messages.append(response)
            if self._observer:
                self._observer.on_llm_response(
                    response.content or "", response.tool_calls or []
                )

            if not response.tool_calls:
                content = response.content or ""
                if (
                    not empty_nudged
                    and not content.strip()
                    and tool_defs
                    and tools_executed == 0
                ):
                    empty_nudged = True
                    logger.warning(
                        "Session '%s' — empty first-pass response at iter %d "
                        "(no content, no tool_call, reasoning=%dch). Nudging once.",
                        self.session_id,
                        iterations,
                        len((result.reasoning_content or "").strip()),
                    )
                    full_messages.append(Message(role="user", content=_NUDGE_EMPTY_REPLY))
                    continue
                # Claimed an action (house, reminder, memory) that never ran:
                # retry once with the tool still available, then flag the reply
                # so the user is not told the light is off when it is not.
                claim = _unbacked_claim(content, tool_defs, backed_claims)
                if claim is not None:
                    if not claim_nudged:
                        claim_nudged = True
                        logger.warning(
                            "Session '%s' — %s action claimed without executing it; nudging",
                            self.session_id,
                            claim.name,
                        )
                        full_messages.append(Message(role="user", content=claim.nudge))
                        continue
                    logger.warning(
                        "Session '%s' — %s action still claimed without executing it; flagging reply",
                        self.session_id,
                        claim.name,
                    )
                    content = f"{content}{claim.warning}"
                # Some small local models announce "voy a reintentar / I'll call X"
                # in plain text without emitting the structured tool_call. Nudge
                # once only before any tool has actually run. After a tool result,
                # a normal synthesis often mentions the tool name ("he ejecutado
                # skill_shell..."), and treating that as intent causes loops.
                if (
                    not nudged
                    and content
                    and tool_defs
                    and tools_executed == 0
                    and not _is_asking_user(content)
                    and _TOOL_INTENT_RE.search(content)
                ):
                    nudged = True
                    logger.info(
                        "Session '%s' — model announced a tool call without emitting it; nudging once",
                        self.session_id,
                    )
                    # Safety net for tool gating: if the announced intent maps to
                    # a group we did not send this turn (the heuristic missed it),
                    # enable that group now so the retry can actually emit the call.
                    if self._gating_enabled:
                        missing = select_groups(content) - active_groups
                        if missing:
                            for g in missing:
                                self._sticky_groups[g] = self._sticky_turns + 1
                            active_groups |= missing
                            tool_defs = self._filter_tool_defs(active_groups)
                            logger.info(
                                "Session '%s' — tool gating: enabling missed group(s) %s after announce",
                                self.session_id,
                                ", ".join(sorted(missing)),
                            )
                    full_messages.append(Message(role="user", content=_NUDGE_ANNOUNCED_CALL))
                    continue
                # Post-tool degeneration recovery: model already ran at least
                # one tool this turn and now returned NOTHING usable (empty
                # content, no tool_call). Force one synthesis pass without
                # tools so the user gets an answer built from what was
                # already fetched, instead of a silent "(no he podido…)".
                if (
                    not content.strip()
                    and iterations > 1
                    and not degeneration_recovered
                ):
                    degeneration_recovered = True
                    logger.warning(
                        "Session '%s' — post-tool degeneration at iter %d "
                        "(empty content, no tool_call, reasoning=%dch). "
                        "Forcing synthesis pass.",
                        self.session_id,
                        iterations,
                        len((result.reasoning_content or "").strip()),
                    )
                    yield await self._forced_answer(
                        full_messages,
                        _FORCE_SYNTHESIS,
                        phase="degeneration_recovery",
                        context="degeneration recovery",
                    )
                    return
                # Scaffolding-leak recovery: after a nudge this turn, the model
                # sometimes replies *to the nudge* — apologizing and narrating
                # its own tool/plan process instead of answering ("lo siento,
                # tienes razón... no necesito más llamadas a herramientas").
                # Never send that meta-commentary; force one clean synthesis
                # pass so the user gets an actual answer.
                if (
                    nudged
                    and not scaffold_recovered
                    and _is_scaffolding_leak(content)
                ):
                    scaffold_recovered = True
                    logger.warning(
                        "Session '%s' — scaffolding-leak reply after nudge at iter %d; "
                        "forcing clean synthesis pass",
                        self.session_id,
                        iterations,
                    )
                    yield await self._forced_answer(
                        full_messages,
                        _FORCE_CLEAN_ANSWER,
                        phase="scaffold_recovery",
                        context="scaffold recovery",
                    )
                    return
                # Final answer — yield it token by token (character-level).
                content = _final_text(content, result.reasoning_content)
                if self._observer:
                    self._observer.on_done(content)
                yield content
                return

            # Log tool calls at INFO level for observability
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

            loop_reason = loops.observe(response.tool_calls)
            if loop_reason:
                logger.warning(
                    "Session '%s' — loop detected: %s. Forcing final answer.",
                    self.session_id,
                    loop_reason,
                )
                yield await self._forced_answer(
                    full_messages,
                    _FORCE_BREAK_LOOP,
                    phase="loop_break",
                    context="loop break",
                )
                return

            for tc, tool_result in await self._execute_tool_calls(
                response.tool_calls, full_messages
            ):
                tools_executed += 1
                if not _tool_failed(tool_result):
                    tool = self.tool_registry._tools.get(tc.function.name)
                    backed_claims.update(g.name for g in _CLAIM_GUARDS if g.done_by(tool))

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
        msg = (
            "He llegado al máximo de pasos sin terminar la tarea. "
            "Prueba a pedírmelo de forma más concreta o en partes."
        )
        max_msg = Message(role="assistant", content=msg)
        self._messages.append(max_msg)
        self._persist_message(max_msg)
        yield msg

    def _append_reasoning(self, reasoning: str | None) -> None:
        """Accumulate this turn's reasoning across LLM calls (``last_reasoning``)."""
        if reasoning:
            self.last_reasoning = (
                (self.last_reasoning + "\n\n" if self.last_reasoning else "") + reasoning
            )

    async def _forced_answer(
        self, full_messages: list[Message], instruction: str, *, phase: str, context: str
    ) -> str:
        """Recovery pass: append ``instruction`` and ask once WITHOUT tools, so
        the model must answer in plain text from what it already has."""
        full_messages.append(Message(role="user", content=instruction))
        with tracking_context(phase=phase):
            full_messages = await self._compact_before_llm(full_messages, reason=phase)
            result = await self.provider.complete(
                messages=full_messages,
                tools=None,  # no tools — force text response
                temperature=self.config.temperature,
            )
        self._mark_force_compact_after_fallback(phase)
        forced = result.message
        self._append_reasoning(result.reasoning_content)
        self._remember_assistant(forced, reasoning=result.reasoning_content, context=context)
        content = _final_text(forced.content, result.reasoning_content)
        if self._observer:
            self._observer.on_done(content)
        return content

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall], full_messages: list[Message]
    ) -> list[tuple[ToolCall, str]]:
        """Run tool calls in parallel and record each result as a tool message
        (history, persistence and ``full_messages``). Returns the successful
        (call, result) pairs; exceptions are logged and skipped."""

        async def _exec_tool(tc: ToolCall) -> tuple[ToolCall, str]:
            if self._observer:
                self._observer.on_tool_start(tc.function.name, tc.function.arguments)
            r = await self.tool_registry.execute(tc.function.name, tc.function.arguments)
            if self._observer:
                self._observer.on_tool_end(tc.function.name, r)
            return tc, r

        exec_results = await asyncio.gather(
            *[_exec_tool(tc) for tc in tool_calls],
            return_exceptions=True,
        )
        done: list[tuple[ToolCall, str]] = []
        for entry in exec_results:
            if isinstance(entry, BaseException):
                logger.error("Session '%s' — tool error: %s", self.session_id, entry)
                if self._observer:
                    self._observer.on_tool_error("?", str(entry))
                continue
            tc, result = entry
            tool_msg = Message(role="tool", tool_call_id=tc.id, content=result)
            self._messages.append(tool_msg)
            self._persist_message(tool_msg)
            full_messages.append(tool_msg)
            self._mark_group_used(tc.function.name)
            logger.info(
                "Session '%s' — tool '%s'(%s) → %d chars: %s",
                self.session_id,
                tc.function.name,
                _truncate(tc.function.arguments, 200),
                len(result),
                _truncate(result, 200),
            )
            done.append((tc, result))
        return done

    async def send_sync(self, user_input: str) -> str:
        """Send a user message and return the complete response (non-streaming)."""
        result = ""
        async for chunk in self.send(user_input):
            result += chunk
        return result

    # ------------------------------------------------------------------
    # Chain mode — decompose complex queries into plan → execute → synthesize
    # ------------------------------------------------------------------

    # Patterns that suggest a message contains multiple instructions
    _MULTI_INSTRUCTION_RE = re.compile(
        r"(\d+[\.\)]\s)"  # numbered list: "1. ... 2. ..."
        r"|(\by\s+(luego|después|además|también)\b)"  # conjunctions
        r"|(\band\s+(then|also|additionally)\b)"
        r"|(\bprimero\b.*\bdespu[eé]s\b)"  # "primero...después"
        r"|(\bfirst\b.*\bthen\b)",
        re.IGNORECASE,
    )

    async def warm_up(self) -> None:
        """Prefill the provider's prompt cache with this session's prefix.

        Sends system prompt + history + the always-on tools with a throwaway
        user message, so the next real turn only processes its new tokens.
        Nothing is persisted. Only helps when the turn's tool set matches the
        always-on one (the usual case for short commands).
        """
        tools: list[ToolDefinition] | None = None
        if self.tool_registry._tools:
            if self._gating_enabled:
                tools = [
                    t
                    for t in self.tool_registry._tools.values()
                    if t.group is None or t.group in self._always_on_groups
                ] or None
            else:
                tools = list(self.tool_registry._tools.values())
        messages = self._get_full_messages()
        messages.append(Message(role="user", content="ok"))
        with tracking_context(phase="warmup"):
            await self.provider.warm_up(messages, tools)

    def _is_complex_query(self, user_input: str) -> bool:
        """Heuristic: decide if a query should use chain mode.

        A query is considered complex if:
        - It exceeds the character threshold AND
        - It contains patterns suggesting multiple instructions/steps
        """
        if not self._root_config:
            return False
        chain_cfg = self._root_config.sessions.chain
        if chain_cfg.mode == "always":
            return True
        if chain_cfg.mode == "never":
            return False
        # auto mode
        if len(user_input) < chain_cfg.complexity_threshold:
            return False
        return bool(self._MULTI_INSTRUCTION_RE.search(user_input))

    async def send_chained(self, user_input: str) -> AsyncIterator[str]:
        """Decompose a complex query into plan → execute → synthesize.

        Phase 1 (Plan): Ask the LLM to analyze the query and produce a step-by-step
                        plan of which tools to use and in what order. No tools available.
        Phase 2 (Execute): Send the plan + original query with tools enabled.
                          The LLM follows its own plan.
        Phase 3 (Synthesize): Ask the LLM to summarize results for the user. No tools.
        """
        logger.info(
            "Session '%s' — using chain mode for complex query (%d chars)",
            self.session_id, len(user_input),
        )

        tool_defs, _active_groups = self._turn_tool_defs(user_input)
        tool_names = (
            ", ".join(t.name for t in tool_defs) if tool_defs else "none"
        )

        # --- Phase 1: Plan (no tools) ---
        plan_prompt = (
            f"Analiza la siguiente petición del usuario y crea un plan paso a paso "
            f"para resolverla. Indica qué herramientas usar y en qué orden.\n\n"
            f"Herramientas disponibles: {tool_names}\n\n"
            f"Petición del usuario:\n{user_input}\n\n"
            f"Responde SOLO con el plan, sin ejecutar nada."
        )

        plan_messages = self._get_full_messages()
        plan_messages.append(Message(role="user", content=plan_prompt))

        current_session_id.set(self.session_id)
        current_conversation_id.set(self.conversation_id or "")
        current_trace_id.set(new_trace_id())

        with tracking_context(phase="chain_plan"):
            plan_messages = await self._compact_before_llm(
                plan_messages, reason="chain_plan"
            )
            plan_result = await self.provider.complete(
                messages=plan_messages,
                tools=None,  # no tools in planning phase
                temperature=self.config.temperature,
            )
        self._mark_force_compact_after_fallback("chain_plan")
        plan_text = plan_result.message.content or ""
        logger.info(
            "Session '%s' — chain plan: %s", self.session_id, plan_text[:200]
        )

        # --- Phase 2: Execute (with tools, plan as context) ---
        execute_prompt = (
            f"Ejecuta el siguiente plan para responder al usuario. "
            f"Usa las herramientas según el plan.\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Petición original del usuario:\n{user_input}"
        )

        # Use the normal send() flow which handles tool-calling loops
        # We inject the execute prompt as the user message
        user_msg = Message(role="user", content=execute_prompt)
        self._messages.append(user_msg)
        self._persist_message(user_msg)

        full_messages = self._get_full_messages()
        execution_output = ""
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1
            with tracking_context(phase="chain_execute"):
                full_messages = await self._compact_before_llm(
                    full_messages, reason=f"chain_execute iter {iterations}"
                )
                result = await self.provider.complete(
                    messages=full_messages,
                    tools=tool_defs,
                    temperature=self.config.temperature,
                )
            self._mark_force_compact_after_fallback("chain_execute")
            response = result.message
            self._append_reasoning(result.reasoning_content)
            self._messages.append(response)
            self._persist_message(response, reasoning=result.reasoning_content)
            full_messages.append(response)

            if not response.tool_calls:
                execution_output = response.content or ""
                break

            await self._execute_tool_calls(response.tool_calls, full_messages)

        # --- Phase 3: Synthesize (no tools) ---
        synth_prompt = (
            f"Resume de forma clara y útil los resultados obtenidos para el usuario. "
            f"Responde directamente a su petición original:\n{user_input}\n\n"
            f"Resultados de la ejecución:\n{execution_output[:3000]}"
        )
        synth_messages = self._get_full_messages()
        synth_messages.append(Message(role="user", content=synth_prompt))

        with tracking_context(phase="chain_synthesize"):
            synth_messages = await self._compact_before_llm(
                synth_messages, reason="chain_synthesize"
            )
            synth_result = await self.provider.complete(
                messages=synth_messages,
                tools=None,
                temperature=self.config.temperature,
            )
        self._mark_force_compact_after_fallback("chain_synthesize")
        final_output = (
            _final_text(synth_result.message.content, synth_result.reasoning_content)
            or execution_output
        )

        # Store the synthesis as the final assistant message
        synth_msg = Message(role="assistant", content=final_output)
        self._messages.append(synth_msg)
        self._persist_message(synth_msg)

        yield final_output

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
        if self._access is not None:
            access_note = self._access.allowed_summary()
            if access_note:
                system_content = f"{system_content}\n\n{access_note}"
        return system_content

    def _get_full_messages(self) -> list[Message]:
        """Build the full messages list including system prompt."""
        system_content = self._build_system_content()
        messages: list[Message] = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.extend(self._messages)
        return messages

    # ------------------------------------------------------------------
    # Conditional tool loading
    # ------------------------------------------------------------------

    def _filter_tool_defs(
        self, active_groups: set[str]
    ) -> list[ToolDefinition] | None:
        """Tools whose group is always-on (None) or in ``active_groups``."""
        selected = [
            t
            for t in self.tool_registry._tools.values()
            if t.group is None or t.group in active_groups
        ]
        if self._compactor:
            self._compactor.set_tool_overhead(selected)
        return selected or None

    def _turn_active_groups(self, user_input: str) -> set[str]:
        """Conditional tool groups to enable this turn.

        Combines (a) groups triggered by the message text, (b) groups still
        within their sticky window from recent use, and (c) on a bare
        confirmation ("sí, hazlo"), the groups active on the previous turn —
        so an action proposed last turn can still be executed.
        """
        # Expire one turn from each sticky group before recomputing.
        self._sticky_groups = {
            g: n - 1 for g, n in self._sticky_groups.items() if n - 1 > 0
        }
        active = (
            select_groups(user_input)
            | set(self._sticky_groups)
            | self._always_on_groups
        )
        if _is_user_confirmation(user_input):
            active |= self._last_active_groups
        self._last_active_groups = set(active)
        return active

    def _turn_tool_defs(
        self, user_input: str
    ) -> tuple[list[ToolDefinition] | None, set[str]]:
        """Return (tool_defs, active_groups) for this turn.

        When gating is disabled or there are no tools, returns the full set
        (or None) and treats all known groups as active so later expansion
        is a no-op.
        """
        if not self.tool_registry._tools:
            if self._compactor:
                self._compactor.set_tool_overhead(None)
            return None, set()
        if not self._gating_enabled:
            all_tools = list(self.tool_registry._tools.values())
            if self._compactor:
                self._compactor.set_tool_overhead(all_tools)
            return all_tools, all_groups()
        active = self._turn_active_groups(user_input)
        return self._filter_tool_defs(active), active

    def _mark_group_used(self, tool_name: str) -> None:
        """Refresh the sticky window for the group of an executed tool.

        Set to ``sticky_turns + 1`` because ``_turn_active_groups`` decrements
        every counter at the start of the next turn; this leaves the group
        active for exactly ``sticky_turns`` follow-up turns.
        """
        if not self._gating_enabled:
            return
        tool = self.tool_registry._tools.get(tool_name)
        if tool and tool.group:
            self._sticky_groups[tool.group] = self._sticky_turns + 1

    async def _compact_before_llm(
        self, full_messages: list[Message], *, reason: str
    ) -> list[Message]:
        """Compact persisted history before an LLM call if the local window is at risk."""
        if not self._compactor:
            return full_messages

        system_content = self._build_system_content()
        force = self._force_compact_next
        if not force and not self._compactor.should_compact(
            self._messages, system_content
        ):
            return full_messages

        self._force_compact_next = False
        persisted_full = self._get_full_messages()
        extra_messages: list[Message] = []
        if (
            len(full_messages) >= len(persisted_full)
            and full_messages[: len(persisted_full)] == persisted_full
        ):
            extra_messages = list(full_messages[len(persisted_full) :])

        logger.info(
            "Session '%s' — %s compaction before LLM call (%s)",
            self.session_id,
            "forced" if force else "preventive",
            reason,
        )
        result = await self._compactor.compact(
            self._messages,
            system_content,
            self.provider,
            f"{self.session_id}.{self.conversation_id}" if self.conversation_id else self.session_id,
            force=force,
        )
        if result.compacted:
            self._apply_compaction(result)
            return self._get_full_messages() + extra_messages
        return full_messages

    def _mark_force_compact_after_fallback(self, phase: str) -> None:
        """Force a compaction before the next local attempt after fallback was used.

        The intent is to give the local model a smaller context on the next
        attempt when the escalation was plausibly caused by context overflow.
        But a fallback can also fire on a low-context degeneration (e.g. an
        empty think-only completion at 46% of the window), in which case
        compacting is pure overhead. Only force it when context is actually
        under pressure — reuse the compactor's own trigger.
        """
        if not self._compactor:
            return
        idx = getattr(self.provider, "last_used_provider_index", None)
        if not isinstance(idx, int) or idx <= 0:
            return
        used = getattr(self.provider, "last_used_provider", None)
        if not self._compactor.should_compact(
            self._messages, self._build_system_content()
        ):
            logger.info(
                "Session '%s' — fallback provider used in phase '%s' (%s); "
                "context below trigger, skipping forced compaction",
                self.session_id,
                phase,
                used,
            )
            return
        self._force_compact_next = True
        logger.info(
            "Session '%s' — fallback provider used in phase '%s' (%s); "
            "forcing compaction before next LLM call",
            self.session_id,
            phase,
            used,
        )

    def _apply_compaction(self, result) -> None:
        """Apply compaction result: archive, rewrite JSONL, update messages."""
        if result.archived_path:
            logger.info("Archived pre-compaction messages to %s", result.archived_path)

        self._messages = result.new_messages
        if self._rewrite_persisted():
            logger.info(
                "Rewrote session JSONL after compaction (%d messages)",
                len(self._messages),
            )

    def _rewrite_persisted(self) -> bool:
        """Write the in-memory history over the session JSONL."""
        if not self._persist_path:
            return False
        try:
            tmp = self._persist_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for msg in self._messages:
                    f.write(json.dumps(msg.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
            tmp.replace(self._persist_path)
            return True
        except Exception as e:
            logger.warning("Failed to rewrite session JSONL %s: %s", self._persist_path, e)
            return False

    def trim_history(
        self,
        *,
        max_user_turns: int,
        keep_user_turns: int,
        stale_tool_turns: int,
        stub_min_chars: int = 300,
    ) -> dict | None:
        """Shrink the stored history in one go; meant for idle time.

        - More than ``max_user_turns`` user turns: archive everything before
          the last ``keep_user_turns`` (the gap between both numbers means
          this happens once in a while, not every turn).
        - Tool results older than the last ``stale_tool_turns`` user turns
          (and longer than ``stub_min_chars``) are replaced by a short note:
          they are stale snapshots (e.g. the whole home state) the model must
          re-query anyway. The assistant's replies, which carry what was
          concluded from them, stay.

        Every change rewrites the prompt prefix, so callers should run this
        when nobody is waiting and warm the prompt cache afterwards.
        Returns what changed, or None if nothing did.
        """
        users = [i for i, m in enumerate(self._messages) if m.role == "user"]
        dropped: list[Message] = []
        if keep_user_turns > 0 and len(users) > max(max_user_turns, keep_user_turns):
            cut = users[-keep_user_turns]
            dropped = self._messages[:cut]
            self._messages = self._messages[cut:]
            users = [i for i, m in enumerate(self._messages) if m.role == "user"]

        stubbed = saved = 0
        if stale_tool_turns > 0 and len(users) > stale_tool_turns:
            boundary = users[-stale_tool_turns]
            for i in range(boundary):
                msg = self._messages[i]
                content = msg.content or ""
                if msg.role != "tool" or len(content) <= stub_min_chars or content.startswith(_STALE_TOOL_PREFIX):
                    continue
                note = (
                    f'{_STALE_TOOL_PREFIX}"resultado de una consulta antigua ({len(content)} '
                    'caracteres) omitido; si hace falta el dato actual, vuelve a llamar a '
                    'la herramienta"}'
                )
                self._messages[i] = msg.model_copy(update={"content": note})
                stubbed += 1
                saved += len(content) - len(note)

        if not dropped and not stubbed:
            return None
        archived = self._archive_trimmed(dropped) if dropped else None
        self._rewrite_persisted()
        stats = {
            "dropped_messages": len(dropped),
            "stubbed_tool_results": stubbed,
            "chars_saved": saved + sum(len(m.content or "") for m in dropped),
            "archived": str(archived) if archived else None,
        }
        logger.info("Session '%s' — history trimmed: %s", self.session_id, stats)
        return stats

    def _archive_trimmed(self, messages: list[Message]) -> Path | None:
        if not self._persist_path:
            return None
        from datetime import datetime

        archive_dir = self._persist_path.parent / "archive"
        if self._compactor is not None and self._compactor._config.archive_dir:
            archive_dir = Path(self._compactor._config.archive_dir).expanduser()
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = archive_dir / f"{self._persist_path.stem}.{stamp}.trimmed.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
            return path
        except Exception as e:
            logger.warning("Could not archive trimmed history: %s", e)
            return None
