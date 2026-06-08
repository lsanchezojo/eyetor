"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

from eyetor.chat.tool_gating import KNOWN_GROUPS, select_groups
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.tracking.context import (
    current_session_id,
    current_trace_id,
    new_trace_id,
    tracking_context,
)

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


# Patterns stripped before scanning for question marks in `_is_asking_user`.
# Without these, a '?' inside a URL query string (e.g. `?item=10523`) or a
# code example makes the heuristic misclassify an announcement as a question
# and suppresses the announce-without-call nudge.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _is_asking_user(text: str) -> bool:
    """True if the model's message is a request for user input.

    Fenced/inline code and URLs are stripped before scanning for '?'/'¿' so
    a query string like ``?item=10523`` or a code example doesn't count as
    a question to the user. The ``_ASK_USER_MARKERS`` check still runs on
    the original text because those are prose phrases.
    """
    if not text:
        return False
    cleaned = _FENCED_CODE_RE.sub("", text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    if "?" in cleaned or "¿" in cleaned:
        return True
    lowered = text.lower()
    return any(m in lowered for m in _ASK_USER_MARKERS)


_CONFIRMATION_RE = re.compile(
    r"^(s[ií]|ok|dale|hazlo|adelante|confirmo|confirma|vale|"
    r"venga|va|claro|por supuesto|yes|yeah|yep|sure|go ahead|do it"
    r")[\s.!,]*$",
    re.IGNORECASE,
)


def _is_user_confirmation(text: str) -> bool:
    """True if ``text`` is a short affirmative confirmation."""
    return bool(_CONFIRMATION_RE.match(text.strip()))


_ACTION_PROPOSAL_RE = re.compile(
    r"\b("
    r"comando|command|herramienta|tool|ejecut\w*|lanz\w*|corr\w*|"
    r"run|execute|launch|dispatch|cerr\w*|abr\w*|borr\w*|elimin\w*|"
    r"instal\w*|reinici\w*|reload"
    r")\b",
    re.IGNORECASE,
)


def _last_assistant_proposed_action(messages: list[Message]) -> bool:
    """True if the last assistant turn asked to confirm an actionable step."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            content = msg.content or ""
            return (
                bool(content.strip())
                and _is_asking_user(content)
                and bool(_ACTION_PROPOSAL_RE.search(content))
            )
        if msg.role == "user":
            return False
    return False


def _is_ghost_assistant(msg: Message) -> bool:
    """True if ``msg`` is an empty assistant turn that should not be remembered.

    Small local models occasionally collapse mid-turn and produce an
    assistant message with no ``content`` and no ``tool_calls`` (the whole
    output ended up in the reasoning/think channel, or was an immediate
    EOS). Persisting these turns poisons future prompts — the model sees
    "the previous assistant reply was empty" and mimics that pattern.
    """
    if msg.role != "assistant":
        return False
    if msg.tool_calls:
        return False
    return not (msg.content or "").strip()


def _final_text(content: str | None, reasoning: str | None) -> str:
    """Pick the user-facing answer.

    Only ``content`` is user-facing — ``reasoning`` is the model's internal
    scratchpad (``<think>`` channel) and may contain tool-call drafts,
    monologue, or other noise that degrades response quality if shown raw.
    The ``reasoning`` argument is accepted for call-site symmetry but
    deliberately not used as a fallback. Channels show their own
    "no response, retry" message when content is empty.
    """
    del reasoning  # intentionally unused
    return (content or "").strip()


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string for log output, collapsing whitespace."""
    if s is None:
        return ""
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(+{len(s) - max_len})"


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Jaccard threshold for the "soft-loop" guard: three consecutive same-tool
# calls whose token-bags pairwise overlap above this ratio are treated as a
# loop, catching cases the exact-match check misses (same core, varying
# satellite words — see _tool_call_bag docstring).
_LOOP_JACCARD_THRESHOLD = 0.6


def _tool_call_bag(name: str, arguments: str) -> tuple[str, frozenset[str]]:
    """Return (tool_name, lowercase token-bag) for a tool call.

    Used by both the exact-match signature (below) and the Jaccard soft-loop
    check. Small models often permute the same keywords into dozens of
    "different" queries; this reduces them to a comparable bag.
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        args = arguments

    def walk(obj) -> list[str]:
        if isinstance(obj, str):
            return [t.lower() for t in _TOKEN_RE.findall(obj)]
        if isinstance(obj, dict):
            out: list[str] = []
            for v in obj.values():
                out.extend(walk(v))
            return out
        if isinstance(obj, list):
            out = []
            for v in obj:
                out.extend(walk(v))
            return out
        return []

    return name, frozenset(walk(args))


def _normalize_tool_call(name: str, arguments: str) -> str:
    """Collapse a tool call to a sorted-token signature for exact-match loop
    detection. Captures pure permutations (``a b c`` ≡ ``c a b``).
    """
    _, tokens = _tool_call_bag(name, arguments)
    return f"{name}:{','.join(sorted(tokens))}"


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


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
        self.last_reasoning: str | None = None  # Accumulated reasoning from the latest send() turn
        self._force_compact_next = False

        # Conditional tool loading (see eyetor.chat.tool_gating). Token-heavy
        # tool groups are only sent on turns whose text triggers them, plus a
        # short sticky window so follow-ups ("cancélalo", "sí") still see them.
        self._gating_enabled = False
        self._sticky_turns = 2
        if root_config is not None:
            gating_cfg = root_config.sessions.tool_gating
            self._gating_enabled = gating_cfg.enabled
            self._sticky_turns = gating_cfg.sticky_turns
        self._sticky_groups: dict[str, int] = {}  # group → turns it stays active
        self._last_active_groups: set[str] = set()  # for confirmation inheritance

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
            kept = [m for m in raw if not _is_ghost_assistant(m)]
            dropped = len(raw) - len(kept)
            self._messages.extend(kept)
            if dropped:
                logger.warning(
                    "Dropped %d ghost assistant message(s) from %s (empty content "
                    "+ no tool_calls). Rewriting JSONL.",
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
            full_messages.append(
                Message(
                    role="system",
                    content=(
                        "El usuario acaba de confirmar la acción que propusiste. "
                        "DEBES ejecutarla AHORA emitiendo la tool_call estructurada "
                        "correspondiente en esta misma respuesta. No respondas solo "
                        "con texto: si hay una herramienta adecuada, llama a la "
                        "herramienta con los parámetros correctos."
                    ),
                )
            )
        iterations = 0
        recent_calls: list[str] = []  # normalized sig for exact-match detection
        recent_bags: list[list[tuple[str, frozenset[str]]]] = []  # for Jaccard
        max_repeat = 3  # max consecutive identical tool calls before forcing answer
        nudged = False  # allow at most one "announce-without-call" nudge per turn
        empty_nudged = False  # allow one retry for immediate empty/no-tool replies
        degeneration_recovered = False  # one-shot post-tool synthesis fallback
        tools_executed = 0

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
            if result.reasoning_content:
                self.last_reasoning = (
                    (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                    + result.reasoning_content
                )
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
                    full_messages.append(
                        Message(
                            role="user",
                            content=(
                                "No has emitido ni respuesta visible ni tool_call estructurada. "
                                "Continúa ahora: si necesitas actuar en el ordenador, emite la "
                                "tool_call correcta; si no necesitas herramientas, responde al "
                                "usuario directamente en español. No dejes la respuesta vacía."
                            ),
                        )
                    )
                    continue
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
                    full_messages.append(
                        Message(
                            role="user",
                            content=(
                                "Sintetiza ahora una respuesta clara y breve al usuario "
                                "usando ÚNICAMENTE la información obtenida en las herramientas "
                                "anteriores. No llames más herramientas. Responde en lenguaje "
                                "natural, en español."
                            ),
                        )
                    )
                    with tracking_context(phase="degeneration_recovery"):
                        full_messages = await self._compact_before_llm(
                            full_messages, reason="degeneration_recovery"
                        )
                        result = await self.provider.complete(
                            messages=full_messages,
                            tools=None,
                            temperature=self.config.temperature,
                        )
                    self._mark_force_compact_after_fallback("degeneration_recovery")
                    forced = result.message
                    if result.reasoning_content:
                        self.last_reasoning = (
                            (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                            + result.reasoning_content
                        )
                    self._remember_assistant(
                        forced,
                        reasoning=result.reasoning_content,
                        context="degeneration recovery",
                    )
                    content = _final_text(forced.content, result.reasoning_content)
                    if self._observer:
                        self._observer.on_done(content)
                    yield content
                    return
                # Final answer — yield it token by token (character-level).
                content = _final_text(content, result.reasoning_content)
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

            # Loop detection — two layers:
            # 1. Exact match on normalized token-bag signature (catches pure
            #    keyword permutations: "a b c" ≡ "c a b").
            # 2. Jaccard soft-loop: N same-tool calls whose token bags
            #    pairwise overlap ≥ threshold (catches "same core + varying
            #    satellite words" — the real failure mode of small models).
            call_signatures = [
                _normalize_tool_call(tc.function.name, tc.function.arguments)
                for tc in response.tool_calls
            ]
            current_bags = [
                _tool_call_bag(tc.function.name, tc.function.arguments)
                for tc in response.tool_calls
            ]
            current_sig = "|".join(sorted(call_signatures))
            recent_calls.append(current_sig)
            recent_bags.append(current_bags)
            if len(recent_calls) > max_repeat:
                recent_calls = recent_calls[-max_repeat:]
                recent_bags = recent_bags[-max_repeat:]

            loop_reason: str | None = None
            if len(recent_calls) == max_repeat and len(set(recent_calls)) == 1:
                loop_reason = f"exact repetition of {call_names}"
            elif (
                len(recent_bags) == max_repeat
                and all(len(b) == 1 for b in recent_bags)
                and len({b[0][0] for b in recent_bags}) == 1
            ):
                # Soft loop: among the last N same-tool calls, require every
                # pair to overlap above threshold. This still catches true
                # repeated loops while allowing legitimate retries that change
                # strategy, parameters, or timeout after a failed attempt.
                bags = [b[0][1] for b in recent_bags]
                pairwise = [
                    _jaccard(bags[i], bags[j])
                    for i in range(len(bags))
                    for j in range(i + 1, len(bags))
                ]
                min_jaccard = min(pairwise) if pairwise else 0.0
                if pairwise and min_jaccard >= _LOOP_JACCARD_THRESHOLD:
                    loop_reason = (
                        f"{max_repeat} near-duplicate '{recent_bags[-1][0][0]}' "
                        f"calls (min Jaccard {min_jaccard:.2f} ≥ {_LOOP_JACCARD_THRESHOLD})"
                    )

            if loop_reason:
                logger.warning(
                    "Session '%s' — loop detected: %s. Forcing final answer.",
                    self.session_id,
                    loop_reason,
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
                with tracking_context(phase="loop_break"):
                    full_messages = await self._compact_before_llm(
                        full_messages, reason="loop_break"
                    )
                    result = await self.provider.complete(
                        messages=full_messages,
                        tools=None,  # no tools — force text response
                        temperature=self.config.temperature,
                    )
                self._mark_force_compact_after_fallback("loop_break")
                forced = result.message
                if result.reasoning_content:
                    self.last_reasoning = (
                        (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                        + result.reasoning_content
                    )
                self._remember_assistant(
                    forced,
                    reasoning=result.reasoning_content,
                    context="loop break",
                )
                content = _final_text(forced.content, result.reasoning_content)
                if self._observer:
                    self._observer.on_done(content)
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
                tools_executed += 1
                self._mark_group_used(tc.function.name)
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
            if result.reasoning_content:
                self.last_reasoning = (
                    (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                    + result.reasoning_content
                )
            self._messages.append(response)
            self._persist_message(response, reasoning=result.reasoning_content)
            full_messages.append(response)

            if not response.tool_calls:
                execution_output = response.content or ""
                break

            # Execute tool calls (same logic as send())
            async def _exec_tool(tc: ToolCall) -> tuple[ToolCall, str]:
                return tc, await self.tool_registry.execute(
                    tc.function.name, tc.function.arguments
                )

            exec_results = await asyncio.gather(
                *[_exec_tool(tc) for tc in response.tool_calls],
                return_exceptions=True,
            )
            for entry in exec_results:
                if isinstance(entry, BaseException):
                    logger.error("Session '%s' chain exec error: %s", self.session_id, entry)
                    continue
                tc, tool_result = entry
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=tool_result)
                self._messages.append(tool_msg)
                self._persist_message(tool_msg)
                full_messages.append(tool_msg)
                self._mark_group_used(tc.function.name)

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
        active = select_groups(user_input) | set(self._sticky_groups)
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
            return None, set()
        if not self._gating_enabled:
            return list(self.tool_registry._tools.values()), set(KNOWN_GROUPS)
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
            self.session_id,
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
                    "Schedule an automatic task. 3 'schedule' modes: "
                    "ONE-SHOT ('2026-04-16 09:00', 'next thursday at 9', 'tomorrow at 8'); "
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
                            "description": "The message sent to the agent when the task fires",
                        },
                        "schedule": {
                            "type": "string",
                            "description": (
                                "When to run. One-shot: '2026-04-16 09:00', "
                                "'next thursday at 9', 'tomorrow at 18:00'. "
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

        self.tool_registry.register(
            ToolDefinition(
                name="list_tasks",
                description="List all scheduled tasks with their next run time and status.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handle_list_tasks,
                group="scheduler",
            )
        )

        self.tool_registry.register(
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
