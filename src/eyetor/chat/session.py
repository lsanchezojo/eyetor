"""ChatSession — maintains conversation history and runs the agentic loop."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from eyetor.models.agents import AgentConfig, TurnBudget
from eyetor.models.messages import Message, ToolCall
from eyetor.models.tools import ToolRegistry, ToolDefinition
from eyetor.providers.base import BaseProvider
from eyetor.providers.tracking import current_session_id
from eyetor.utils.tool_calls import parse_textual_tool_calls, strip_textual_tool_calls

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


# Patterns stripped before scanning for question marks in `_is_asking_user`.
# Without these, a '?' inside a URL query string (e.g. `?item=10523`) or a
# code example makes the heuristic misclassify an announcement as a question
# and suppresses the announce-without-call nudge.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def _strip_textual_tool_calls(text: str) -> tuple[str, bool]:
    """Strip textual tool-call markup from model output.

    Returns (cleaned, had_markup). Cleaned output has markup removed and
    surrounding whitespace collapsed; had_markup is True if any block was
    stripped, so callers can log / take fallback action.
    """
    return strip_textual_tool_calls(text)


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

# Per-tool overrides (prefix match). Web search loops are the dominant failure
# mode for small models — they permute keywords endlessly when the answer
# isn't on the open web. A lower threshold trips the guard sooner.
_TOOL_LOOP_THRESHOLDS: dict[str, float] = {
    "skill_web_search": 0.4,
}

_KB_QUERY_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "que", "qué", "cual", "cuál", "sobre", "para", "por", "con", "en",
    "y", "o", "a", "me", "di", "dime", "explica", "busca", "consulta",
}


def _loop_threshold_for(tool_name: str) -> float:
    for prefix, threshold in _TOOL_LOOP_THRESHOLDS.items():
        if tool_name.startswith(prefix):
            return threshold
    return _LOOP_JACCARD_THRESHOLD


def _is_empty_search_result(result: str) -> bool:
    """True if a web-search tool result has no hits.

    The web-search skill prints a JSON array; empty means ``[]``. Be lenient
    with whitespace / wrapping objects (``{"results": []}``) since other
    backends might emit slightly different shapes.
    """
    if not result:
        return True
    stripped = result.strip()
    if stripped in ("[]", "{}", ""):
        return True
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        for key in ("results", "items", "hits"):
            if key in data and isinstance(data[key], list) and not data[key]:
                return True
    return False


def _reformulate_kb_query(query: str) -> str:
    """Cheap second-pass KB query for tiny models: keep discriminative terms."""
    tokens = [
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) > 2 and t not in _KB_QUERY_STOPWORDS
    ]
    deduped = list(dict.fromkeys(tokens))
    return " ".join(deduped[:8]) or query


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


def _filter_tool_defs(
    all_tools: list[ToolDefinition],
    allowlist: list[str] | None,
) -> list[ToolDefinition] | None:
    """Apply a route-scoped allowlist to the full tool registry.

    * ``None`` → return all tools verbatim (no filtering).
    * ``[]``   → return ``None`` so the provider is called with no tools.
    * list of patterns (exact name or fnmatch glob) → keep matches only.
    """
    if allowlist is None:
        return list(all_tools) if all_tools else None
    if not allowlist:
        return None
    keep: list[ToolDefinition] = []
    for td in all_tools:
        if any(fnmatch.fnmatchcase(td.name, pat) for pat in allowlist):
            keep.append(td)
    return keep or None


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

        # Per-turn budget: YAML (sessions.budget) overrides the AgentConfig
        # default so deployments can tune this without rebuilding the agent.
        if root_config is not None:
            self._budget = TurnBudget(
                max_tool_calls=root_config.sessions.budget.max_tool_calls,
                max_wall_seconds=root_config.sessions.budget.max_wall_seconds,
            )
        else:
            self._budget = config.budget

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

    def _profile(self, name: str):
        if not self._root_config:
            return None
        return getattr(self._root_config.profiles, name, None)

    def _profile_temperature(self, name: str, default: float) -> float:
        profile = self._profile(name)
        value = getattr(profile, "temperature", None) if profile else None
        return default if value is None else value

    def _profile_thinking(self, name: str, default: bool | None = None) -> bool | None:
        profile = self._profile(name)
        value = getattr(profile, "thinking", None) if profile else None
        return default if value is None else value

    def _profile_budget(self, name: str, fallback: TurnBudget) -> TurnBudget:
        profile = self._profile(name)
        if not profile:
            return fallback
        return TurnBudget(
            max_tool_calls=(
                fallback.max_tool_calls
                if profile.max_tool_calls is None
                else profile.max_tool_calls
            ),
            max_wall_seconds=(
                fallback.max_wall_seconds
                if profile.max_wall_seconds is None
                else profile.max_wall_seconds
            ),
        )

    async def _complete_with_profile(
        self,
        profile_name: str,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        default_temperature: float,
        default_thinking: bool | None = None,
    ):
        """Call the provider with optional profile overrides."""
        profile = self._profile(profile_name)
        temperature = self._profile_temperature(profile_name, default_temperature)
        thinking = self._profile_thinking(profile_name, default_thinking)
        targets: list[Any] = []

        def _collect_provider_targets(provider: Any) -> None:
            if provider in targets:
                return
            targets.append(provider)
            inner = getattr(provider, "_inner", None)
            if inner is not None:
                _collect_provider_targets(inner)
            for child in getattr(provider, "_providers", []) or []:
                _collect_provider_targets(child)

        _collect_provider_targets(self.provider)
        saved: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        if profile and (profile.extra_body or profile.options):
            for target in targets:
                saved.append((target, dict(target.extra_body), dict(target.options)))
                target.extra_body = {**target.extra_body, **profile.extra_body}
                target.options = {**target.options, **profile.options}
        try:
            return await self.provider.complete(
                messages=messages,
                tools=tools,
                temperature=temperature,
                thinking=thinking,
            )
        finally:
            for target, extra_body, options in saved:
                target.extra_body = extra_body
                target.options = options

    def _promote_textual_tool_calls(
        self,
        response: Message,
        tool_defs: list[ToolDefinition] | None,
    ) -> None:
        """Promote textual tool-call markup to structured calls when possible."""
        if response.tool_calls or not response.content or not tool_defs:
            return
        parsed = parse_textual_tool_calls(
            response.content,
            available_tool_names=[tool.name for tool in tool_defs],
        )
        if not parsed.had_markup:
            return
        response.content = parsed.cleaned_text or None
        if parsed.tool_calls:
            response.tool_calls = parsed.tool_calls
            logger.warning(
                "Session '%s' — recovered %d textual tool_call(s): %s",
                self.session_id,
                len(parsed.tool_calls),
                ", ".join(tc.function.name for tc in parsed.tool_calls),
            )
            return
        if parsed.unresolved_names:
            response.content = (
                "No he podido ejecutar la herramienta solicitada porque "
                "no está disponible o su nombre es ambiguo: "
                + ", ".join(parsed.unresolved_names)
            )
            logger.warning(
                "Session '%s' — ignored unresolved textual tool_call(s): %s",
                self.session_id,
                ", ".join(parsed.unresolved_names),
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

    async def send(
        self,
        user_input: str,
        *,
        allow_chain: bool = True,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Send a user message; yield streaming tokens from the assistant.

        Tool calls are executed silently. The final response is streamed.
        When chain mode is active and the query is complex, delegates to
        send_chained() automatically. Pass ``allow_chain=False`` to force
        single-turn execution (e.g. for photo handlers where the prompt is
        long but doesn't need decomposition).

        ``tools_override`` narrows the toolset for this turn only (used by
        the intent router to scope routes to relevant tools). See
        ``_filter_tool_defs`` for semantics.
        """
        if allow_chain and self._is_complex_query(user_input):
            async for chunk in self.send_chained(user_input, tools_override=tools_override):
                yield chunk
            return

        try:
            async for chunk in self._send_single_turn(user_input, tools_override=tools_override):
                yield chunk
        finally:
            self._squash_tool_messages()

    async def _send_single_turn(
        self,
        user_input: str,
        *,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Single-turn body of ``send()``. Wrapped so squash runs exactly once."""
        self.last_reasoning = None
        user_msg = Message(role="user", content=user_input)
        self._messages.append(user_msg)
        self._persist_message(user_msg)
        all_tools = list(self.tool_registry._tools.values())
        tool_defs = _filter_tool_defs(all_tools, tools_override)
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
        recent_calls: list[str] = []  # normalized sig for exact-match detection
        recent_bags: list[list[tuple[str, frozenset[str]]]] = []  # for Jaccard
        max_repeat = 3  # max consecutive identical tool calls before forcing answer
        nudged = False  # allow at most one "announce-without-call" nudge per turn
        empty_web_search_streak = 0  # consecutive web-search calls returning 0 hits
        kb_nudge_sent = False  # one-shot nudge per turn
        tool_calls_used = 0
        turn_start = time.monotonic()
        budget = self._profile_budget("tool_use", self._budget)

        while iterations < self.config.max_iterations:
            iterations += 1
            # Turn-budget guard — primary stopper, runs before max_iterations
            # catches anything. 0 on either field disables that specific budget.
            elapsed = time.monotonic() - turn_start
            over_calls = budget.max_tool_calls > 0 and tool_calls_used >= budget.max_tool_calls
            over_wall = budget.max_wall_seconds > 0 and elapsed >= budget.max_wall_seconds
            if over_calls or over_wall:
                reason = (
                    f"presupuesto agotado ({tool_calls_used} tool calls, "
                    f"{elapsed:.0f}s de {budget.max_wall_seconds}s)"
                )
                async for chunk in self._force_final_answer(
                    full_messages, user_input, reason=reason
                ):
                    yield chunk
                return
            if self._observer:
                self._observer.on_iteration(iterations)
            # When no tools are exposed (chat route, or no registry), the
            # call is pure conversation — disable reasoning so the local
            # thinking-mode model doesn't waste ~30 s per turn on a greeting.
            profile_name = "tool_use" if tool_defs else "chat"
            iter_thinking = None if tool_defs else False
            result = await self._complete_with_profile(
                profile_name,
                messages=full_messages,
                tools=tool_defs,
                default_temperature=self.config.temperature,
                default_thinking=iter_thinking,
            )
            response = result.message
            if result.reasoning_content:
                self.last_reasoning = (
                    (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                    + result.reasoning_content
                )

            self._promote_textual_tool_calls(response, tool_defs)
            self._messages.append(response)
            self._persist_message(response, reasoning=result.reasoning_content)
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
                cleaned, had_markup = _strip_textual_tool_calls(content)
                if had_markup:
                    logger.warning(
                        "Session '%s' — final answer contained textual tool-call markup; stripped.",
                        self.session_id,
                    )
                if not cleaned:
                    cleaned = (
                        "No he podido completar la consulta con las herramientas disponibles. "
                        "¿Puedes reformular la pregunta o darme más contexto?"
                    )
                if self._observer:
                    self._observer.on_done(cleaned)
                yield cleaned
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
                # Soft loop: among the last N same-tool calls, if ANY pair
                # overlaps ≥ threshold, the model is re-asking what it just
                # asked. Using ``max`` (not ``min``) is deliberate — small
                # models vary satellite tokens wildly, so min would never
                # trip; max catches "iter N is nearly iter N-1".
                bags = [b[0][1] for b in recent_bags]
                tool_name = recent_bags[-1][0][0]
                threshold = _loop_threshold_for(tool_name)
                pairwise = [
                    _jaccard(bags[i], bags[j])
                    for i in range(len(bags))
                    for j in range(i + 1, len(bags))
                ]
                if pairwise and max(pairwise) >= threshold:
                    loop_reason = (
                        f"{max_repeat} near-duplicate '{tool_name}' "
                        f"calls (max Jaccard {max(pairwise):.2f} ≥ {threshold})"
                    )

            if loop_reason:
                async for chunk in self._force_final_answer(
                    full_messages, user_input, reason=f"loop detectado ({loop_reason})"
                ):
                    yield chunk
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
            tool_calls_used += len(response.tool_calls)
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
                if tc.function.name.startswith("skill_web_search"):
                    if _is_empty_search_result(result):
                        empty_web_search_streak += 1
                    else:
                        empty_web_search_streak = 0

            # Nudge toward kb_search after two consecutive empty web searches.
            # Small models otherwise re-permute keywords endlessly when the
            # answer is in the local KB rather than on the open web.
            if (
                empty_web_search_streak >= 2
                and not kb_nudge_sent
                and any(t.name == "kb_search" for t in tool_defs or [])
            ):
                kb_nudge_sent = True
                logger.info(
                    "Session '%s' — %d consecutive empty web searches; nudging toward kb_search",
                    self.session_id,
                    empty_web_search_streak,
                )
                full_messages.append(
                    Message(
                        role="user",
                        content=(
                            "Las búsquedas web no están dando resultados. "
                            "Si la información puede estar en el knowledge base local, "
                            "usa kb_search en su lugar antes de seguir reintentando."
                        ),
                    )
                )

            # Intra-turn compaction: tool outputs can blow the context mid-loop
            # even when the turn started under the threshold. Re-check before
            # the next LLM call so we don't bust the window.
            if self._compactor:
                system_content = self._build_system_content()
                if self._compactor.should_compact(self._messages, system_content):
                    logger.info(
                        "Session '%s' — intra-turn compaction at iter %d",
                        self.session_id,
                        iterations,
                    )
                    result = await self._compactor.compact(
                        self._messages, system_content, self.provider, self.session_id
                    )
                    if result.compacted:
                        self._apply_compaction(result)
                        full_messages = self._get_full_messages()

        # Max iterations reached — last-resort safety net; budget/loop guards
        # should normally trip first. Force a final answer instead of bailing
        # with a canned English message so the user gets something useful.
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
        async for chunk in self._force_final_answer(
            full_messages,
            user_input,
            reason=f"max_iterations ({self.config.max_iterations}) alcanzado",
        ):
            yield chunk

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

    async def send_chained(
        self,
        user_input: str,
        *,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
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
        try:
            async for chunk in self._send_chained_body(user_input, tools_override=tools_override):
                yield chunk
        finally:
            self._squash_tool_messages()

    async def _send_chained_body(
        self,
        user_input: str,
        *,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
        all_tools = list(self.tool_registry._tools.values())
        tool_defs = _filter_tool_defs(all_tools, tools_override)
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

        plan_result = await self._complete_with_profile(
            "classifier",
            messages=plan_messages,
            tools=None,  # no tools in planning phase
            default_temperature=self.config.temperature,
            default_thinking=False,
        )
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

        current_session_id.set(self.session_id)
        full_messages = self._get_full_messages()
        execution_output = ""
        iterations = 0
        tool_calls_used = 0
        turn_start = time.monotonic()
        budget = self._profile_budget("tool_use", self._budget)

        while iterations < self.config.max_iterations:
            iterations += 1
            elapsed = time.monotonic() - turn_start
            over_calls = budget.max_tool_calls > 0 and tool_calls_used >= budget.max_tool_calls
            over_wall = budget.max_wall_seconds > 0 and elapsed >= budget.max_wall_seconds
            if over_calls or over_wall:
                logger.warning(
                    "Session '%s' — chain exec budget exhausted (%d calls, %.0fs); stopping.",
                    self.session_id, tool_calls_used, elapsed,
                )
                break
            result = await self._complete_with_profile(
                "tool_use",
                messages=full_messages,
                tools=tool_defs,
                default_temperature=self.config.temperature,
            )
            response = result.message
            self._promote_textual_tool_calls(response, tool_defs)
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
            tool_calls_used += len(response.tool_calls)
            for entry in exec_results:
                if isinstance(entry, BaseException):
                    logger.error("Session '%s' chain exec error: %s", self.session_id, entry)
                    continue
                tc, tool_result = entry
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=tool_result)
                self._messages.append(tool_msg)
                self._persist_message(tool_msg)
                full_messages.append(tool_msg)

        # --- Phase 3: Synthesize (no tools) ---
        synth_prompt = (
            f"Resume de forma clara y útil los resultados obtenidos para el usuario. "
            f"Responde directamente a su petición original:\n{user_input}\n\n"
            f"Resultados de la ejecución:\n{execution_output[:3000]}"
        )
        synth_messages = self._get_full_messages()
        synth_messages.append(Message(role="user", content=synth_prompt))

        synth_result = await self._complete_with_profile(
            "synthesis",
            messages=synth_messages,
            tools=None,
            default_temperature=self.config.temperature,
            default_thinking=False,
        )
        final_output = synth_result.message.content or execution_output
        cleaned, had_markup = _strip_textual_tool_calls(final_output)
        if had_markup:
            logger.warning(
                "Session '%s' — synthesis output contained textual tool-call markup; stripped.",
                self.session_id,
            )
        if not cleaned:
            cleaned = (
                "No he podido completar la consulta con las herramientas disponibles. "
                "¿Puedes reformular la pregunta o darme más contexto?"
            )

        # Store the synthesis as the final assistant message
        synth_msg = Message(role="assistant", content=cleaned)
        self._messages.append(synth_msg)
        self._persist_message(synth_msg)

        yield cleaned

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
        self._rewrite_persist_file(label="compaction")

    def _rewrite_persist_file(self, *, label: str = "rewrite") -> None:
        """Rewrite the JSONL to match current ``self._messages`` verbatim."""
        if not self._persist_path:
            return
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
                "Rewrote session JSONL after %s (%d messages)",
                label, len(self._messages),
            )
        except Exception as e:
            logger.warning("Failed to rewrite JSONL after %s: %s", label, e)

    def _squash_tool_messages(self) -> None:
        """Collapse raw tool-result content into compact snapshots.

        Run at the end of every turn so the next turn doesn't carry the raw
        tool output (PDF dumps, 6 kB JSON blobs) into the provider context.
        Pure structural rewrite — no LLM call — because per-result LLM
        condensation was exactly what regressed the local-provider path in
        the previous iteration. The snapshot keeps the tool_call_id pairing
        intact so OpenAI's message schema stays valid.
        """
        changed = False
        for i, msg in enumerate(self._messages):
            if msg.role != "tool":
                continue
            content = msg.content or ""
            if not content or content.startswith("[snapshot"):
                continue
            preview = _truncate(content, 300)
            snapshot = f"[snapshot · {len(content)} chars] {preview}"
            self._messages[i] = Message(
                role="tool",
                tool_call_id=msg.tool_call_id,
                content=snapshot,
            )
            changed = True
        if not changed:
            return
        logger.info(
            "Session '%s' — squashed tool messages (now %d msgs in history)",
            self.session_id, len(self._messages),
        )
        self._rewrite_persist_file(label="squash")

    # ------------------------------------------------------------------
    # KB 2-phase handler (research → synthesis)
    # ------------------------------------------------------------------

    async def send_kb_query(
        self,
        user_input: str,
        *,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Answer a KB question in two explicit phases.

        Phase 1 — research: the model is exposed to KB tools only (by
        default ``kb_search`` / ``kb_read`` / ``kb_list_sources``). Each raw
        tool result is *condensed* into a few bullets with a micro-LLM call
        before re-entering the context, so the model never drags 5 kB of raw
        PDF back into its next decision.

        Phase 2 — synthesis: the model is called once more with ``tools=None``
        and only the condensed bullets in context. This is the step that
        actually answers the user; SLMs handle it much more reliably than
        the generic "decide when to stop" loop.

        Budget is tighter than the generic ``TurnBudget``: 3 tool calls and
        60 s max, because research should be focused, not exhaustive.
        """
        try:
            async for chunk in self._send_kb_query_body(user_input, tools_override=tools_override):
                yield chunk
        finally:
            self._squash_tool_messages()

    async def _send_kb_query_body(
        self,
        user_input: str,
        *,
        tools_override: list[str] | None = None,
    ) -> AsyncIterator[str]:
        self.last_reasoning = None
        user_msg = Message(role="user", content=user_input)
        self._messages.append(user_msg)
        self._persist_message(user_msg)
        current_session_id.set(self.session_id)

        all_tools = list(self.tool_registry._tools.values())
        allow = (
            tools_override
            if tools_override is not None
            else ["kb_search", "kb_read", "kb_list_sources"]
        )
        kb_tools = _filter_tool_defs(all_tools, allow)
        if not kb_tools:
            msg = (
                "No hay herramientas de KB disponibles. Revisa la configuración "
                "de knowledge en config/default.yaml."
            )
            final = Message(role="assistant", content=msg)
            self._messages.append(final)
            self._persist_message(final)
            yield msg
            return

        research_system = self._build_system_content() + (
            "\n\n[Modo investigación KB]\n"
            "Dispones sólo de herramientas KB y un presupuesto estricto de 3 "
            "llamadas a tools. Úsalas para localizar y leer lo que necesites. "
            "Tras reunir la información, DEJA DE LLAMAR TOOLS; una síntesis "
            "posterior redactará la respuesta al usuario. No repitas búsquedas "
            "ni vuelvas a leer secciones ya consultadas."
        )
        research_messages: list[Message] = [
            Message(role="system", content=research_system),
            Message(role="user", content=user_input),
        ]
        scratchpad: list[str] = []
        read_doc_ids: set[int] = set()
        retried_empty_search = False

        budget = self._profile_budget("kb_research", self._budget)
        cfg_calls = budget.max_tool_calls if budget.max_tool_calls > 0 else 3
        cfg_wall = budget.max_wall_seconds if budget.max_wall_seconds > 0 else 60
        max_calls = min(cfg_calls, 3)
        max_wall = min(cfg_wall, 60)
        tool_calls_used = 0
        turn_start = time.monotonic()

        logger.info(
            "Session '%s' — KB 2-phase research start (budget %d calls, %ds)",
            self.session_id, max_calls, max_wall,
        )

        while tool_calls_used < max_calls:
            elapsed = time.monotonic() - turn_start
            if elapsed >= max_wall:
                logger.info(
                    "Session '%s' — KB research wall-time reached (%.0fs)",
                    self.session_id, elapsed,
                )
                break
            try:
                result = await self._complete_with_profile(
                    "kb_research",
                    messages=research_messages,
                    tools=kb_tools,
                    default_temperature=self.config.temperature,
                    default_thinking=False,
                )
            except Exception as exc:
                logger.warning(
                    "Session '%s' — KB research LLM error: %s",
                    self.session_id, exc,
                )
                break
            response = result.message
            self._promote_textual_tool_calls(response, kb_tools)
            if result.reasoning_content:
                self.last_reasoning = (
                    (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                    + result.reasoning_content
                )
            self._messages.append(response)
            self._persist_message(response, reasoning=result.reasoning_content)
            research_messages.append(response)

            if not response.tool_calls:
                break

            async def _exec(tc: ToolCall) -> tuple[ToolCall, str]:
                r = await self.tool_registry.execute(
                    tc.function.name, tc.function.arguments
                )
                return tc, r

            exec_results = await asyncio.gather(
                *[_exec(tc) for tc in response.tool_calls],
                return_exceptions=True,
            )
            for entry in exec_results:
                if isinstance(entry, BaseException):
                    logger.error(
                        "Session '%s' — KB tool error: %s",
                        self.session_id, entry,
                    )
                    continue
                tc, raw = entry
                tool_calls_used += 1
                # KB tools already bound their own output (kb_search snippets
                # ≤400 chars, kb_read sections ≤1800 chars). A further LLM
                # condensation call used to live here but burned 1 extra LLM
                # round-trip per tool call for marginal benefit — removed so
                # the local thinking-mode model doesn't pay that overhead.
                kept = raw[:2000] if len(raw) > 2000 else raw
                logger.info(
                    "Session '%s' — KB call %d/%d: %s(%s) → %d chars raw (kept %d)",
                    self.session_id, tool_calls_used, max_calls,
                    tc.function.name, _truncate(tc.function.arguments, 100),
                    len(raw), len(kept),
                )
                scratchpad.append(
                    f"[{tc.function.name} · {_truncate(tc.function.arguments, 80)}]\n{kept}"
                )
                tool_msg = Message(role="tool", tool_call_id=tc.id, content=kept)
                self._messages.append(tool_msg)
                self._persist_message(tool_msg)
                research_messages.append(tool_msg)

                if tc.function.name == "kb_read":
                    try:
                        read_doc_ids.add(int(json.loads(tc.function.arguments).get("doc_id")))
                    except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
                        pass
                    continue
                if tc.function.name != "kb_search":
                    continue

                try:
                    search_args = json.loads(tc.function.arguments or "{}")
                    search_data = json.loads(raw)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                results = search_data.get("results") if isinstance(search_data, dict) else None
                if not isinstance(results, list):
                    continue

                meta_lines = [
                    "doc_id={doc_id} path={path} heading={heading} score={score}".format(
                        doc_id=hit.get("doc_id"),
                        path=hit.get("path") or "",
                        heading=hit.get("heading") or "",
                        score=hit.get("score") or "",
                    )
                    for hit in results[:3]
                    if isinstance(hit, dict)
                ]
                if meta_lines:
                    scratchpad.append("[kb_search metadata]\n" + "\n".join(meta_lines))

                if not results and not retried_empty_search and tool_calls_used < max_calls:
                    original_query = str(search_args.get("query") or "")
                    second_query = _reformulate_kb_query(original_query)
                    if second_query and second_query != original_query:
                        retried_empty_search = True
                        retry_args = {
                            "query": second_query,
                            "top_k": search_args.get("top_k", 5),
                        }
                        if search_args.get("workspace"):
                            retry_args["workspace"] = search_args["workspace"]
                        retry_raw = await self.tool_registry.execute(
                            "kb_search",
                            json.dumps(retry_args, ensure_ascii=False),
                        )
                        tool_calls_used += 1
                        kept_retry = retry_raw[:2000] if len(retry_raw) > 2000 else retry_raw
                        scratchpad.append(
                            f"[kb_search retry Â· {second_query}]\n{kept_retry}"
                        )
                        research_messages.append(
                            Message(
                                role="user",
                                content=(
                                    "Resultado automatico de segunda busqueda KB "
                                    f"({second_query}):\n{kept_retry}"
                                ),
                            )
                        )
                        try:
                            retry_data = json.loads(retry_raw)
                            retry_results = retry_data.get("results", [])
                            if isinstance(retry_results, list):
                                results = retry_results
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass

                for hit in results[:2]:
                    if tool_calls_used >= max_calls or not isinstance(hit, dict):
                        break
                    try:
                        doc_id = int(hit["doc_id"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if doc_id in read_doc_ids:
                        continue
                    read_doc_ids.add(doc_id)
                    read_args = {
                        "doc_id": doc_id,
                        "section": hit.get("heading"),
                        "max_chars": 1800,
                    }
                    read_raw = await self.tool_registry.execute(
                        "kb_read",
                        json.dumps(read_args, ensure_ascii=False),
                    )
                    tool_calls_used += 1
                    kept_read = read_raw[:2000] if len(read_raw) > 2000 else read_raw
                    scratchpad.append(
                        f"[kb_read auto Â· doc_id={doc_id} path={hit.get('path')} heading={hit.get('heading')}]\n{kept_read}"
                    )
                    research_messages.append(
                        Message(
                            role="user",
                            content=(
                                "Lectura automatica del resultado KB principal "
                                f"(doc_id={doc_id}, path={hit.get('path')}, "
                                f"heading={hit.get('heading')}):\n{kept_read}"
                            ),
                        )
                    )

        logger.info(
            "Session '%s' — KB research done: %d calls, %d bullet blocks, %.0fs",
            self.session_id, tool_calls_used, len(scratchpad),
            time.monotonic() - turn_start,
        )

        scratch_text = "\n\n".join(scratchpad) if scratchpad else "(no se obtuvo información de la KB)"
        synth_messages = [
            Message(
                role="system",
                content=(
                    "Eres un asistente que responde en castellano basándose en "
                    "notas de investigación ya recabadas. No uses herramientas. "
                    "Cita documento y sección cuando aporte valor. Si las notas "
                    "no bastan, dilo con claridad y pide al usuario lo que falta."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Pregunta del usuario:\n{user_input}\n\n"
                    f"Notas de la KB:\n{scratch_text}\n\n"
                    "Redacta la respuesta final."
                ),
            ),
        ]
        cleaned = ""
        try:
            # thinking=False: synthesis is a straight "summarise these notes"
            # task — no reasoning pass needed. Skipping it shaves ~10-30 s on
            # the local thinking-mode model.
            synth_result = await self._complete_with_profile(
                "synthesis",
                messages=synth_messages,
                tools=None,
                default_temperature=self.config.temperature,
                default_thinking=False,
            )
            if synth_result.reasoning_content:
                self.last_reasoning = (
                    (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                    + synth_result.reasoning_content
                )
            final_text = synth_result.message.content or ""
            cleaned, _ = _strip_textual_tool_calls(final_text)
            if not cleaned and synth_result.reasoning_content:
                raw_reasoning, _ = _strip_textual_tool_calls(
                    synth_result.reasoning_content
                )
                cleaned = raw_reasoning
        except Exception as exc:
            logger.error(
                "Session '%s' — KB synthesis failed: %s",
                self.session_id, exc,
            )
        if not cleaned:
            cleaned = (
                "No he podido sintetizar una respuesta. Reformula la pregunta "
                "o indícame qué documento quieres consultar."
            )
        final_msg = Message(role="assistant", content=cleaned)
        self._messages.append(final_msg)
        self._persist_message(final_msg)
        if self._observer:
            self._observer.on_done(cleaned)
        yield cleaned

    async def _force_final_answer(
        self,
        full_messages: list[Message],
        user_input: str,
        *,
        reason: str,
    ) -> AsyncIterator[str]:
        """Force a final answer with ``tools=None`` and yield the cleaned text.

        Shared path for both the loop detector and the per-turn budget
        guard. Append a "stop calling tools" nudge, call the provider once
        more, and fall back through three tiers of recovery when the model
        returns empty content: (1) synthesise from reasoning_content,
        (2) expose the raw reasoning, (3) generic apology.
        """
        logger.warning(
            "Session '%s' — forcing final answer: %s",
            self.session_id,
            reason,
        )
        full_messages.append(
            Message(
                role="user",
                content=(
                    "IMPORTANTE: deja de llamar a herramientas. Responde AHORA al "
                    "usuario con la información que ya tengas en castellano. "
                    "Si no basta para contestar, dilo honestamente y pide lo que falta."
                ),
            )
        )
        # thinking=False: we just need a short synthesis, not another reasoning pass.
        result = await self._complete_with_profile(
            "synthesis",
            messages=full_messages,
            tools=None,
            default_temperature=self.config.temperature,
            default_thinking=False,
        )
        forced = result.message
        if result.reasoning_content:
            self.last_reasoning = (
                (self.last_reasoning + "\n\n" if self.last_reasoning else "")
                + result.reasoning_content
            )
        self._messages.append(forced)
        self._persist_message(forced, reasoning=result.reasoning_content)
        content = forced.content or ""
        cleaned, had_markup = _strip_textual_tool_calls(content)
        if had_markup:
            logger.warning(
                "Session '%s' — forced answer contained textual tool-call markup; stripped.",
                self.session_id,
            )
        if not cleaned and result.reasoning_content:
            logger.warning(
                "Session '%s' — forced answer empty; summarising reasoning_content.",
                self.session_id,
            )
            synth_messages = [
                Message(
                    role="system",
                    content=(
                        "Eres un asistente que transforma razonamientos internos "
                        "en respuestas útiles al usuario. No uses herramientas. "
                        "No menciones que estabas pensando ni que te hayan pasado "
                        "un razonamiento. Responde en primera persona como si "
                        "fueras directamente el agente."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Pregunta original del usuario:\n{user_input}\n\n"
                        f"Razonamiento interno que se produjo:\n{result.reasoning_content}\n\n"
                        "Redacta una respuesta directa y útil basada en ese "
                        "razonamiento. Si el razonamiento no basta para responder, "
                        "dilo honestamente y pide al usuario lo que falta."
                    ),
                ),
            ]
            try:
                synth_result = await self._complete_with_profile(
                    "synthesis",
                    messages=synth_messages,
                    tools=None,
                    default_temperature=self.config.temperature,
                    default_thinking=False,
                )
                synth_content = synth_result.message.content or ""
                cleaned, _ = _strip_textual_tool_calls(synth_content)
            except Exception as exc:
                logger.warning(
                    "Session '%s' — reasoning synthesis failed: %s",
                    self.session_id,
                    exc,
                )
            if not cleaned:
                logger.warning(
                    "Session '%s' — synthesis empty; exposing raw reasoning.",
                    self.session_id,
                )
                raw, _ = _strip_textual_tool_calls(result.reasoning_content or "")
                cleaned = raw
        if not cleaned:
            cleaned = (
                "No he podido completar la consulta con las herramientas disponibles. "
                "¿Puedes reformular la pregunta o darme más contexto?"
            )
        final_msg = Message(role="assistant", content=cleaned)
        self._messages.append(final_msg)
        self._persist_message(final_msg)
        if self._observer:
            self._observer.on_done(cleaned)
        yield cleaned

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
                    "Programa una tarea one-shot, recurrente cron o intervalo. "
                    "No inventes hora si falta. Para 00:00 usa "
                    "user_confirmed_midnight=true solo si el usuario pidio medianoche. "
                    "Notify: telegram, log o none."
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
                                "Cuando ejecutar: fecha relativa/absoluta, cron de 5 campos "
                                "o intervalo como 'every 30m'."
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
                    "Cancela una tarea por task_id o por name. Si name es ambiguo, "
                    "devuelve candidatos para pedir confirmacion."
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
