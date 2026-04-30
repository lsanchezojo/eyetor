"""SessionManager — manages multiple concurrent ChatSessions."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, TYPE_CHECKING

from eyetor.chat.session import ChatSession
from eyetor.models.agents import AgentConfig
from eyetor.models.tools import ToolRegistry
from eyetor.providers.base import BaseProvider

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.knowledge.manager import KnowledgeManager
    from eyetor.memory.manager import MemoryManager
    from eyetor.scheduler.channel import SchedulerChannel
    from eyetor.tracking.usage import UsageTracker
    from eyetor.tracking.pricing import CostEstimator

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages a collection of ChatSessions keyed by session_id.

    Each session represents an independent conversation (e.g., one per
    Telegram chat_id or CLI user).

    When routing is configured, incoming messages are classified before
    being sent to the session, and the session's system prompt is overridden
    with a route-specific prompt for that turn.
    """

    def __init__(
        self,
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
    ) -> None:
        self._config = config
        self._provider = provider
        self._tool_registry = tool_registry
        self._system_prompt_suffix = system_prompt_suffix
        self._memory = memory_manager
        self._scheduler = scheduler
        self._knowledge = knowledge
        self._root_config = root_config
        self._tracker = tracker
        self._cost_estimator = cost_estimator
        self._sessions: dict[str, ChatSession] = {}

        # Pre-build routing structures if enabled
        self._routing_enabled = False
        self._routing_routes: dict | None = None
        self._routing_votes: int = 1
        if root_config and root_config.routing.enabled and root_config.routing.routes:
            self._init_routing(root_config)

    def _provider_targets(self) -> list[Any]:
        targets: list[Any] = []

        def collect(provider: Any) -> None:
            if provider in targets:
                return
            targets.append(provider)
            inner = getattr(provider, "_inner", None)
            if inner is not None:
                collect(inner)
            for child in getattr(provider, "_providers", []) or []:
                collect(child)

        collect(self._provider)
        return targets

    def _init_routing(self, root_config: "VectorConfig") -> None:
        """Initialize routing classifier from config."""
        from eyetor.workflows.router import Route

        routing_cfg = root_config.routing
        self._routing_routes = {
            name: Route(
                name=name,
                description=rcfg.description,
                system_prompt=rcfg.system_prompt,
            )
            for name, rcfg in routing_cfg.routes.items()
        }
        # Tool allowlists per route — stored separately because Route (a
        # workflows primitive) has no tools field. None = all tools, [] = no
        # tools, [patterns] = whitelist. See ``RouteConfig.tools``.
        self._routing_tools: dict[str, list[str] | None] = {
            name: rcfg.tools for name, rcfg in routing_cfg.routes.items()
        }
        # Named handler per route (e.g. "kb_2phase"). None = generic loop.
        self._routing_handlers: dict[str, str | None] = {
            name: rcfg.handler for name, rcfg in routing_cfg.routes.items()
        }
        self._routing_votes = routing_cfg.classifier_votes
        self._routing_enabled = True
        logger.info(
            "Routing enabled: %d routes, %d classifier votes",
            len(self._routing_routes),
            self._routing_votes,
        )

    async def route_and_send(
        self,
        session_id: str,
        user_input: str,
        *,
        allow_chain: bool = True,
    ) -> AsyncIterator[str]:
        """Classify the message (if routing enabled), then send to session.

        When routing is active, the classifier determines which route best
        matches the input and the route's system_prompt + tool allowlist are
        applied for this turn only. The session's base system_prompt is
        restored afterwards.

        ``allow_chain`` is forwarded to ``ChatSession.send`` so callers that
        already build a long prompt (photo/voice handlers) can suppress the
        plan→execute→synthesise decomposition.
        """
        session = self.get_or_create(session_id)

        if self._routing_enabled and self._routing_routes:
            route_name, reasoning, confidence = await self._classify(
                user_input, history=session.get_history()
            )
            if route_name and route_name in self._routing_routes:
                route = self._routing_routes[route_name]
                tools_override = self._routing_tools.get(route_name)
                handler = self._routing_handlers.get(route_name)
                logger.info(
                    "Routed session '%s' to '%s' (%.0f%% confidence, handler=%s, tools=%s): %s",
                    session_id, route_name, confidence * 100,
                    handler or "default",
                    "all" if tools_override is None else tools_override,
                    reasoning,
                )
                # kb_2phase has its own research+synthesis flow and bakes its
                # own system prompt; skip the generic suffix override in that
                # case to avoid two conflicting guidance blocks.
                if handler == "kb_2phase":
                    async for chunk in session.send_kb_query(
                        user_input, tools_override=tools_override
                    ):
                        yield chunk
                    return
                original_suffix = session._system_prompt_suffix
                route_context = (
                    f"\n\n[Route: {route_name}]\n{route.system_prompt}"
                )
                session._system_prompt_suffix = original_suffix + route_context
                try:
                    async for chunk in session.send(
                        user_input,
                        allow_chain=allow_chain,
                        tools_override=tools_override,
                    ):
                        yield chunk
                finally:
                    session._system_prompt_suffix = original_suffix
                return

        # No routing — direct send
        async for chunk in session.send(user_input, allow_chain=allow_chain):
            yield chunk

    async def _classify(
        self, user_input: str, *, history: list | None = None
    ) -> tuple[str, str, float]:
        """Run the routing classifier with optional voting.

        ``history`` — the session's recent messages, forwarded to the
        classifier so context-dependent follow-ups ("sí", "las credenciales
        están ahí") route to the same ruta as the turn they refer back to.
        """
        from eyetor.workflows.router import classify

        profile = self._root_config.profiles.classifier if self._root_config else None
        temperature = 0.0 if not profile or profile.temperature is None else profile.temperature
        saved: list[tuple[Any, dict, dict]] = []
        if profile and (profile.extra_body or profile.options):
            for target in self._provider_targets():
                saved.append((target, dict(target.extra_body), dict(target.options)))
                target.extra_body = {**target.extra_body, **profile.extra_body}
                target.options = {**target.options, **profile.options}
        try:
            return await classify(
                user_input=user_input,
                routes=self._routing_routes,
                provider=self._provider,
                n_votes=self._routing_votes,
                temperature=temperature,
                history=history,
            )
        finally:
            for target, extra_body, options in saved:
                target.extra_body = extra_body
                target.options = options

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(
                session_id=session_id,
                config=self._config,
                provider=self._provider,
                tool_registry=self._tool_registry,
                system_prompt_suffix=self._system_prompt_suffix,
                memory_manager=self._memory,
                scheduler=self._scheduler,
                knowledge=self._knowledge,
                root_config=self._root_config,
                tracker=self._tracker,
                cost_estimator=self._cost_estimator,
            )
            logger.debug("Created new session: %s", session_id)
        return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        """Remove and discard a session."""
        self._sessions.pop(session_id, None)
        logger.debug("Closed session: %s", session_id)

    def reset(self, session_id: str) -> None:
        """Reset the history of an existing session (or create empty one)."""
        session = self.get_or_create(session_id)
        session.reset()
        if self._tracker:
            self._tracker.clear_session(session_id)

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def list_providers(self) -> dict[str, str]:
        """Return available providers: name → default model."""
        if not self._root_config:
            return {}
        return {name: cfg.model for name, cfg in self._root_config.providers.items()}
