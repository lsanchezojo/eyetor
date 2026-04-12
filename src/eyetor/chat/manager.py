"""SessionManager — manages multiple concurrent ChatSessions."""

from __future__ import annotations

import logging
from typing import AsyncIterator, TYPE_CHECKING

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
        self._routing_votes = routing_cfg.classifier_votes
        self._routing_enabled = True
        logger.info(
            "Routing enabled: %d routes, %d classifier votes",
            len(self._routing_routes),
            self._routing_votes,
        )

    async def route_and_send(
        self, session_id: str, user_input: str
    ) -> AsyncIterator[str]:
        """Classify the message (if routing enabled), then send to session.

        When routing is active, the classifier determines which route best
        matches the input, and the route's system_prompt is applied as a
        temporary override for this turn. The session's base system_prompt
        is restored after the turn completes.
        """
        session = self.get_or_create(session_id)

        if self._routing_enabled and self._routing_routes:
            route_name, reasoning, confidence = await self._classify(user_input)
            if route_name and route_name in self._routing_routes:
                route = self._routing_routes[route_name]
                logger.info(
                    "Routed session '%s' to '%s' (%.0f%% confidence): %s",
                    session_id, route_name, confidence * 100, reasoning,
                )
                # Apply route-specific system prompt as suffix for this turn
                original_suffix = session._system_prompt_suffix
                route_context = (
                    f"\n\n[Route: {route_name}]\n{route.system_prompt}"
                )
                session._system_prompt_suffix = original_suffix + route_context
                try:
                    async for chunk in session.send(user_input):
                        yield chunk
                finally:
                    session._system_prompt_suffix = original_suffix
                return

        # No routing — direct send
        async for chunk in session.send(user_input):
            yield chunk

    async def _classify(self, user_input: str) -> tuple[str, str, float]:
        """Run the routing classifier with optional voting."""
        from eyetor.workflows.router import classify

        return await classify(
            user_input=user_input,
            routes=self._routing_routes,
            provider=self._provider,
            n_votes=self._routing_votes,
        )

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
