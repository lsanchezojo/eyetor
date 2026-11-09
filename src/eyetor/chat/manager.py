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
        enable_conversations: bool = False,
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

        # ChatGPT-style conversations (opt-in per manager: CLI/Telegram yes,
        # scheduler no). Requires session persistence for the JSONL files.
        self._conv_index = None
        if enable_conversations and root_config and root_config.sessions.persist:
            from pathlib import Path

            from eyetor.chat.conversations import ConversationIndex

            sessions_dir = Path(root_config.sessions.dir).expanduser()
            self._conv_index = ConversationIndex(sessions_dir / "conversations.json")

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
        """Return the existing session or create a new one.

        With conversations enabled the session is bound to the chat's active
        conversation; ``session_id`` stays the stable chat identity (access
        policy, memory, scheduler scoping are all keyed by it).
        """
        if session_id not in self._sessions:
            from eyetor.access import resolve

            access = resolve(self._root_config, session_id)
            conv_id = None
            if self._conv_index is not None:
                from pathlib import Path

                from eyetor.chat.conversations import sanitize_id

                # First contact migrates a legacy single-file history, if any.
                legacy = (
                    Path(self._root_config.sessions.dir).expanduser()
                    / f"{sanitize_id(session_id)}.jsonl"
                )
                conv_id = self._conv_index.active(session_id, legacy_file=legacy)
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
                access=access,
                conversation_id=conv_id,
                conversation_index=self._conv_index,
            )
            logger.debug("Created new session: %s (conversation=%s)", session_id, conv_id)
        return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        """Remove and discard a session."""
        self._sessions.pop(session_id, None)
        logger.debug("Closed session: %s", session_id)

    # ------------------------------------------------------------------
    # Conversations (ChatGPT-style)
    # ------------------------------------------------------------------

    def new_conversation(self, session_id: str) -> str | None:
        """Open a fresh conversation for a chat; return its id.

        Without an index (conversations disabled or persist off) it falls back
        to wiping the single session's history, preserving old behavior.
        """
        if self._conv_index is None:
            self.get_or_create(session_id).reset()
            return None
        conv_id = self._conv_index.new(session_id)
        # Drop the cached session so the next message binds to the new file.
        self._sessions.pop(session_id, None)
        return conv_id

    def switch_conversation(self, session_id: str, conv_id: str) -> bool:
        """Resume a past conversation of this chat."""
        if self._conv_index is None:
            return False
        if not self._conv_index.switch(session_id, conv_id):
            return False
        self._sessions.pop(session_id, None)
        return True

    def list_conversations(self, session_id: str) -> list[dict]:
        """Conversations of a chat, newest first (empty if disabled)."""
        if self._conv_index is None:
            return []
        return self._conv_index.list(session_id)

    def active_conversation_id(self, session_id: str) -> str | None:
        """The chat's active conversation id, or None."""
        if self._conv_index is None:
            return None
        return self._conv_index.active_id(session_id)

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def list_providers(self) -> dict[str, str]:
        """Return available providers: name → default model."""
        if not self._root_config:
            return {}
        return {name: cfg.model for name, cfg in self._root_config.providers.items()}
