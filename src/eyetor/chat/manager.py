"""SessionManager — manages multiple concurrent ChatSessions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eyetor.chat.session import ChatSession
from eyetor.models.agents import AgentConfig
from eyetor.models.tools import ToolRegistry
from eyetor.providers.base import BaseProvider

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.memory.manager import MemoryManager
    from eyetor.scheduler.channel import SchedulerChannel
    from eyetor.tracking.usage import UsageTracker
    from eyetor.tracking.pricing import CostEstimator

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages a collection of ChatSessions keyed by session_id.

    Each session represents an independent conversation (e.g., one per
    Telegram chat_id or CLI user).
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: BaseProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt_suffix: str = "",
        memory_manager: "MemoryManager | None" = None,
        scheduler: "SchedulerChannel | None" = None,
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
        self._root_config = root_config
        self._tracker = tracker
        self._cost_estimator = cost_estimator
        self._sessions: dict[str, ChatSession] = {}

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
