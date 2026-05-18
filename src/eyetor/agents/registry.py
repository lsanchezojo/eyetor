"""In-memory registry for loaded subagent definitions."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.agents.loader import AgentDefinition, discover_agents

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Holds the agent definitions discovered at startup."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def discover(self, agents_dirs: list[str | Path]) -> None:
        """Scan the configured directories and register all valid agents."""
        found = discover_agents(agents_dirs)
        for agent in found:
            self._agents[agent.name] = agent
            logger.debug("Discovered agent: %s (%s)", agent.name, agent.path)
        logger.info("Agents discovered: %d", len(found))

    def list_names(self) -> list[str]:
        return list(self._agents.keys())

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get(self, name: str) -> AgentDefinition:
        if name not in self._agents:
            raise KeyError(f"Agent not found: {name!r}")
        return self._agents[name]

    def has(self, name: str) -> bool:
        return name in self._agents
