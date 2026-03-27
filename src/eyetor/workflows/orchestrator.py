"""Orchestrator-Workers workflow.

Pattern: An orchestrator LLM decomposes a task and delegates subtasks
to specialized worker agents via a `delegate` tool.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from eyetor.agents.tool_agent import ToolAgent
from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message
from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class WorkerDefinition:
    """Definition of a worker agent."""

    name: str
    system_prompt: str
    provider: BaseProvider | None = None  # Falls back to orchestrator's provider


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator-workers workflow."""

    final_output: str
    delegations: list[dict] = field(default_factory=list)  # {worker, task, result}
    iterations: int = 0


class OrchestratorWorkflow:
    """Orchestrator that decomposes tasks and delegates to workers.

    The orchestrator has access to a `delegate` tool that routes subtasks
    to registered workers. It synthesizes the results into a final answer.

    Usage:
        workflow = OrchestratorWorkflow(
            orchestrator_provider=provider,
            workers=[
                WorkerDefinition("researcher", "You research topics thoroughly."),
                WorkerDefinition("writer", "You write clear, concise content."),
            ],
        )
        result = await workflow.run("Write a blog post about quantum computing")
    """

    def __init__(
        self,
        orchestrator_provider: BaseProvider,
        workers: list[WorkerDefinition],
        model: str = "",
        temperature: float = 0.0,
        max_iterations: int = 10,
    ) -> None:
        self._provider = orchestrator_provider
        self._workers = {w.name: w for w in workers}
        self._model = model or orchestrator_provider.model
        self._temperature = temperature
        self._max_iterations = max_iterations
        self._delegations: list[dict] = []

    async def run(self, task: str) -> OrchestratorResult:
        """Run the orchestrator-workers workflow."""
        self._delegations = []
        registry = ToolRegistry()

        # Register the delegate tool
        async def delegate(worker_name: str, subtask: str) -> str:
            return await self._delegate(worker_name, subtask)

        registry.register(ToolDefinition(
            name="delegate",
            description=(
                "Delegate a subtask to a specialized worker agent. "
                f"Available workers: {', '.join(self._workers.keys())}. "
                "Use this to decompose the main task into specialized parts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "worker_name": {
                        "type": "string",
                        "description": f"Name of the worker to delegate to. One of: {list(self._workers.keys())}",
                    },
                    "subtask": {
                        "type": "string",
                        "description": "The specific subtask for the worker to perform.",
                    },
                },
                "required": ["worker_name", "subtask"],
            },
            handler=delegate,
        ))

        workers_desc = "\n".join(
            f"- {name}: {w.system_prompt[:100]}..."
            for name, w in self._workers.items()
        )
        orchestrator_system = (
            f"You are an orchestrator. Decompose the task into subtasks and "
            f"delegate them to the appropriate workers using the `delegate` tool.\n\n"
            f"Available workers:\n{workers_desc}\n\n"
            f"After receiving all results, synthesize a final answer."
        )

        agent = ToolAgent(
            config=AgentConfig(
                name="orchestrator",
                provider="",
                model=self._model,
                system_prompt=orchestrator_system,
                temperature=self._temperature,
                max_iterations=self._max_iterations,
            ),
            provider=self._provider,
            tool_registry=registry,
        )
        result = await agent.run(task)
        return OrchestratorResult(
            final_output=result.final_output,
            delegations=list(self._delegations),
            iterations=result.iterations,
        )

    async def _delegate(self, worker_name: str, subtask: str) -> str:
        """Execute a subtask on the named worker."""
        worker = self._workers.get(worker_name)
        if not worker:
            available = list(self._workers.keys())
            return json.dumps({"error": f"Unknown worker '{worker_name}'. Available: {available}"})

        provider = worker.provider or self._provider
        agent = BaseAgent(
            config=AgentConfig(
                name=worker_name,
                provider="",
                model=self._model,
                system_prompt=worker.system_prompt,
                temperature=self._temperature,
            ),
            provider=provider,
        )
        result = await agent.run(subtask)
        delegation = {"worker": worker_name, "task": subtask, "result": result.final_output}
        self._delegations.append(delegation)
        logger.info("Delegated to '%s': %d chars result", worker_name, len(result.final_output))
        return result.final_output
