"""Orchestrator-Workers workflow.

Pattern: An orchestrator LLM decomposes a task and delegates subtasks
to specialized worker agents via a `delegate` tool.

Supports two protocols:
- tool_calling: orchestrator uses structured tool_calls (requires capable model)
- text: orchestrator outputs JSON instructions in plain text (SLM-friendly)
- auto: tries tool_calling first, falls back to text on failure
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from eyetor.agents.tool_agent import ToolAgent
from eyetor.agents.base import BaseAgent
from eyetor.agents.registry import AgentRegistry
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message
from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Regex to extract JSON objects from free-form text
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from text, with fallback to embedded JSON extraction."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        )
        try:
            return json.loads(inner.strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # Search for embedded JSON objects
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            candidate = json.loads(match.group())
            if isinstance(candidate, dict) and ("action" in candidate or "worker" in candidate):
                return candidate
        except (json.JSONDecodeError, ValueError):
            continue
    return None


@dataclass
class WorkerDefinition:
    """Inline worker definition built in code.

    Prefer declaring agents as ``<name>.md`` files and passing their names to
    ``OrchestratorWorkflow`` along with an :class:`AgentRegistry`. This class
    is kept for callers that construct workers programmatically.
    """

    name: str
    system_prompt: str
    provider: BaseProvider | None = None  # Falls back to orchestrator's provider
    model: str = ""           # Falls back to orchestrator's model
    temperature: float | None = None  # Falls back to orchestrator's temperature


def _workers_from_registry(
    names: list[str], registry: AgentRegistry
) -> list[WorkerDefinition]:
    """Resolve agent names from the registry into worker definitions."""
    workers: list[WorkerDefinition] = []
    for name in names:
        if not registry.has(name):
            available = ", ".join(registry.list_names()) or "<none>"
            raise KeyError(
                f"Orchestrator worker '{name}' not found in agents registry "
                f"(available: {available})"
            )
        definition = registry.get(name)
        workers.append(WorkerDefinition(
            name=definition.name,
            system_prompt=definition.system_prompt,
            model=definition.model,
            temperature=definition.temperature,
        ))
    return workers


@dataclass
class OrchestratorResult:
    """Result of running the orchestrator-workers workflow."""

    final_output: str
    delegations: list[dict] = field(default_factory=list)  # {worker, task, result}
    iterations: int = 0


_TEXT_PROTOCOL_SYSTEM = """\
You are an orchestrator. Decompose the task into subtasks and delegate them \
to specialized workers.

Available workers:
{workers_desc}

IMPORTANT: You must respond ONLY with a JSON object on each turn. Two actions are available:

1. Delegate a subtask to a worker:
{{"action": "delegate", "worker": "<worker_name>", "task": "<subtask description>"}}

2. Provide your final synthesized answer (after receiving all worker results):
{{"action": "final_answer", "content": "<your final answer>"}}

Rules:
- Output ONLY the JSON object, no extra text.
- Delegate one subtask at a time.
- After receiving a worker's result, decide whether to delegate again or give the final answer.
- Synthesize all worker results into a coherent final answer."""


class OrchestratorWorkflow:
    """Orchestrator that decomposes tasks and delegates to workers.

    The orchestrator has access to a `delegate` tool that routes subtasks
    to registered workers. It synthesizes the results into a final answer.

    Supports three protocols via the ``protocol`` parameter:
    - ``tool_calling``: uses structured tool_calls (original behavior)
    - ``text``: uses JSON-in-text protocol (SLM-friendly, no tool_calls needed)
    - ``auto``: tries tool_calling first, falls back to text on failure

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
        workers: list[WorkerDefinition] | list[str],
        model: str = "",
        temperature: float = 0.0,
        max_iterations: int = 10,
        protocol: Literal["tool_calling", "text", "auto"] = "auto",
        agent_registry: AgentRegistry | None = None,
    ) -> None:
        # Resolve worker names against the agent registry when given as strings.
        if workers and isinstance(workers[0], str):
            if agent_registry is None:
                raise ValueError(
                    "OrchestratorWorkflow: workers were given as names but no "
                    "agent_registry was provided"
                )
            resolved = _workers_from_registry(list(workers), agent_registry)  # type: ignore[arg-type]
        else:
            resolved = list(workers)  # type: ignore[arg-type]

        self._provider = orchestrator_provider
        self._workers = {w.name: w for w in resolved}
        self._model = model or orchestrator_provider.model
        self._temperature = temperature
        self._max_iterations = max_iterations
        self._protocol = protocol
        self._delegations: list[dict] = []

    async def run(self, task: str) -> OrchestratorResult:
        """Run the orchestrator-workers workflow."""
        if self._protocol == "text":
            return await self._run_text_protocol(task)
        if self._protocol == "tool_calling":
            return await self._run_tool_protocol(task)

        # auto: try tool_calling, fall back to text
        try:
            result = await self._run_tool_protocol(task)
            if result.delegations:
                return result
            # No delegations means the model didn't use the delegate tool —
            # possibly a tool-calling failure. Try text protocol.
            logger.info(
                "Orchestrator tool_calling produced no delegations, falling back to text protocol"
            )
        except Exception as exc:
            logger.warning(
                "Orchestrator tool_calling failed (%s), falling back to text protocol",
                exc,
            )
        return await self._run_text_protocol(task)

    # ------------------------------------------------------------------
    # Tool-calling protocol (original)
    # ------------------------------------------------------------------

    async def _run_tool_protocol(self, task: str) -> OrchestratorResult:
        """Run using structured tool_calls (requires capable model)."""
        self._delegations = []
        registry = ToolRegistry()

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

    # ------------------------------------------------------------------
    # Text protocol (SLM-friendly, no tool_calls needed)
    # ------------------------------------------------------------------

    async def _run_text_protocol(self, task: str) -> OrchestratorResult:
        """Run using JSON-in-text protocol for SLMs that struggle with tool_calls."""
        self._delegations = []

        workers_desc = "\n".join(
            f"- {name}: {w.system_prompt[:100]}"
            for name, w in self._workers.items()
        )
        system_prompt = _TEXT_PROTOCOL_SYSTEM.format(workers_desc=workers_desc)

        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]

        iterations = 0
        max_parse_retries = 2

        while iterations < self._max_iterations:
            iterations += 1

            result = await self._provider.complete(
                messages=messages,
                tools=None,
                temperature=self._temperature,
            )
            response_text = result.message.content or ""
            messages.append(Message(role="assistant", content=response_text))

            logger.info(
                "Orchestrator text protocol iter %d: %s",
                iterations,
                response_text[:200],
            )

            # Parse the JSON instruction
            instruction = _extract_json(response_text)

            if instruction is None:
                # Retry: ask the model to fix its output
                if max_parse_retries > 0:
                    max_parse_retries -= 1
                    messages.append(Message(
                        role="user",
                        content=(
                            "Tu respuesta no es JSON válido. Recuerda: debes responder "
                            "SOLO con un objeto JSON. Ejemplo:\n"
                            '{"action": "delegate", "worker": "nombre", "task": "tarea"}\n'
                            "o\n"
                            '{"action": "final_answer", "content": "respuesta final"}\n'
                            "Inténtalo de nuevo."
                        ),
                    ))
                    continue
                # Give up — treat last response as final answer
                logger.warning(
                    "Orchestrator text protocol: failed to parse JSON after retries"
                )
                return OrchestratorResult(
                    final_output=response_text,
                    delegations=list(self._delegations),
                    iterations=iterations,
                )

            action = instruction.get("action", "")

            if action == "final_answer":
                return OrchestratorResult(
                    final_output=instruction.get("content", response_text),
                    delegations=list(self._delegations),
                    iterations=iterations,
                )

            if action == "delegate":
                worker_name = instruction.get("worker", "")
                subtask = instruction.get("task", "")

                if not worker_name or not subtask:
                    messages.append(Message(
                        role="user",
                        content=(
                            "La delegación necesita 'worker' y 'task'. "
                            f"Workers disponibles: {list(self._workers.keys())}. "
                            "Inténtalo de nuevo."
                        ),
                    ))
                    continue

                worker_result = await self._delegate(worker_name, subtask)
                messages.append(Message(
                    role="user",
                    content=f"Resultado del worker '{worker_name}':\n\n{worker_result}",
                ))
                continue

            # Unknown action
            messages.append(Message(
                role="user",
                content=(
                    f"Acción desconocida: '{action}'. Las acciones válidas son "
                    "'delegate' y 'final_answer'. Inténtalo de nuevo."
                ),
            ))

        # Max iterations reached
        logger.warning("Orchestrator text protocol reached max_iterations=%d", self._max_iterations)
        last_content = next(
            (m.content for m in reversed(messages) if m.role == "assistant" and m.content),
            "Max iterations reached.",
        )
        return OrchestratorResult(
            final_output=last_content,
            delegations=list(self._delegations),
            iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Shared: delegate to a worker
    # ------------------------------------------------------------------

    async def _delegate(self, worker_name: str, subtask: str) -> str:
        """Execute a subtask on the named worker."""
        from eyetor.workflows.observer import WorkerObserver

        worker = self._workers.get(worker_name)
        if not worker:
            available = list(self._workers.keys())
            return json.dumps({"error": f"Unknown worker '{worker_name}'. Available: {available}"})

        provider = worker.provider or self._provider
        observer = WorkerObserver()
        worker_model = worker.model or self._model
        worker_temperature = (
            worker.temperature if worker.temperature is not None else self._temperature
        )
        agent = BaseAgent(
            config=AgentConfig(
                name=worker_name,
                provider="",
                model=worker_model,
                system_prompt=worker.system_prompt,
                temperature=worker_temperature,
            ),
            provider=provider,
        )
        result = await agent.run(subtask)
        observer.on_done(result.final_output)
        delegation = {
            "worker": worker_name,
            "task": subtask,
            "result": result.final_output,
            "summary": observer.get_summary(),
            "events": observer.get_events(),
        }
        self._delegations.append(delegation)
        logger.info("Delegated to '%s': %d chars result", worker_name, len(result.final_output))
        return result.final_output
