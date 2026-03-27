"""Prompt Chaining workflow — sequential pipeline of LLM calls.

Pattern: output of step N becomes input to step N+1.
An optional gate function can abort the chain early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

GateFn = Callable[[str], Awaitable[bool]] | Callable[[str], bool]


@dataclass
class ChainStep:
    """A single step in a prompt chain."""

    name: str
    system_prompt: str
    # Optional transform: converts previous output to this step's input
    input_transform: Callable[[str], str] | None = None
    # Optional gate: return False to abort the chain after this step
    gate: GateFn | None = None


@dataclass
class ChainResult:
    """Result of running a PromptChain."""

    steps_completed: int
    outputs: list[str]  # Output of each completed step
    aborted: bool = False
    abort_reason: str = ""

    @property
    def final_output(self) -> str:
        return self.outputs[-1] if self.outputs else ""


class PromptChain:
    """Executes a sequence of LLM calls, passing each output to the next step.

    Example use cases:
    - Document → Summary → Translation → Final Review
    - Raw text → Extract facts → Format as JSON → Validate

    Usage:
        chain = PromptChain(provider=provider, steps=[
            ChainStep("summarize", "Summarize the following text concisely."),
            ChainStep("translate", "Translate to Spanish."),
        ])
        result = await chain.run("Long document text here...")
    """

    def __init__(
        self,
        provider: BaseProvider,
        steps: list[ChainStep],
        model: str = "",
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._steps = steps
        self._model = model or provider.model
        self._temperature = temperature

    async def run(self, initial_input: str) -> ChainResult:
        """Execute the chain starting with initial_input."""
        outputs: list[str] = []
        current_input = initial_input

        for i, step in enumerate(self._steps):
            logger.debug("Chain step %d/%d: %s", i + 1, len(self._steps), step.name)

            # Apply optional input transform
            if step.input_transform:
                current_input = step.input_transform(current_input)

            agent = BaseAgent(
                config=AgentConfig(
                    name=step.name,
                    provider="",  # Not used directly
                    model=self._model,
                    system_prompt=step.system_prompt,
                    temperature=self._temperature,
                ),
                provider=self._provider,
            )
            result = await agent.run(current_input)
            output = result.final_output
            outputs.append(output)
            current_input = output

            # Check gate function
            if step.gate is not None:
                import asyncio, inspect
                if inspect.iscoroutinefunction(step.gate):
                    should_continue = await step.gate(output)
                else:
                    should_continue = step.gate(output)
                if not should_continue:
                    logger.info("Chain aborted at step '%s' by gate function", step.name)
                    return ChainResult(
                        steps_completed=i + 1,
                        outputs=outputs,
                        aborted=True,
                        abort_reason=f"Gate rejected output at step '{step.name}'",
                    )

        return ChainResult(steps_completed=len(self._steps), outputs=outputs)
