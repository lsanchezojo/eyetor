"""Parallelization workflow — run multiple agents concurrently.

Two modes:
1. Sectioning: divide a task into subtasks, run in parallel, merge results.
2. Voting: run the same task N times, return the consensus answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A subtask for parallel sectioning."""

    name: str
    system_prompt: str
    input_transform: "Callable[[str], str] | None" = None


@dataclass
class SectioningResult:
    """Result of a parallel sectioning run."""

    section_outputs: dict[str, str]  # section_name → output
    merged_output: str


@dataclass
class VotingResult:
    """Result of a parallel voting run."""

    votes: list[str]
    winner: str
    confidence: float  # fraction of votes for the winner


class Parallel:
    """Runs agents in parallel using asyncio.gather.

    Usage — sectioning:
        parallel = Parallel(provider)
        result = await parallel.section(
            "Analyze this text",
            sections=[
                Section("sentiment", "Analyze the sentiment of this text."),
                Section("topics", "Extract the main topics from this text."),
                Section("summary", "Summarize this text in 2 sentences."),
            ],
        )

    Usage — voting:
        result = await parallel.vote(
            "Is Python better than JavaScript for backend development?",
            system_prompt="Answer yes or no, then explain briefly.",
            n_votes=5,
        )
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str = "",
        temperature: float = 0.7,  # Higher for diversity in voting
    ) -> None:
        self._provider = provider
        self._model = model or provider.model
        self._temperature = temperature

    async def section(
        self,
        user_input: str,
        sections: list[Section],
        merge_prompt: str | None = None,
        merge_provider: BaseProvider | None = None,
    ) -> SectioningResult:
        """Run each section in parallel, then optionally merge with an LLM call."""
        async def run_section(s: Section) -> tuple[str, str]:
            transformed = s.input_transform(user_input) if s.input_transform else user_input
            agent = BaseAgent(
                config=AgentConfig(
                    name=s.name,
                    provider="",
                    model=self._model,
                    system_prompt=s.system_prompt,
                    temperature=self._temperature,
                ),
                provider=self._provider,
            )
            result = await agent.run(transformed)
            return s.name, result.final_output

        tasks = [run_section(s) for s in sections]
        pairs = await asyncio.gather(*tasks)
        section_outputs = dict(pairs)
        logger.info("Parallel sections completed: %s", list(section_outputs.keys()))

        # Merge results
        if merge_prompt:
            sections_text = "\n\n".join(
                f"## {name}\n{output}" for name, output in section_outputs.items()
            )
            merge_input = f"Section results:\n\n{sections_text}\n\nOriginal input:\n{user_input}"
            merger = BaseAgent(
                config=AgentConfig(
                    name="merger",
                    provider="",
                    model=self._model,
                    system_prompt=merge_prompt,
                    temperature=0.0,
                ),
                provider=merge_provider or self._provider,
            )
            merge_result = await merger.run(merge_input)
            merged = merge_result.final_output
        else:
            merged = "\n\n".join(
                f"**{name}**:\n{output}" for name, output in section_outputs.items()
            )

        return SectioningResult(section_outputs=section_outputs, merged_output=merged)

    async def vote(
        self,
        user_input: str,
        system_prompt: str,
        n_votes: int = 5,
    ) -> VotingResult:
        """Run the same prompt N times and return the majority answer."""
        agent_config = AgentConfig(
            name="voter",
            provider="",
            model=self._model,
            system_prompt=system_prompt,
            temperature=self._temperature,
        )

        async def single_vote(_: int) -> str:
            agent = BaseAgent(config=agent_config, provider=self._provider)
            result = await agent.run(user_input)
            return result.final_output

        votes = await asyncio.gather(*[single_vote(i) for i in range(n_votes)])
        votes_list = list(votes)

        # Simple majority: normalize whitespace for comparison
        normalized = [v.strip().lower()[:100] for v in votes_list]
        counter = Counter(normalized)
        winner_norm, win_count = counter.most_common(1)[0]
        # Find the original (un-normalized) vote that matches the winner
        winner = next(v for v, n in zip(votes_list, normalized) if n == winner_norm)
        confidence = win_count / n_votes

        logger.info("Voting result: winner has %.0f%% agreement (%d/%d)", confidence * 100, win_count, n_votes)
        return VotingResult(votes=votes_list, winner=winner, confidence=confidence)
