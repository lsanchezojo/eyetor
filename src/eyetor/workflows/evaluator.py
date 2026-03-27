"""Evaluator-Optimizer workflow.

Pattern: Generator produces output → Evaluator scores and critiques →
If not PASS, Generator receives feedback and retries (up to max_rounds).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_EVALUATOR_PROMPT = """You are an evaluator. Assess the quality of the following output.

Task description: {task_description}
Quality criteria: {criteria}

Output to evaluate:
---
{output}
---

Respond with JSON only:
{{
  "verdict": "PASS" or "FAIL",
  "score": <0-10>,
  "feedback": "<specific, actionable feedback for improvement>"
}}

PASS means the output meets all criteria. FAIL means it needs improvement."""


@dataclass
class EvalRound:
    """Record of a single generation+evaluation round."""

    round: int
    output: str
    verdict: str  # "PASS" or "FAIL"
    score: float
    feedback: str


@dataclass
class EvaluatorResult:
    """Result of the evaluator-optimizer workflow."""

    final_output: str
    rounds: list[EvalRound]
    passed: bool

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)


class EvaluatorOptimizer:
    """Iteratively improves output quality using an evaluator feedback loop.

    Usage:
        workflow = EvaluatorOptimizer(
            generator_provider=provider,
            evaluator_provider=provider,
            generator_prompt="Write a professional email.",
            criteria="Clear, professional tone. Under 200 words. No typos.",
            max_rounds=3,
        )
        result = await workflow.run("Email to reschedule a meeting to next Thursday")
    """

    def __init__(
        self,
        generator_provider: BaseProvider,
        evaluator_provider: BaseProvider | None = None,
        generator_prompt: str = "You are a helpful assistant.",
        evaluator_prompt: str | None = None,
        criteria: str = "The output should be high quality, accurate, and well-structured.",
        max_rounds: int = 3,
        model: str = "",
        temperature: float = 0.7,
    ) -> None:
        self._gen_provider = generator_provider
        self._eval_provider = evaluator_provider or generator_provider
        self._gen_prompt = generator_prompt
        self._eval_prompt = evaluator_prompt or _EVALUATOR_PROMPT
        self._criteria = criteria
        self._max_rounds = max_rounds
        self._model = model or generator_provider.model
        self._temperature = temperature

    async def run(self, task: str) -> EvaluatorResult:
        """Run the generation+evaluation loop."""
        rounds: list[EvalRound] = []
        current_input = task
        feedback = ""

        for round_num in range(1, self._max_rounds + 1):
            # Generator step
            gen_input = current_input
            if feedback:
                gen_input = (
                    f"{task}\n\n"
                    f"Previous attempt feedback:\n{feedback}\n\n"
                    f"Please improve your response based on this feedback."
                )

            generator = BaseAgent(
                config=AgentConfig(
                    name="generator",
                    provider="",
                    model=self._model,
                    system_prompt=self._gen_prompt,
                    temperature=self._temperature,
                ),
                provider=self._gen_provider,
            )
            gen_result = await generator.run(gen_input)
            output = gen_result.final_output

            # Evaluator step
            eval_system = self._eval_prompt.format(
                task_description=task,
                criteria=self._criteria,
                output=output,
            )
            evaluator = BaseAgent(
                config=AgentConfig(
                    name="evaluator",
                    provider="",
                    model=self._model,
                    system_prompt="You are a strict quality evaluator. Always respond with valid JSON.",
                    temperature=0.0,
                ),
                provider=self._eval_provider,
            )
            eval_result = await evaluator.run(eval_system)

            verdict, score, feedback = self._parse_eval(eval_result.final_output)
            round_record = EvalRound(
                round=round_num,
                output=output,
                verdict=verdict,
                score=score,
                feedback=feedback,
            )
            rounds.append(round_record)
            logger.info(
                "Round %d/%d: verdict=%s score=%.1f",
                round_num, self._max_rounds, verdict, score,
            )

            if verdict == "PASS":
                return EvaluatorResult(final_output=output, rounds=rounds, passed=True)

        # Max rounds reached — return last output
        last = rounds[-1]
        logger.warning("EvaluatorOptimizer reached max_rounds=%d without PASS", self._max_rounds)
        return EvaluatorResult(final_output=last.output, rounds=rounds, passed=False)

    def _parse_eval(self, raw: str) -> tuple[str, float, str]:
        """Parse evaluator JSON response. Returns (verdict, score, feedback)."""
        try:
            data = json.loads(raw)
            return (
                str(data.get("verdict", "FAIL")).upper(),
                float(data.get("score", 0)),
                str(data.get("feedback", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            # Best-effort: check if PASS or FAIL appears in text
            verdict = "PASS" if "PASS" in raw.upper() else "FAIL"
            return verdict, 0.0, raw
