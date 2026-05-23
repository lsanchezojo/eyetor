"""Routing workflow — classify input and dispatch to a specialized agent.

Pattern: Classifier LLM determines the route → specialized agent handles it.

Supports voting: run the classifier N times and choose the route by consensus.
This improves reliability with SLMs that are inconsistent in classification.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.providers.base import BaseProvider
from eyetor.tracking.context import tracking_context

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """A named route with its specialized agent configuration."""

    name: str
    description: str  # Used by classifier to decide routing
    system_prompt: str
    # Optional: use a different provider for this route
    provider: BaseProvider | None = None


@dataclass
class RouterResult:
    """Result of running a Router workflow."""

    chosen_route: str
    output: str
    classifier_reasoning: str = ""
    classifier_confidence: float = 1.0  # fraction of votes for the chosen route


_CLASSIFIER_PROMPT = """You are a routing classifier. Given a user request, choose the most appropriate route.

Available routes:
{routes_list}

Respond with JSON only:
{{"route": "<route_name>", "reasoning": "<brief explanation>"}}

Choose the single best route. If no route fits well, choose the closest one."""


async def classify(
    user_input: str,
    routes: dict[str, Route],
    provider: BaseProvider,
    model: str = "",
    n_votes: int = 1,
    temperature: float = 0.0,
) -> tuple[str, str, float]:
    """Classify user input into a route with optional voting.

    Returns (route_name, reasoning, confidence).
    When n_votes > 1, runs the classifier multiple times and picks by consensus.
    """
    routes_list = "\n".join(
        f"- {name}: {route.description}"
        for name, route in routes.items()
    )
    system = _CLASSIFIER_PROMPT.format(routes_list=routes_list)
    effective_model = model or provider.model

    async def _single_classify() -> tuple[str, str]:
        agent = BaseAgent(
            config=AgentConfig(
                name="classifier",
                provider="",
                model=effective_model,
                system_prompt=system,
                temperature=temperature if n_votes == 1 else 0.7,
            ),
            provider=provider,
        )
        result = await agent.run(user_input)
        return _parse_classification(result.final_output, routes)

    with tracking_context(phase="routing", skip_limit_flag=True):
        if n_votes <= 1:
            route_name, reasoning = await _single_classify()
            return route_name, reasoning, 1.0

        # Voting: run classifier n_votes times in parallel (child tasks copy
        # the context, so each vote records phase=routing).
        results = await asyncio.gather(
            *[_single_classify() for _ in range(n_votes)]
        )

    # Count route votes
    route_names = [r[0] for r in results]
    counter = Counter(route_names)
    winner, win_count = counter.most_common(1)[0]
    confidence = win_count / n_votes

    # Use reasoning from the first vote that matches the winner
    reasoning = next(
        (r[1] for r in results if r[0] == winner),
        "",
    )

    logger.info(
        "Router voting: %d/%d votes for '%s' (%.0f%% confidence). Votes: %s",
        win_count, n_votes, winner, confidence * 100, dict(counter),
    )

    return winner, reasoning, confidence


def _parse_classification(
    text: str, routes: dict[str, Route]
) -> tuple[str, str]:
    """Parse classifier output into (route_name, reasoning)."""
    try:
        data = json.loads(text)
        return data.get("route", ""), data.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        # Try to extract route name from plain text
        for name in routes:
            if name.lower() in text.lower():
                return name, text
        return "", text


class Router:
    """Routes user input to the most appropriate specialized agent.

    Usage:
        router = Router(
            classifier_provider=provider,
            routes=[
                Route("coding", "Programming and code questions", "You are an expert programmer."),
                Route("research", "Research and information lookup", "You are a research specialist."),
            ],
            classifier_votes=3,  # use voting for reliable classification
        )
        result = await router.run("Write a Python function to sort a list")
    """

    def __init__(
        self,
        classifier_provider: BaseProvider,
        routes: list[Route],
        default_provider: BaseProvider | None = None,
        model: str = "",
        temperature: float = 0.0,
        classifier_votes: int = 1,
    ) -> None:
        self._classifier = classifier_provider
        self._routes = {r.name: r for r in routes}
        self._default_provider = default_provider or classifier_provider
        self._model = model or classifier_provider.model
        self._temperature = temperature
        self._classifier_votes = classifier_votes

    async def run(self, user_input: str) -> RouterResult:
        """Classify the input and run the chosen route's agent."""
        route_name, reasoning, confidence = await classify(
            user_input=user_input,
            routes=self._routes,
            provider=self._classifier,
            model=self._model,
            n_votes=self._classifier_votes,
            temperature=self._temperature,
        )

        route = self._routes.get(route_name)
        if not route:
            # Fallback to first route
            route = next(iter(self._routes.values()))
            logger.warning("Classifier chose unknown route '%s', using '%s'", route_name, route.name)

        provider = route.provider or self._default_provider
        agent = BaseAgent(
            config=AgentConfig(
                name=route.name,
                provider="",
                model=self._model,
                system_prompt=route.system_prompt,
                temperature=self._temperature,
            ),
            provider=provider,
        )
        result = await agent.run(user_input)
        logger.info("Routed to '%s': %s", route.name, reasoning)
        return RouterResult(
            chosen_route=route.name,
            output=result.final_output,
            classifier_reasoning=reasoning,
            classifier_confidence=confidence,
        )
