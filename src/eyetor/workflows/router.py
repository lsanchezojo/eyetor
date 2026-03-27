"""Routing workflow — classify input and dispatch to a specialized agent.

Pattern: Classifier LLM determines the route → specialized agent handles it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.providers.base import BaseProvider

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


_CLASSIFIER_PROMPT = """You are a routing classifier. Given a user request, choose the most appropriate route.

Available routes:
{routes_list}

Respond with JSON only:
{{"route": "<route_name>", "reasoning": "<brief explanation>"}}

Choose the single best route. If no route fits well, choose the closest one."""


class Router:
    """Routes user input to the most appropriate specialized agent.

    Usage:
        router = Router(
            classifier_provider=provider,
            routes=[
                Route("coding", "Programming and code questions", "You are an expert programmer."),
                Route("research", "Research and information lookup", "You are a research specialist."),
            ],
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
    ) -> None:
        self._classifier = classifier_provider
        self._routes = {r.name: r for r in routes}
        self._default_provider = default_provider or classifier_provider
        self._model = model or classifier_provider.model
        self._temperature = temperature

    async def run(self, user_input: str) -> RouterResult:
        """Classify the input and run the chosen route's agent."""
        route_name, reasoning = await self._classify(user_input)
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
        )

    async def _classify(self, user_input: str) -> tuple[str, str]:
        """Run the classifier and return (route_name, reasoning)."""
        routes_list = "\n".join(
            f"- {name}: {route.description}"
            for name, route in self._routes.items()
        )
        system = _CLASSIFIER_PROMPT.format(routes_list=routes_list)
        agent = BaseAgent(
            config=AgentConfig(
                name="classifier",
                provider="",
                model=self._model,
                system_prompt=system,
                temperature=0.0,
            ),
            provider=self._classifier,
        )
        result = await agent.run(user_input)
        try:
            data = json.loads(result.final_output)
            return data.get("route", ""), data.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            # Try to extract route name from plain text
            for name in self._routes:
                if name.lower() in result.final_output.lower():
                    return name, result.final_output
            return "", result.final_output
