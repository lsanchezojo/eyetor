"""Routing workflow — classify input and dispatch to a specialized agent.

Pattern: Classifier LLM determines the route → specialized agent handles it.

Supports voting: run the classifier N times and choose the route by consensus.
This improves reliability with SLMs that are inconsistent in classification.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass

from eyetor.agents.base import BaseAgent
from eyetor.models.agents import AgentConfig
from eyetor.models.messages import Message
from eyetor.providers.base import BaseProvider
from eyetor.utils.json import extract_json_object

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

{history_block}Respond with JSON only:
{{"route": "<route_name>", "reasoning": "<brief explanation>"}}

Choose the single best route. If no route fits well, choose the closest one.
When recent conversation is provided, use it to disambiguate short or context-dependent messages (e.g. a follow-up like "las credenciales están ahí" refers back to whatever the agent was just doing)."""


# Keep the last K user/assistant turns as classifier context. Tool messages are
# skipped — for routing, "what did the user ask and what did the agent say" is
# the useful signal; raw tool outputs are noise and bloat the prompt.
_HISTORY_MAX_MESSAGES = 6
_HISTORY_MESSAGE_CHARS = 200


def _format_history(history: list | None) -> str:
    """Render conversation history into a compact block for the classifier."""
    if not history:
        return ""
    recent: list = []
    for msg in reversed(history):
        if getattr(msg, "role", None) not in ("user", "assistant"):
            continue
        content = getattr(msg, "content", None) or ""
        content = " ".join(content.split())
        if not content:
            continue
        if len(content) > _HISTORY_MESSAGE_CHARS:
            content = content[:_HISTORY_MESSAGE_CHARS] + "…"
        recent.append(f"{msg.role}: {content}")
        if len(recent) >= _HISTORY_MAX_MESSAGES:
            break
    if not recent:
        return ""
    body = "\n".join(reversed(recent))
    return f"Recent conversation (oldest → newest):\n{body}\n\n"


async def classify(
    user_input: str,
    routes: dict[str, Route],
    provider: BaseProvider,
    model: str = "",
    n_votes: int = 1,
    temperature: float = 0.0,
    history: list | None = None,
) -> tuple[str, str, float]:
    """Classify user input into a route with optional voting.

    Returns (route_name, reasoning, confidence).
    When n_votes > 1, runs the classifier multiple times and picks by consensus.
    ``history`` is an optional list of ``Message`` objects from the ongoing
    session. The last few user/assistant turns are formatted into the prompt
    so the classifier can disambiguate short follow-ups.
    """
    routes_list = "\n".join(
        f"- {name}: {route.description}"
        for name, route in routes.items()
    )
    history_block = _format_history(history)
    system = _CLASSIFIER_PROMPT.format(
        routes_list=routes_list,
        history_block=history_block,
    )

    async def _single_classify() -> tuple[str, str]:
        # Call the provider directly with thinking=False. The classifier
        # outputs a 2-field JSON — reasoning-mode overhead here is pure
        # latency with no gain. Going around BaseAgent lets us pass the
        # flag through; BaseAgent has no tools/loop anyway.
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user_input),
        ]
        call_temp = temperature if n_votes == 1 else 0.7
        result = await provider.complete(
            messages=messages,
            tools=None,
            temperature=call_temp,
            thinking=False,
        )
        output = result.message.content or ""
        return _parse_classification(output, routes)

    if n_votes <= 1:
        route_name, reasoning = await _single_classify()
        return route_name, reasoning, 1.0

    # Voting: run classifier n_votes times in parallel
    results = await asyncio.gather(*[_single_classify() for _ in range(n_votes)])

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
    data = extract_json_object(text)
    if data:
        route = str(data.get("route", ""))
        if route in routes:
            return route, str(data.get("reasoning", ""))
        if route:
            best = _score_route(route, routes)
            if best:
                return best, str(data.get("reasoning", ""))
    best = _score_route(text, routes)
    return (best or "", text)


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _score_route(text: str, routes: dict[str, Route]) -> str:
    """Choose the route whose name/description overlaps most with text."""
    text_tokens = {t.lower() for t in _WORD_RE.findall(text)}
    if not text_tokens:
        return ""
    best_name = ""
    best_score = 0.0
    for name, route in routes.items():
        name_l = name.lower()
        score = 2.0 if name_l in text.lower() else 0.0
        route_tokens = {t.lower() for t in _WORD_RE.findall(f"{name} {route.description}")}
        if route_tokens:
            score += len(text_tokens & route_tokens) / len(route_tokens)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name if best_score > 0 else ""


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
