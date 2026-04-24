"""Agent configuration and result models."""

from __future__ import annotations

from pydantic import BaseModel

from eyetor.models.messages import Message, ToolCall


class TurnBudget(BaseModel):
    """Per-turn runtime budget for a ChatSession.

    Unlike ``max_iterations`` (a raw safety cap), these budgets model what a
    user is actually willing to wait for and how many tool calls a question
    ought to take. When either is exceeded, the session should stop
    dispatching tools and force a synthesis phase with whatever has been
    gathered so far.

    ``0`` on either field disables that specific budget; use ``max_tool_calls=0``
    for routes (e.g. small-talk) where no tools are expected.
    """

    max_tool_calls: int = 6
    max_wall_seconds: int = 180


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""

    name: str
    provider: str  # Key into providers config
    model: str
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = []  # Tool names to load
    skills: list[str] = []  # Skill names to load
    max_iterations: int = 20  # Agentic loop last-resort safety limit
    temperature: float = 0.0
    budget: TurnBudget = TurnBudget()


class AgentResult(BaseModel):
    """Result of running an agent."""

    messages: list[Message]  # Full conversation trace
    final_output: str  # Last assistant content
    iterations: int
    tool_calls_made: list[ToolCall] = []
