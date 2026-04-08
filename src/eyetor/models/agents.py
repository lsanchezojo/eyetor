"""Agent configuration and result models."""

from __future__ import annotations

from pydantic import BaseModel

from eyetor.models.messages import Message, ToolCall


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""

    name: str
    provider: str  # Key into providers config
    model: str
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = []  # Tool names to load
    skills: list[str] = []  # Skill names to load
    max_iterations: int = 20  # Agentic loop safety limit
    temperature: float = 0.0


class AgentResult(BaseModel):
    """Result of running an agent."""

    messages: list[Message]  # Full conversation trace
    final_output: str  # Last assistant content
    iterations: int
    tool_calls_made: list[ToolCall] = []
