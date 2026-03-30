"""Cost estimation for LLM API calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Per-token pricing for a model (USD per 1K tokens)."""

    prompt_per_1k: float
    completion_per_1k: float


# Known pricing — extend as models are added.
# For local models (ollama, llamacpp) no entry is needed; cost defaults to 0.
_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(0.0025, 0.01),
    "gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    "gpt-4.1": ModelPricing(0.002, 0.008),
    "gpt-4.1-mini": ModelPricing(0.0004, 0.0016),
    "gpt-4.1-nano": ModelPricing(0.0001, 0.0004),
    # Anthropic
    "claude-sonnet-4": ModelPricing(0.003, 0.015),
    "claude-3.5-sonnet": ModelPricing(0.003, 0.015),
    "claude-3.5-haiku": ModelPricing(0.0008, 0.004),
    # Google
    "gemini-2.5-pro": ModelPricing(0.00125, 0.01),
    "gemini-2.5-flash": ModelPricing(0.00015, 0.0006),
    "gemini-2.0-flash": ModelPricing(0.0001, 0.0004),
    # Meta
    "llama-4-maverick": ModelPricing(0.0002, 0.0006),
    "llama-4-scout": ModelPricing(0.00015, 0.0004),
    # NVIDIA
    "nemotron": ModelPricing(0.0, 0.0),
    # DeepSeek
    "deepseek-chat": ModelPricing(0.00014, 0.00028),
    "deepseek-r1": ModelPricing(0.00055, 0.00219),
    # Qwen
    "qwen": ModelPricing(0.00014, 0.00028),
}


class CostEstimator:
    """Estimates cost of an LLM call based on a static pricing table."""

    def __init__(self, overrides: dict[str, ModelPricing] | None = None) -> None:
        self._pricing = {**_PRICING, **(overrides or {})}

    def estimate(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        pricing = self._find_pricing(model)
        if not pricing:
            return 0.0
        return (
            prompt_tokens / 1000 * pricing.prompt_per_1k
            + completion_tokens / 1000 * pricing.completion_per_1k
        )

    def _find_pricing(self, model: str) -> ModelPricing | None:
        model_lower = model.lower()
        # Exact match
        if model_lower in self._pricing:
            return self._pricing[model_lower]
        # Substring match (e.g. "openai/gpt-4o-2024-08-06" matches "gpt-4o")
        for key, pricing in self._pricing.items():
            if key in model_lower:
                return pricing
        return None
