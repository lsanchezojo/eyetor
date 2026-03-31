"""Cost estimation for LLM and image generation API calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Per-token pricing for a model (USD per 1K tokens)."""

    prompt_per_1k: float
    completion_per_1k: float


@dataclass
class ImagePricing:
    """Per-image pricing for an image generation model (USD per image)."""

    cost_per_image: float


# Known LLM pricing — extend as models are added.
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
    # Google Gemini — https://ai.google.dev/gemini-api/docs/pricing
    "gemini-2.0-flash": ModelPricing(0.0001, 0.0004),
    "gemini-2.5-flash": ModelPricing(0.0003, 0.0025),
    "gemini-2.5-flash-lite": ModelPricing(0.0001, 0.0004),
    "gemini-2.5-pro": ModelPricing(0.00125, 0.01),
    "gemini-3-flash": ModelPricing(0.0005, 0.003),
    "gemini-3.1-pro": ModelPricing(0.002, 0.012),
    "gemini-3.1-flash-lite": ModelPricing(0.00025, 0.0015),
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

# Known image generation pricing (USD per image).
_IMAGE_PRICING: dict[str, ImagePricing] = {
    # Gemini image models
    "gemini-2.5-flash": ImagePricing(0.039),
    "gemini-3.1-flash-image": ImagePricing(0.067),
    "gemini-3-pro-image": ImagePricing(0.134),
    # Imagen 4
    "imagen-4-fast": ImagePricing(0.02),
    "imagen-4": ImagePricing(0.04),
    "imagen-4-ultra": ImagePricing(0.06),
    # OpenAI DALL-E
    "dall-e-3": ImagePricing(0.04),
    "dall-e-2": ImagePricing(0.02),
}

# Providers that run local models — always $0 cost.
_LOCAL_PROVIDERS = {"ollama", "llamacpp"}


class CostEstimator:
    """Estimates cost of LLM and image generation calls."""

    def __init__(
        self,
        overrides: dict[str, ModelPricing] | None = None,
        image_overrides: dict[str, ImagePricing] | None = None,
    ) -> None:
        self._pricing = {**_PRICING, **(overrides or {})}
        self._image_pricing = {**_IMAGE_PRICING, **(image_overrides or {})}

    def estimate(
        self, model: str, prompt_tokens: int, completion_tokens: int,
        provider: str | None = None,
    ) -> float:
        """Estimate cost of an LLM call."""
        if provider and provider in _LOCAL_PROVIDERS:
            return 0.0
        pricing = self._find_pricing(model)
        if not pricing:
            return 0.0
        return (
            prompt_tokens / 1000 * pricing.prompt_per_1k
            + completion_tokens / 1000 * pricing.completion_per_1k
        )

    def estimate_image(self, model: str, num_images: int = 1) -> float:
        """Estimate cost of an image generation call."""
        pricing = self._find_image_pricing(model)
        if not pricing:
            return 0.0
        return pricing.cost_per_image * num_images

    def _find_pricing(self, model: str) -> ModelPricing | None:
        model_lower = model.lower()
        if model_lower in self._pricing:
            return self._pricing[model_lower]
        # Substring match (e.g. "openai/gpt-4o-2024-08-06" matches "gpt-4o")
        for key, pricing in self._pricing.items():
            if key in model_lower:
                return pricing
        return None

    def _find_image_pricing(self, model: str) -> ImagePricing | None:
        model_lower = model.lower()
        if model_lower in self._image_pricing:
            return self._image_pricing[model_lower]
        for key, pricing in self._image_pricing.items():
            if key in model_lower:
                return pricing
        return None
