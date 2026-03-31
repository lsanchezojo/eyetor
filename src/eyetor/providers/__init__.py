"""LLM provider abstraction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import OpenRouterProvider
from eyetor.providers.ollama import OllamaProvider
from eyetor.providers.llamacpp import LlamaCppProvider
from eyetor.providers.gemini import GeminiProvider
from eyetor.providers.fallback import FallbackProvider
from eyetor.providers.tracking import TrackingProvider, UsageLimitExceeded
from eyetor.config import ProviderConfig, VectorConfig
import logging

if TYPE_CHECKING:
    from eyetor.tracking.pricing import CostEstimator
    from eyetor.tracking.usage import UsageTracker

logger = logging.getLogger(__name__)

__all__ = [
    "BaseProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "LlamaCppProvider",
    "GeminiProvider",
    "FallbackProvider",
    "TrackingProvider",
    "UsageLimitExceeded",
    "create_provider",
    "get_provider",
    "get_fallback_provider",
]

_PROVIDER_MAP = {
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
    "gemini": GeminiProvider,
}


def create_provider(config: ProviderConfig) -> BaseProvider:
    """Factory: config -> concrete provider instance."""
    cls = _PROVIDER_MAP.get(config.type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {config.type}")
    return cls(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        ssl_verify=config.ssl_verify,
        temperature=config.temperature,
    )


def get_provider(
    config: VectorConfig,
    name: str | None = None,
    tracker: "UsageTracker | None" = None,
    cost_estimator: "CostEstimator | None" = None,
) -> BaseProvider:
    """Return a provider by name (or the default provider)."""
    name = name or config.default_provider
    if name not in config.providers:
        raise KeyError(f"Provider '{name}' not found. Available: {list(config.providers)}")
    prov = create_provider(config.providers[name])
    if tracker:
        prov = TrackingProvider(prov, tracker, name, cost_estimator)
    return prov


def get_fallback_provider(
    config: VectorConfig,
    tracker: "UsageTracker | None" = None,
    cost_estimator: "CostEstimator | None" = None,
) -> FallbackProvider:
    """Build a FallbackProvider from the fallback_chain in config."""
    chain = config.fallback.fallback_chain or [config.default_provider]
    providers: list[BaseProvider] = []
    for name in chain:
        if name in config.providers:
            prov = create_provider(config.providers[name])
            if tracker:
                prov = TrackingProvider(prov, tracker, name, cost_estimator)
            providers.append(prov)
        else:
            logger.warning("Fallback chain references unknown provider: %s", name)
    if not providers:
        raise ValueError("No valid providers found for fallback chain")
    return FallbackProvider(
        providers=providers,
        retry_on=set(config.fallback.retry_on),
    )
