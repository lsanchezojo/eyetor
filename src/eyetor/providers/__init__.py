"""LLM provider abstraction layer."""

from eyetor.providers.base import BaseProvider
from eyetor.providers.openrouter import OpenRouterProvider
from eyetor.providers.ollama import OllamaProvider
from eyetor.providers.llamacpp import LlamaCppProvider
from eyetor.providers.fallback import FallbackProvider
from eyetor.config import ProviderConfig, VectorConfig
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "BaseProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "LlamaCppProvider",
    "FallbackProvider",
    "create_provider",
    "get_provider",
    "get_fallback_provider",
]

_PROVIDER_MAP = {
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
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


def get_provider(config: VectorConfig, name: str | None = None) -> BaseProvider:
    """Return a provider by name (or the default provider)."""
    name = name or config.default_provider
    if name not in config.providers:
        raise KeyError(f"Provider '{name}' not found. Available: {list(config.providers)}")
    return create_provider(config.providers[name])


def get_fallback_provider(config: VectorConfig) -> FallbackProvider:
    """Build a FallbackProvider from the fallback_chain in config."""
    chain = config.fallback.fallback_chain or [config.default_provider]
    providers = []
    for name in chain:
        if name in config.providers:
            providers.append(create_provider(config.providers[name]))
        else:
            logger.warning("Fallback chain references unknown provider: %s", name)
    if not providers:
        raise ValueError("No valid providers found for fallback chain")
    return FallbackProvider(
        providers=providers,
        retry_on=set(config.fallback.retry_on),
    )
