"""Image provider abstraction layer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eyetor.image_providers.base import BaseImageProvider
from eyetor.image_providers.openai_compat import OpenAICompatImageProvider
from eyetor.image_providers.gemini import GeminiImageProvider
from eyetor.image_providers.automatic1111 import Automatic1111ImageProvider
from eyetor.image_providers.comfyui import ComfyUIImageProvider

if TYPE_CHECKING:
    from eyetor.config import ImageProviderConfig, VectorConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BaseImageProvider",
    "OpenAICompatImageProvider",
    "GeminiImageProvider",
    "Automatic1111ImageProvider",
    "ComfyUIImageProvider",
    "create_image_provider",
    "get_image_provider",
]

_IMAGE_PROVIDER_MAP: dict[str, type[BaseImageProvider]] = {
    "openai_compat": OpenAICompatImageProvider,
    "gemini": GeminiImageProvider,
    "automatic1111": Automatic1111ImageProvider,
    "comfyui": ComfyUIImageProvider,
}


def create_image_provider(
    config: ImageProviderConfig,
    root_config: VectorConfig | None = None,
) -> BaseImageProvider:
    """Factory: ImageProviderConfig -> concrete image provider instance.

    When ``config.provider`` references an LLM provider name, connection
    details (base_url, api_key, ssl_verify) are inherited from that provider
    unless explicitly set in the image config.
    """
    base_url = config.base_url
    api_key = config.api_key
    ssl_verify = config.ssl_verify

    # Inherit from referenced LLM provider
    if config.provider and root_config:
        llm_cfg = root_config.providers.get(config.provider)
        if llm_cfg:
            if base_url is None:
                base_url = llm_cfg.base_url
            if api_key is None:
                api_key = llm_cfg.api_key
            if config.ssl_verify is True and llm_cfg.ssl_verify is not True:
                ssl_verify = llm_cfg.ssl_verify
        else:
            logger.warning(
                "Image provider references unknown LLM provider '%s'",
                config.provider,
            )

    if base_url is None:
        raise ValueError(
            f"Image provider config has no base_url and no provider reference to inherit from"
        )

    cls = _IMAGE_PROVIDER_MAP.get(config.type)
    if cls is None:
        raise ValueError(f"Unknown image provider type: {config.type}")

    kwargs: dict = dict(
        base_url=base_url,
        model=config.model,
        api_key=api_key,
        ssl_verify=ssl_verify,
        output_dir=config.output_dir,
        default_timeout=config.default_timeout,
    )

    # ComfyUI-specific: workflow template
    if config.type == "comfyui" and config.workflow_template:
        kwargs["workflow_template"] = config.workflow_template

    return cls(**kwargs)


def get_image_provider(
    config: VectorConfig,
    name: str | None = None,
) -> BaseImageProvider:
    """Return an image provider by name (or the default)."""
    name = name or config.default_image_provider
    if not name:
        raise KeyError("No image provider specified and no default_image_provider configured")
    if name not in config.image_providers:
        raise KeyError(
            f"Image provider '{name}' not found. Available: {list(config.image_providers)}"
        )
    return create_image_provider(config.image_providers[name], config)
