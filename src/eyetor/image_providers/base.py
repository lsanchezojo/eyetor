"""Abstract base class for image generation providers."""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from eyetor.models.images import ImageFile, ImageGenerationRequest, ImageGenerationResult

logger = logging.getLogger(__name__)


class BaseImageProvider(ABC):
    """Abstract base for all image generation providers.

    Follows the same patterns as ``BaseProvider`` for LLMs:
    httpx async client, configurable SSL, auth headers.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str | None = None,
        ssl_verify: bool | str = True,
        output_dir: str = "~/.eyetor/generated_images",
        default_timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.output_dir = Path(output_dir).expanduser()
        self.default_timeout = default_timeout

    @abstractmethod
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image(s) from a text prompt."""

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        """Return a configured async HTTP client."""
        t = timeout or self.default_timeout
        if not self.ssl_verify:
            logger.warning(
                "SSL verification disabled for %s — insecure.", self.base_url
            )
        return httpx.AsyncClient(timeout=t, verify=self.ssl_verify)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _save_image(self, data: bytes, provider_name: str, ext: str = "png") -> Path:
        """Save raw image bytes to output_dir and return the file path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        h = hashlib.md5(data).hexdigest()[:8]
        filename = f"{provider_name}_{ts}_{h}.{ext}"
        path = self.output_dir / filename
        path.write_bytes(data)
        logger.info("Image saved: %s", path)
        return path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
