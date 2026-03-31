"""OpenAI-compatible image generation provider.

Covers any service exposing ``/v1/images/generations``:
NanoBanano, Together AI, OpenAI DALL-E, etc.
"""

from __future__ import annotations

import base64
import logging
import time

from eyetor.image_providers.base import BaseImageProvider
from eyetor.models.images import ImageFile, ImageGenerationRequest, ImageGenerationResult

logger = logging.getLogger(__name__)


class OpenAICompatImageProvider(BaseImageProvider):
    """Image provider for OpenAI-compatible ``/v1/images/generations`` APIs."""

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        model = request.model or self.model
        payload: dict = {
            "model": model,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
        }

        t0 = time.monotonic()
        async with self._client() as client:
            resp = await client.post(
                f"{self.base_url}/images/generations",
                json=payload,
                headers=self._build_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.monotonic() - t0
        images: list[ImageFile] = []

        for item in data.get("data", []):
            if "b64_json" in item:
                raw = base64.b64decode(item["b64_json"])
            elif "url" in item:
                # Fallback: download from URL
                async with self._client(timeout=60.0) as dl_client:
                    dl_resp = await dl_client.get(item["url"])
                    dl_resp.raise_for_status()
                    raw = dl_resp.content
            else:
                logger.warning("Image item has neither b64_json nor url, skipping")
                continue

            path = await self._save_image(raw, "openai_compat")
            images.append(ImageFile(
                path=path,
                url=item.get("url"),
                width=request.width,
                height=request.height,
            ))

        return ImageGenerationResult(
            images=images,
            provider="openai_compat",
            model=model,
            generation_time_s=elapsed,
        )
