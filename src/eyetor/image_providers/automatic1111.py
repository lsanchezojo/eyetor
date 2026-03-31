"""Automatic1111 / Forge WebUI image generation provider.

Uses the Stable Diffusion WebUI API at ``/sdapi/v1/txt2img``.
"""

from __future__ import annotations

import base64
import logging
import time

from eyetor.image_providers.base import BaseImageProvider
from eyetor.models.images import ImageFile, ImageGenerationRequest, ImageGenerationResult

logger = logging.getLogger(__name__)


class Automatic1111ImageProvider(BaseImageProvider):
    """Image provider for Automatic1111 / Forge Stable Diffusion WebUI."""

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        model = request.model or self.model
        payload: dict = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "batch_size": request.num_images,
        }
        if request.steps is not None:
            payload["steps"] = request.steps
        if request.cfg_scale is not None:
            payload["cfg_scale"] = request.cfg_scale
        if request.seed is not None:
            payload["seed"] = request.seed
        if model:
            payload["override_settings"] = {"sd_model_checkpoint": model}

        t0 = time.monotonic()
        async with self._client() as client:
            resp = await client.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                headers=self._build_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.monotonic() - t0
        images: list[ImageFile] = []

        for img_b64 in data.get("images", []):
            raw = base64.b64decode(img_b64)
            path = await self._save_image(raw, "a1111")
            images.append(ImageFile(
                path=path,
                width=request.width,
                height=request.height,
            ))

        return ImageGenerationResult(
            images=images,
            provider="automatic1111",
            model=model,
            generation_time_s=elapsed,
        )
