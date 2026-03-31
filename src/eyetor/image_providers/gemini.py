"""Google Gemini image generation provider.

Uses the Gemini API's image generation endpoint (Imagen).
This provider supports dual configuration: it can inherit connection
details from an LLM provider configured in the ``providers`` section.
"""

from __future__ import annotations

import base64
import logging
import time

from eyetor.image_providers.base import BaseImageProvider
from eyetor.models.images import ImageFile, ImageGenerationRequest, ImageGenerationResult

logger = logging.getLogger(__name__)


class GeminiImageProvider(BaseImageProvider):
    """Image provider using Google Gemini's image generation capabilities.

    Endpoint: ``{base_url}/models/{model}:generateContent``
    Uses the Gemini generateContent API with image output modality.
    """

    def _build_headers(self) -> dict[str, str]:
        # Gemini uses API key as query param, not Bearer token
        return {"Content-Type": "application/json"}

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        model = request.model or self.model
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": request.prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        url = f"{self.base_url}/models/{model}:generateContent"
        params = {}
        if self.api_key:
            params["key"] = self.api_key

        t0 = time.monotonic()
        async with self._client() as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._build_headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.monotonic() - t0
        images: list[ImageFile] = []

        # Parse Gemini response: candidates[].content.parts[] may contain inlineData
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("mimeType", "").startswith("image/"):
                    raw = base64.b64decode(inline_data["data"])
                    mime = inline_data["mimeType"]
                    ext = mime.split("/")[-1] if "/" in mime else "png"
                    if ext == "jpeg":
                        ext = "jpg"
                    path = await self._save_image(raw, "gemini", ext=ext)
                    images.append(ImageFile(
                        path=path,
                        width=request.width,
                        height=request.height,
                        format=ext,
                    ))

        return ImageGenerationResult(
            images=images,
            provider="gemini",
            model=model,
            generation_time_s=elapsed,
        )
