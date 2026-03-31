"""Models for image generation requests and results."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    """Parameters for an image generation request."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    num_images: int = 1
    model: str | None = None  # override provider default


class ImageFile(BaseModel):
    """A single generated image on disk."""

    path: Path
    url: str | None = None
    width: int
    height: int
    format: str = "png"


class ImageGenerationResult(BaseModel):
    """Result of an image generation call."""

    images: list[ImageFile]
    provider: str
    model: str
    generation_time_s: float | None = None
