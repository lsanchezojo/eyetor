"""Image generation tool backed by the configured image providers."""

from __future__ import annotations

import json

from eyetor.models.tools import ToolDefinition, ToolRegistry


def register_image_tool(tool_registry: ToolRegistry, cfg, tracker, cost_estimator) -> str:
    """Register ``generate_image``; return the system-prompt note it needs ("" if off)."""
    if not (cfg.default_image_provider and cfg.image_providers):
        return ""
    from eyetor.image_providers import get_image_provider
    from eyetor.models.images import ImageGenerationRequest

    _img_provider_names = list(cfg.image_providers.keys())

    async def generate_image_handler(
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
        provider: str | None = None,
    ) -> str:
        from eyetor.providers.tracking import current_session_id

        img_prov = get_image_provider(cfg, provider)
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )
        result = await img_prov.generate(request)
        img = result.images[0]

        # Track image generation usage
        duration_ms = int((result.generation_time_s or 0) * 1000)
        img_cost = cost_estimator.estimate_image(
            result.model, num_images=len(result.images)
        )
        tracker.record(
            session_id=current_session_id.get(),
            provider=f"image:{result.provider}",
            model=result.model,
            prompt_tokens=0,
            completion_tokens=0,
            estimated_cost=img_cost,
            duration_ms=duration_ms,
            finish_reason="image_generated",
        )

        return json.dumps(
            {
                "status": "ok",
                "image_path": str(img.path),
                "width": img.width,
                "height": img.height,
                "provider": result.provider,
                "model": result.model,
            }
        )

    tool_registry.register(
        ToolDefinition(
            name="generate_image",
            description=(
                "Generate an image from a text prompt. Returns local file path. "
                f"Available image providers: {', '.join(_img_provider_names)}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What to avoid in the image (optional)",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default 1024)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default 1024)",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of generation steps (optional, provider default)",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                    "provider": {
                        "type": "string",
                        "description": f"Image provider name. One of: {', '.join(_img_provider_names)} (optional, uses default)",
                    },
                },
                "required": ["prompt"],
            },
            handler=generate_image_handler,
            group="image",
        )
    )

    return (
        "\n\nWhen you generate an image with generate_image, include the path in your "
        "reply as [IMAGE:/path/to/file.png] so it is shown to the user correctly."
    )
