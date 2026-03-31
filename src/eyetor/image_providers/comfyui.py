"""ComfyUI image generation provider.

Uses the ComfyUI REST API:
1. POST /prompt — queue a workflow
2. WebSocket /ws — monitor progress
3. GET /history/{id} — retrieve results
4. GET /view — download generated images
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from eyetor.image_providers.base import BaseImageProvider
from eyetor.models.images import ImageFile, ImageGenerationRequest, ImageGenerationResult

logger = logging.getLogger(__name__)

# Minimal txt2img workflow template — uses KSampler + VAEDecode + SaveImage
_DEFAULT_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ""},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "eyetor", "images": ["8", 0]},
    },
}


class ComfyUIImageProvider(BaseImageProvider):
    """Image provider for ComfyUI workflow-based generation.

    Optionally loads a custom workflow template from ``workflow_template`` path.
    Falls back to a built-in minimal txt2img workflow.
    """

    def __init__(self, workflow_template: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._workflow_template = workflow_template

    def _load_workflow(self) -> dict:
        """Load workflow JSON from file or use default."""
        if self._workflow_template:
            p = Path(self._workflow_template).expanduser()
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
            logger.warning("Workflow template not found: %s, using default", p)
        return json.loads(json.dumps(_DEFAULT_WORKFLOW))  # deep copy

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        workflow = self._load_workflow()
        model = request.model or self.model

        # Patch workflow with request parameters
        if "4" in workflow:
            workflow["4"]["inputs"]["ckpt_name"] = model
        if "5" in workflow:
            workflow["5"]["inputs"]["width"] = request.width
            workflow["5"]["inputs"]["height"] = request.height
            workflow["5"]["inputs"]["batch_size"] = request.num_images
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = request.prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = request.negative_prompt
        if "3" in workflow:
            if request.steps is not None:
                workflow["3"]["inputs"]["steps"] = request.steps
            if request.cfg_scale is not None:
                workflow["3"]["inputs"]["cfg"] = request.cfg_scale
            if request.seed is not None:
                workflow["3"]["inputs"]["seed"] = request.seed

        client_id = str(uuid.uuid4())
        prompt_payload = {"prompt": workflow, "client_id": client_id}

        t0 = time.monotonic()

        # Queue the prompt
        async with self._client() as client:
            resp = await client.post(
                f"{self.base_url}/prompt",
                json=prompt_payload,
                headers=self._build_headers(),
            )
            resp.raise_for_status()
            prompt_data = resp.json()
            prompt_id = prompt_data["prompt_id"]

        # Poll /history until the prompt completes
        images: list[ImageFile] = []
        async with self._client() as client:
            while True:
                resp = await client.get(f"{self.base_url}/history/{prompt_id}")
                resp.raise_for_status()
                history = resp.json()

                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        for img_info in node_output.get("images", []):
                            filename = img_info["filename"]
                            subfolder = img_info.get("subfolder", "")
                            img_type = img_info.get("type", "output")

                            params = {
                                "filename": filename,
                                "subfolder": subfolder,
                                "type": img_type,
                            }
                            img_resp = await client.get(
                                f"{self.base_url}/view",
                                params=params,
                            )
                            img_resp.raise_for_status()

                            path = await self._save_image(
                                img_resp.content, "comfyui"
                            )
                            images.append(ImageFile(
                                path=path,
                                width=request.width,
                                height=request.height,
                            ))
                    break

                import asyncio
                await asyncio.sleep(1.0)

        elapsed = time.monotonic() - t0

        return ImageGenerationResult(
            images=images,
            provider="comfyui",
            model=model,
            generation_time_s=elapsed,
        )
