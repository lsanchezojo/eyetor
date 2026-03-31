"""Configuration loading and management."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    type: Literal["openrouter", "ollama", "llamacpp", "gemini"]
    base_url: str
    api_key: str | None = None
    model: str
    ssl_verify: bool | str = True  # False to disable, or path to CA bundle
    temperature: float = 0.7  # Sampling temperature sent in every request


class TrackingLimits(BaseModel):
    """Limits for a single provider."""

    daily_cost_usd: float | None = None
    daily_tokens: int | None = None


class TrackingConfig(BaseModel):
    """Configuration for usage tracking."""

    db_path: str = "~/.eyetor/tracking.db"
    limits: dict[str, TrackingLimits] = {}


class TelegramAuthConfig(BaseModel):
    """Telegram authentication config."""

    enabled: bool = True
    allowed_users: list[str | int] = []


class TelegramChannelConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    bot_token: str | None = None
    streaming_chunk_size: int = 20
    ssl_verify: bool = True
    auth: TelegramAuthConfig = TelegramAuthConfig()


class CliChannelConfig(BaseModel):
    """CLI channel configuration."""

    host_tools: bool = True  # Enable shell/filesystem/browser/web-search skills


class ChannelsConfig(BaseModel):
    """Configuration for communication channels."""

    cli: CliChannelConfig = CliChannelConfig()
    telegram: TelegramChannelConfig = TelegramChannelConfig()


class OrchestratorWorkerConfig(BaseModel):
    """Configuration for a single orchestrator worker."""

    provider: str
    model: str
    system_prompt: str = "You are a helpful assistant."


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator auto-delegation."""

    auto_delegate: bool = False
    workers: dict[str, OrchestratorWorkerConfig] = {}


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    transport: Literal["stdio", "http"]
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None


class FallbackConfig(BaseModel):
    """Fallback chain configuration."""

    fallback_chain: list[str] = []
    retry_on: list[str] = ["timeout", "connection_error", "500", "502", "503", "529"]


class SchedulerConfig(BaseModel):
    """Configuration for the task scheduler."""

    enabled: bool = True
    db_path: str = "~/.eyetor/scheduler.db"
    default_timezone: str = "Europe/Madrid"


class ImageProviderConfig(BaseModel):
    """Configuration for a single image generation provider.

    When ``provider`` is set, connection details (base_url, api_key, ssl_verify)
    are inherited from the named LLM provider in ``providers`` unless explicitly
    overridden here.
    """

    type: Literal["openai_compat", "gemini", "automatic1111", "comfyui"]
    provider: str | None = None  # reference to an LLM provider for shared config
    base_url: str | None = None
    api_key: str | None = None
    model: str = ""
    ssl_verify: bool | str = True
    output_dir: str = "~/.eyetor/generated_images"
    default_timeout: float = 300.0
    workflow_template: str | None = None  # ComfyUI only
    extra_params: dict[str, Any] = {}


class VectorConfig(BaseModel):
    """Root configuration for Eyetor."""

    providers: dict[str, ProviderConfig] = {}
    default_provider: str = "ollama"
    fallback: FallbackConfig = FallbackConfig()
    skills_dirs: list[str] = ["./skills"]
    memory_db_path: str = "~/.eyetor/memory.db"
    tracking: TrackingConfig = TrackingConfig()
    channels: ChannelsConfig = ChannelsConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    mcp_servers: dict[str, McpServerConfig] = {}
    image_providers: dict[str, ImageProviderConfig] = {}
    default_image_provider: str | None = None
    log_level: str = "INFO"


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _load_dotenv() -> None:
    """Load .env file into os.environ if present. Searches .env then ~/.eyetor/.env."""
    for env_path in [Path(".env"), Path.home() / ".eyetor" / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:  # don't override existing env vars
                    os.environ[key] = value


def load_config(path: Path | None = None) -> VectorConfig:
    """Load configuration from YAML file with env var substitution.

    Search order:
    1. Explicit path argument
    2. ./config/default.yaml
    3. ~/.eyetor/config.yaml
    4. Empty defaults
    """
    _load_dotenv()

    search_paths = []
    if path:
        search_paths.append(path)
    search_paths.extend([
        Path("config/default.yaml"),
        Path.home() / ".eyetor" / "config.yaml",
    ])

    for config_path in search_paths:
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if raw:
                resolved = _resolve_env_vars(raw)
                return VectorConfig(**resolved)

    return VectorConfig()
