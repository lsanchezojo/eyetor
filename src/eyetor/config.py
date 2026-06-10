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
    thinking: bool = False  # Enable thinking/reasoning mode (llamacpp/ollama)
    request_timeout: float = 600.0  # HTTP timeout (s) per chat/completions call
    max_tokens: int | None = None  # local generation cap for all phases
    max_tokens_by_phase: dict[str, int] = {}  # local per-phase overrides
    reasoning_budget: int | None = None  # llama.cpp max tokens for the reasoning block


class TrackingLimits(BaseModel):
    """Limits for a single provider."""

    daily_cost_usd: float | None = None
    daily_tokens: int | None = None


class TrackingConfig(BaseModel):
    """Configuration for usage tracking."""

    db_path: str = "~/.eyetor/tracking.db"
    limits: dict[str, TrackingLimits] = {}
    month_start_day: int = 1
    month_start_hour: int = 0


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
    """Inline definition of an orchestrator worker (legacy/inline form).

    Prefer declaring workers as ``<name>.md`` files under ``agents_dirs`` and
    referencing them by name in ``OrchestratorConfig.workers``. This inline
    form is kept for callers that build workers programmatically.
    """

    provider: str
    model: str
    system_prompt: str = "You are a helpful assistant."


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator auto-delegation.

    ``workers`` is a list of agent names resolved against the
    :class:`AgentRegistry` populated from ``agents_dirs``. Each name must
    correspond to a ``<name>.md`` file in one of those directories.
    """

    auto_delegate: bool = False
    protocol: Literal["tool_calling", "text", "auto"] = "auto"
    workers: list[str] = []


class RouteConfig(BaseModel):
    """Configuration for a single route in the routing system."""

    description: str
    system_prompt: str


class RoutingConfig(BaseModel):
    """Configuration for message routing — classify input and apply a specialized prompt."""

    enabled: bool = False
    classifier_votes: int = 3  # number of voting rounds for reliable classification
    routes: dict[str, RouteConfig] = {}


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
    retry_on: list[str] = [
        "timeout",
        "connection_error",
        "malformed_response",
        "400",
        "408",
        "413",
        "422",
        "429",
        "500",
        "502",
        "503",
        "529",
    ]


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction."""

    enabled: bool = False
    context_window: int = 128_000
    trigger_at_percent: float = 0.80
    tool_output_max_chars: int = 2000
    keep_last_n_user_turns: int = 2
    summary_max_percent: float = 0.05  # 5% of context_window (≈16k chars for 128k)
    archive_dir: str | None = None
    summary_model: str | None = None
    summary_provider: str | None = None


class ChainConfig(BaseModel):
    """Configuration for chain mode — decompose complex queries into steps."""

    mode: Literal["auto", "always", "never"] = "never"
    complexity_threshold: int = 200  # min chars to consider a message complex
    plan_votes: int = 1  # voting rounds for the planning step (1 = no voting)


class ToolGatingConfig(BaseModel):
    """Configuration for conditional (per-turn) tool loading.

    When enabled, token-heavy tool groups (scheduler, image, kb, install,
    delegate) are only sent to the model on turns whose text triggers them.
    Always-on tools (skills, memory) are never gated. ``enabled: false``
    restores the previous behavior of sending every tool every turn.
    """

    enabled: bool = True
    sticky_turns: int = 2  # keep a group active for N turns after it is used


class SessionsConfig(BaseModel):
    """Configuration for session persistence."""

    persist: bool = False
    dir: str = "~/.eyetor/sessions"
    max_messages: int = 200
    chain: ChainConfig = ChainConfig()
    compaction: CompactionConfig = CompactionConfig()
    tool_gating: ToolGatingConfig = ToolGatingConfig()


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


class KnowledgeChunkConfig(BaseModel):
    """Chunker settings."""

    max_chars: int = 1500
    overlap_chars: int = 150


class KnowledgeRetrievalConfig(BaseModel):
    """Hybrid retrieval settings (BM25 + vector fused via RRF)."""

    top_k_default: int = 5
    snippet_chars: int = 400
    rrf_k: int = 60
    candidate_multiplier: int = 3


class KnowledgeEmbeddingConfig(BaseModel):
    """Local embedding model settings (fastembed + sqlite-vec)."""

    enabled: bool = True
    model: str = "intfloat/multilingual-e5-small"
    model_dir: str = "~/.eyetor/models/fastembed"
    dim: int = 384
    batch_size: int = 64


class KnowledgeWorkspaceConfig(BaseModel):
    """A single indexable workspace (named source of documents)."""

    name: str
    path: str
    include: list[str] = []
    exclude: list[str] = []


class KnowledgeConfig(BaseModel):
    """Root config for the knowledge base / RAG subsystem."""

    enabled: bool = False
    db_path: str = "~/.eyetor/knowledge.db"
    auto_reindex_on_start: bool = True
    auto_cwd_workspace: bool = True
    max_file_size_bytes: int = 5 * 1024 * 1024
    workspaces: list[KnowledgeWorkspaceConfig] = []
    chunk: KnowledgeChunkConfig = KnowledgeChunkConfig()
    retrieval: KnowledgeRetrievalConfig = KnowledgeRetrievalConfig()
    embedding: KnowledgeEmbeddingConfig = KnowledgeEmbeddingConfig()


class VectorConfig(BaseModel):
    """Root configuration for Eyetor."""

    providers: dict[str, ProviderConfig] = {}
    fallback: FallbackConfig = FallbackConfig()
    skills_dirs: list[str] = ["./skills"]
    plugins_dirs: list[str] = []
    agents_dirs: list[str] = ["./agents", "~/.eyetor/agents"]
    agent_instructions: str = "~/.eyetor/AGENTS.md"
    memory_db_path: str = "~/.eyetor/memory.db"
    tracking: TrackingConfig = TrackingConfig()
    channels: ChannelsConfig = ChannelsConfig()
    sessions: SessionsConfig = SessionsConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    routing: RoutingConfig = RoutingConfig()
    mcp_servers: dict[str, McpServerConfig] = {}
    image_providers: dict[str, ImageProviderConfig] = {}
    default_image_provider: str | None = None
    knowledge: KnowledgeConfig | None = None
    vision_provider: str | None = (
        None  # provider name (from providers:) used for image description
    )
    vision_model: str | None = (
        None  # model override; uses provider's default model if None
    )
    log_level: str = "INFO"


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
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
    search_paths.extend(
        [
            Path("config/default.yaml"),
            Path.home() / ".eyetor" / "config.yaml",
        ]
    )

    for config_path in search_paths:
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if raw:
                resolved = _resolve_env_vars(raw)
                return VectorConfig(**resolved)

    return VectorConfig()
