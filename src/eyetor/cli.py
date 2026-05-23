"""Eyetor CLI — command-line interface for the multi-agent system."""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(level: str, *, interactive: bool = False) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    if interactive:
        # In interactive CLI mode, suppress INFO logs to avoid polluting the
        # Rich console.  Only WARNING+ go to stderr.
        log_level = max(log_level, logging.WARNING)

    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setLevel(log_level)


def _load_cfg(config_path: str | None):
    from eyetor.config import load_config

    path = Path(config_path) if config_path else None
    return load_config(path)


def _default_cli_session_id() -> str:
    """Default CLI session id: stable while the parent shell lives."""
    return f"cli-{getpass.getuser()}-{os.getppid()}"


def _resolve_cli_session_id(session_flag: str | None) -> str:
    """Resolve the CLI session id (precedence: --session > $EYETOR_SESSION > default)."""
    if session_flag:
        return session_flag
    env = os.environ.get("EYETOR_SESSION")
    if env:
        return env
    return _default_cli_session_id()


def _resolve_provider(cfg, provider, model, tracker, cost_estimator):
    """Resolve the runtime provider based on CLI flags and config.

    Rules:
      - ``--provider X``: single provider X (with optional ``--model Y`` override).
      - ``--model Y`` alone: single provider = first entry of ``fallback_chain``.
      - No flags: full ``FallbackProvider`` over ``fallback_chain``.
    """
    from eyetor.providers import get_provider, get_fallback_provider

    chain = cfg.fallback.fallback_chain

    if provider is None and model is None:
        if not chain:
            raise click.UsageError(
                "No fallback_chain configured and no --provider given. "
                "Add providers to fallback.fallback_chain or pass --provider."
            )
        logging.getLogger(__name__).info("Fallback provider chain: %s", chain)
        return get_fallback_provider(
            cfg, tracker=tracker, cost_estimator=cost_estimator
        )

    target = provider
    if target is None:
        if not chain:
            raise click.UsageError(
                "--model requires --provider when no fallback_chain is configured."
            )
        target = chain[0]

    prov = get_provider(cfg, target, tracker=tracker, cost_estimator=cost_estimator)
    if model:
        prov.model = model
        if hasattr(prov, "_inner"):
            prov._inner.model = model
    return prov


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file.")
@click.option("--log-level", default=None, help="Log level (DEBUG/INFO/WARNING/ERROR).")
@click.pass_context
def cli(ctx: click.Context, config: str | None, log_level: str | None) -> None:
    """Eyetor — Multi-agent AI system based on Anthropic's agent patterns."""
    ctx.ensure_object(dict)
    cfg = _load_cfg(config)
    if log_level:
        cfg.log_level = log_level.upper()
    _setup_logging(cfg.log_level)
    ctx.obj["cfg"] = cfg


# ---------------------------------------------------------------------------
# eyetor start  (unified entry point)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--provider", "-p", default=None, help="Provider name (default from config)."
)
@click.option("--model", "-m", default=None, help="Model override.")
@click.option(
    "--host-tools/--no-host-tools",
    default=None,
    help="Enable/disable host skills (shell, filesystem, browser). Default from config.",
)
@click.option(
    "--session",
    "session_id",
    default=None,
    help=(
        "Persistent CLI session id. Resumes if it already exists. "
        "Precedence: --session > $EYETOR_SESSION > cli-<user>-<ppid>."
    ),
)
@click.pass_context
def start(
    ctx: click.Context,
    provider: str | None,
    model: str | None,
    host_tools: bool | None,
    session_id: str | None,
) -> None:
    """Start the agent — launches all configured channels (CLI and/or Telegram)."""
    import sys

    cfg = ctx.obj["cfg"]
    resolved_session_id = _resolve_cli_session_id(session_id)

    # --host-tools flag overrides config; config default is True
    use_host_tools = (
        host_tools if host_tools is not None else cfg.channels.cli.host_tools
    )
    interactive = sys.stdin.isatty()

    if interactive:
        # Reconfigure logging: suppress INFO noise in interactive CLI mode
        _setup_logging(cfg.log_level, interactive=True)

    async def _run():
        from eyetor.models.agents import AgentConfig
        from eyetor.chat.manager import SessionManager
        from eyetor.models.tools import ToolRegistry, ToolDefinition
        from eyetor.memory.manager import MemoryManager
        from eyetor.skills.registry import SkillRegistry
        from eyetor.skills.executor import run_script, DEFAULT_TIMEOUT
        from eyetor.tracking.usage import UsageTracker
        from eyetor.tracking.pricing import CostEstimator
        from eyetor.runtime import runtime_path, write_snapshot

        # In interactive (CLI) mode the systemd service is usually the snapshot
        # owner — only write it if no snapshot exists yet, to avoid races.
        if not interactive or not runtime_path().exists():
            write_snapshot(cfg)

        tracker = UsageTracker.from_config(cfg.tracking)
        cost_estimator = CostEstimator()
        prov = _resolve_provider(cfg, provider, model, tracker, cost_estimator)

        memory = MemoryManager.from_path(cfg.memory_db_path)

        # Knowledge base (optional, hybrid BM25 + semantic retrieval)
        knowledge = None
        if cfg.knowledge and cfg.knowledge.enabled and cfg.knowledge.workspaces:
            from eyetor.knowledge.manager import KnowledgeManager

            knowledge = KnowledgeManager.from_config(cfg.knowledge)
            if interactive and cfg.knowledge.auto_cwd_workspace:
                knowledge.register_cwd_workspace(Path.cwd())

        # Shared skill registry
        skill_reg = SkillRegistry()
        skill_reg.discover(cfg.skills_dirs)
        skill_names = skill_reg.list_names()

        # Shared subagent registry (Markdown files under agents_dirs)
        from eyetor.agents.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.discover(cfg.agents_dirs)

        # System prompt
        if use_host_tools:
            base_system = (
                "Eres Eyetor (suena como Aitor), un asistente de IA con acceso a herramientas que pueden actuar en el ordenador del usuario. "
                "Puedes ejecutar comandos de shell, gestionar ficheros, abrir URLs y buscar en internet. "
                "Responde siempre en español de España. "
                "Explica brevemente lo que vas a hacer antes de hacerlo. "
                "Pide confirmación antes de operaciones destructivas (borrar, sobreescribir, formatear).\n\n"
                "## Uso de herramientas\n\n"
                "Cuando el usuario pida información que requiera varias llamadas a herramientas, encadénalas tú mismo sin pedir aclaraciones innecesarias. "
                "No le pidas al usuario que elija qué comando ejecutar ni le muestres la sintaxis de los scripts: ejecútalos directamente. "
                "Si necesitas varios pasos (por ejemplo, listar tiendas y luego consultar precios de cada una), hazlos todos seguidos. "
                "Solo pregunta al usuario cuando haya una ambigüedad real que no puedas resolver con los datos disponibles."
            )
        else:
            base_system = (
                "Eres Eyetor (suena como Aitor), un asistente de IA útil. "
                "Responde siempre en español de España.\n\n"
                "## Uso de herramientas\n\n"
                "Cuando el usuario pida información que requiera varias llamadas a herramientas, encadénalas tú mismo sin pedir aclaraciones innecesarias. "
                "No le pidas al usuario que elija qué comando ejecutar ni le muestres la sintaxis de los scripts: ejecútalos directamente. "
                "Si necesitas varios pasos (por ejemplo, listar tiendas y luego consultar precios de cada una), hazlos todos seguidos. "
                "Solo pregunta al usuario cuando haya una ambigüedad real que no puedas resolver con los datos disponibles."
            )

        # Inject current date/time so the model knows "today"
        from zoneinfo import ZoneInfo
        _DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        _MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                      "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        tz = ZoneInfo(cfg.scheduler.default_timezone)
        now = datetime.now(tz)
        now_str = (
            f"{_DAYS_ES[now.weekday()]} {now.day} de {_MONTHS_ES[now.month - 1]} "
            f"de {now.year}, {now.strftime('%H:%M')} ({now.tzname()})"
        )
        base_system = f"{base_system}\n\nFecha y hora actual: {now_str}"

        skills_context = skill_reg.build_skills_context(skill_names)
        if skills_context:
            base_system = f"{base_system}\n\n{skills_context}"

        # Agent instructions (user-managed file with custom behavior rules)
        instructions_path = Path(cfg.agent_instructions).expanduser()
        if instructions_path.exists():
            instructions_text = instructions_path.read_text(encoding="utf-8").strip()
            if instructions_text:
                base_system = (
                    f"{base_system}\n\n## Agent Instructions\n\n{instructions_text}"
                )
                logging.getLogger(__name__).info(
                    "Loaded agent instructions from %s", instructions_path
                )

        # Plugin registry
        plugin_registry = None
        if cfg.plugins_dirs:
            from eyetor.plugins.registry import PluginRegistry

            plugin_registry = PluginRegistry()
            plugin_registry.load_all(cfg.plugins_dirs)
            await plugin_registry.run_init()

        # Shared tool registry
        tool_registry = ToolRegistry(plugin_registry=plugin_registry)
        if skill_names:
            from eyetor.skills.router import RoutingError, ScriptRouter

            # Register one tool per skill — always just an "args" param.
            # ScriptRouter handles mapping the first token to the correct
            # script for multi-script skills; single-script skills pass
            # everything through unchanged.
            for _sn in skill_names:
                _meta = skill_reg.get_metadata(_sn)
                _scripts = skill_reg.list_scripts(_sn)
                if not _scripts:
                    continue

                _router = ScriptRouter(_sn, _scripts)

                def _make_handler(skill_name: str, router: ScriptRouter, meta):
                    async def _handler(args: str = "") -> str:
                        try:
                            script_path, arg_list = router.route(args)
                        except RoutingError as exc:
                            return json.dumps({"error": str(exc)})
                        routed_args = " ".join(arg_list)
                        if use_host_tools and _is_dangerous(skill_name, script_path.name, routed_args):
                            confirmed = await _ask_confirm(skill_name, script_path.name, routed_args)
                            if not confirmed:
                                return json.dumps({"error": "Operation cancelled by user."})
                        timeout = meta.timeout if meta.timeout is not None else DEFAULT_TIMEOUT
                        return await run_script(script_path, arg_list, timeout=timeout)
                    return _handler

                _handler = _make_handler(_sn, _router, _meta)

                _desc = _meta.description[:200]
                _public = _router.public_scripts
                if len(_public) > 1:
                    _desc += f" Scripts: {', '.join(s.stem for s in _public)}."

                tool_registry.register(
                    ToolDefinition(
                        name=f"skill_{_sn.replace('-', '_')}",
                        description=_desc,
                        parameters={
                            "type": "object",
                            "properties": {
                                "args": {
                                    "type": "string",
                                    "description": (
                                        "Subcommand and flags to pass to the skill script. "
                                        "Pass only the arguments — do NOT include the script name, "
                                        "interpreter, or wrapper variables (e.g. $PWCLI, bash, python). "
                                        "Example: 'open https://example.com --headed'"
                                    ),
                                },
                            },
                            "required": [],
                        },
                        handler=_handler,
                    )
                )

        # Image generation tool
        if cfg.default_image_provider and cfg.image_providers:
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
                )
            )

            base_system += (
                "\n\nCuando generes una imagen con generate_image, incluye la ruta en tu respuesta "
                "con el formato [IMAGE:/ruta/al/archivo.png] para que se muestre correctamente al usuario."
            )

        # Knowledge base tools
        if knowledge is not None:
            _kb_ws_names = knowledge.list_workspaces()
            _kb_ws_desc = ", ".join(_kb_ws_names) if _kb_ws_names else "none"

            async def kb_search_handler(
                query: str, top_k: int = 5, workspace: str | None = None
            ) -> str:
                top_k = max(1, min(int(top_k or 5), 20))
                hits = await knowledge.search(query, workspace=workspace, top_k=top_k)
                return json.dumps(
                    {
                        "query": query,
                        "count": len(hits),
                        "results": [
                            {
                                "doc_id": h.doc_id,
                                "chunk_id": h.chunk_id,
                                "workspace": h.workspace,
                                "path": h.path,
                                "title": h.title,
                                "heading": h.heading,
                                "snippet": h.snippet,
                                "score": round(h.score, 4),
                                "sources": h.sources,
                            }
                            for h in hits
                        ],
                    },
                    ensure_ascii=False,
                )

            async def kb_read_handler(
                doc_id: int, section: str | None = None, max_chars: int = 1800
            ) -> str:
                result = knowledge.read_doc(
                    int(doc_id), section=section, max_chars=int(max_chars or 1800)
                )
                if result is None:
                    return json.dumps(
                        {"error": f"Document {doc_id} not found"}, ensure_ascii=False
                    )
                if not result.section_matched:
                    return json.dumps(
                        {
                            "error": (
                                f"Section '{section}' not found in document {doc_id}."
                            ),
                            "doc_id": result.doc_id,
                            "path": result.path,
                            "title": result.title,
                            "available_sections": result.available_sections or [],
                            "hint": (
                                "Call kb_read without the 'section' parameter to "
                                "read the whole document, or pick one of "
                                "available_sections verbatim."
                            ),
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "doc_id": result.doc_id,
                        "path": result.path,
                        "title": result.title,
                        "section": result.section,
                        "content": result.content,
                        "truncated": result.truncated,
                        "total_chars": result.total_chars,
                    },
                    ensure_ascii=False,
                )

            async def kb_list_sources_handler(
                workspace: str | None = None, limit: int = 50
            ) -> str:
                sources = knowledge.list_sources(
                    workspace=workspace, limit=int(limit or 50)
                )
                return json.dumps(sources, ensure_ascii=False)

            tool_registry.register(
                ToolDefinition(
                    name="kb_search",
                    description=(
                        "Search the workspace knowledge base for documentation, guides, "
                        "specs and notes. Returns ranked snippets. Use this for conceptual "
                        "or factual questions about project docs. For literal string matching "
                        "in code, prefer the filesystem grep skill. "
                        f"Available workspaces: {_kb_ws_desc}."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Query string. Supports FTS5 syntax: AND, OR, NEAR, "
                                    '"quoted phrase".'
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (1-20, default 5).",
                            },
                            "workspace": {
                                "type": "string",
                                "description": f"Optional workspace filter. One of: {_kb_ws_desc}.",
                            },
                        },
                        "required": ["query"],
                    },
                    handler=kb_search_handler,
                )
            )

            tool_registry.register(
                ToolDefinition(
                    name="kb_read",
                    description=(
                        "Read a specific document or section by id from the knowledge "
                        "base. Use after kb_search to get more context around a match."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "integer",
                                "description": "Document id (from kb_search results).",
                            },
                            "section": {
                                "type": "string",
                                "description": "Optional heading-path prefix to narrow the read.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 1800).",
                            },
                        },
                        "required": ["doc_id"],
                    },
                    handler=kb_read_handler,
                )
            )

            tool_registry.register(
                ToolDefinition(
                    name="kb_list_sources",
                    description=(
                        "List indexed documents in the knowledge base. Use to discover "
                        "what documentation is available before searching."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "workspace": {
                                "type": "string",
                                "description": f"Optional workspace filter. One of: {_kb_ws_desc}.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of documents (default 50).",
                            },
                        },
                        "required": [],
                    },
                    handler=kb_list_sources_handler,
                )
            )

        # MCP servers — connect and register tools
        mcp_registry = None
        if cfg.mcp_servers:
            from eyetor.mcp.registry import McpRegistry

            mcp_registry = McpRegistry(cfg.mcp_servers)
            await mcp_registry.connect_all()
            mcp_registry.register_all_into(tool_registry)
            report = mcp_registry.get_degraded_report()
            if report.is_degraded:
                degraded_text = report.format_for_prompt()
                base_system = f"{base_system}\n\n{degraded_text}"
                logging.getLogger(__name__).warning(
                    "MCP degraded — failed servers: %s", list(report.failed.keys())
                )

        # Subagent delegation (auto_delegate): expose a `delegate` tool that
        # routes subtasks to named subagents loaded from agents_dirs.
        orch_cfg = cfg.orchestrator
        if orch_cfg.auto_delegate:
            _requested = list(orch_cfg.workers)
            _resolved_workers = []
            _logger = logging.getLogger(__name__)
            for _wn in _requested:
                if agent_reg.has(_wn):
                    _resolved_workers.append(agent_reg.get(_wn))
                else:
                    _logger.warning(
                        "orchestrator.workers references unknown agent %r — "
                        "skipping (available: %s)",
                        _wn, ", ".join(agent_reg.list_names()) or "<none>",
                    )

            if not _requested:
                _logger.warning(
                    "auto_delegate is enabled but orchestrator.workers is empty — "
                    "no delegate tool registered"
                )
            elif not _resolved_workers:
                _logger.warning(
                    "auto_delegate is enabled but none of the configured workers "
                    "were found in agents_dirs — no delegate tool registered"
                )
            else:
                from eyetor.agents.base import BaseAgent
                from eyetor.models.agents import AgentConfig as _WorkerAgentCfg

                _workers_by_name = {w.name: w for w in _resolved_workers}
                _worker_names_list = list(_workers_by_name.keys())

                async def delegate_handler(worker: str, task: str) -> str:
                    wdef = _workers_by_name.get(worker)
                    if wdef is None:
                        return json.dumps({
                            "error": (
                                f"Unknown worker {worker!r}. "
                                f"Available: {_worker_names_list}"
                            )
                        }, ensure_ascii=False)
                    if not task or not task.strip():
                        return json.dumps(
                            {"error": "task is required"}, ensure_ascii=False
                        )
                    sub_temp = (
                        wdef.temperature
                        if wdef.temperature is not None
                        else prov.temperature
                    )
                    sub_model = wdef.model or prov.model
                    sub_agent = BaseAgent(
                        config=_WorkerAgentCfg(
                            name=wdef.name,
                            provider="",
                            model=sub_model,
                            system_prompt=wdef.system_prompt,
                            temperature=sub_temp,
                        ),
                        provider=prov,
                    )
                    try:
                        result = await sub_agent.run(task)
                    except Exception as exc:  # noqa: BLE001
                        return json.dumps(
                            {"error": f"Worker {worker!r} failed: {exc}"},
                            ensure_ascii=False,
                        )
                    return json.dumps(
                        {"worker": wdef.name, "result": result.final_output},
                        ensure_ascii=False,
                    )

                _workers_desc_lines = [
                    f"- {w.name}: {w.description}" for w in _resolved_workers
                ]
                _workers_desc = "\n".join(_workers_desc_lines)

                tool_registry.register(
                    ToolDefinition(
                        name="delegate",
                        description=(
                            "Delegate a self-contained subtask to a specialised "
                            "subagent and return its answer. Use this when the "
                            "subtask clearly matches one of the available workers. "
                            "Do NOT delegate trivial questions you can answer directly.\n\n"
                            f"Available workers:\n{_workers_desc}"
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "worker": {
                                    "type": "string",
                                    "enum": _worker_names_list,
                                    "description": (
                                        "Name of the subagent to delegate to. "
                                        f"One of: {_worker_names_list}"
                                    ),
                                },
                                "task": {
                                    "type": "string",
                                    "description": (
                                        "The complete, self-contained subtask "
                                        "for the worker. Include all the context "
                                        "the worker needs — it cannot see the "
                                        "main conversation."
                                    ),
                                },
                            },
                            "required": ["worker", "task"],
                        },
                        handler=delegate_handler,
                    )
                )

                base_system += (
                    "\n\n## Subagent delegation\n\n"
                    "You have a `delegate` tool that routes a subtask to a "
                    "specialised subagent. Available subagents:\n\n"
                    f"{_workers_desc}\n\n"
                    "When the user's request maps to one of these specialities, "
                    "call `delegate` with a self-contained `task` — the subagent "
                    "cannot see the main conversation, so include the necessary "
                    "context. Summarise the worker's answer in your final reply; "
                    "do not paste raw JSON. Only delegate when the speciality "
                    "clearly fits — answer directly otherwise."
                )

                _logger.info(
                    "Delegate tool registered with workers: %s",
                    _worker_names_list,
                )

        agent_cfg = AgentConfig(
            name="eyetor",
            provider=provider or "fallback",
            model=model or prov.model,
            system_prompt=base_system,
            temperature=prov.temperature,
        )

        # Scheduler (shared across all channels). Only the non-interactive
        # "server" instance owns APScheduler — interactive CLIs would
        # duplicate every job otherwise.
        scheduler = None
        sched_cfg = cfg.scheduler
        if sched_cfg.enabled and not interactive:
            from eyetor.scheduler.store import SchedulerStore
            from eyetor.scheduler.channel import SchedulerChannel

            sched_store = SchedulerStore(sched_cfg.db_path)
            tg_token = (
                cfg.channels.telegram.bot_token
                if cfg.channels.telegram.enabled
                else None
            )
            # Scheduler needs a SessionManager; create a dedicated one (no scheduler to avoid recursion)
            sched_session_mgr = SessionManager(
                agent_cfg,
                prov,
                tool_registry=tool_registry,
                memory_manager=memory,
                knowledge=knowledge,
                root_config=cfg,
                tracker=tracker,
                cost_estimator=cost_estimator,
            )
            scheduler = SchedulerChannel(
                store=sched_store,
                session_manager=sched_session_mgr,
                bot_token=tg_token,
                default_timezone=sched_cfg.default_timezone,
            )

        # Build channels
        channels = []

        if interactive:
            from eyetor.channels.cli_channel import CliChannel

            session_mgr_cli = SessionManager(
                agent_cfg,
                prov,
                tool_registry=tool_registry,
                memory_manager=memory,
                scheduler=scheduler,
                knowledge=knowledge,
                root_config=cfg,
                tracker=tracker,
                cost_estimator=cost_estimator,
            )
            channels.append(
                CliChannel(
                    session_mgr_cli,
                    session_id=resolved_session_id,
                    skill_reg=skill_reg,
                    agent_reg=agent_reg,
                )
            )

        tg_cfg = cfg.channels.telegram
        if tg_cfg.enabled and tg_cfg.bot_token and not interactive:
            from eyetor.channels.telegram import TelegramChannel

            session_mgr_tg = SessionManager(
                agent_cfg,
                prov,
                tool_registry=tool_registry,
                memory_manager=memory,
                scheduler=scheduler,
                knowledge=knowledge,
                root_config=cfg,
                tracker=tracker,
                cost_estimator=cost_estimator,
            )
            channels.append(
                TelegramChannel(
                    session_mgr_tg,
                    tg_cfg,
                    skill_reg=skill_reg,
                    scheduler=scheduler,
                    tracker=tracker,
                    full_config=cfg,
                    agent_reg=agent_reg,
                )
            )

        if scheduler:
            channels.append(scheduler)

        if not channels or (len(channels) == 1 and scheduler in channels):
            console.print(
                "[red]No channels available. Run interactively or configure Telegram.[/red]"
            )
            return

        # Kick off knowledge-base reindex in the background (fire-and-forget).
        kb_reindex_task = None
        if knowledge is not None and cfg.knowledge.auto_reindex_on_start:
            log = logging.getLogger("eyetor.knowledge")

            async def _kb_reindex_bg():
                try:
                    reports = await knowledge.index_all()
                except Exception as exc:
                    log.warning("kb auto-reindex failed: %s", exc)
                    return
                for name, r in reports.items():
                    log.info(
                        "kb auto-reindex [%s]: scanned=%d indexed=%d updated=%d skipped=%d pruned=%d errors=%d chunks=%d in %.2fs",
                        name,
                        r.scanned,
                        r.indexed,
                        r.updated,
                        r.skipped,
                        r.pruned,
                        r.errors,
                        r.chunks_written,
                        r.duration_s,
                    )

            kb_reindex_task = asyncio.create_task(_kb_reindex_bg())

        try:
            if interactive:
                # In interactive mode the CLI channel is the "primary" — when
                # it finishes (user typed /exit or EOF) we must tear down all
                # background channels (Telegram, scheduler) so the process
                # exits cleanly.
                cli_channel = next(
                    ch for ch in channels
                    if type(ch).__name__ == "CliChannel"
                )
                background = [ch for ch in channels if ch is not cli_channel]
                bg_tasks = [asyncio.create_task(ch.start()) for ch in background]
                try:
                    await cli_channel.start()
                finally:
                    for t in bg_tasks:
                        t.cancel()
                    await asyncio.gather(*bg_tasks, return_exceptions=True)
                    for ch in background:
                        await ch.stop()
            else:
                # Register signal handlers so systemd SIGTERM triggers
                # graceful shutdown instead of hanging until SIGABRT.
                loop = asyncio.get_running_loop()
                stop_event = asyncio.Event()

                def _signal_handler():
                    if not stop_event.is_set():
                        stop_event.set()

                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, _signal_handler)

                tasks = [asyncio.create_task(ch.start()) for ch in channels]
                await stop_event.wait()
                for ch in channels:
                    await ch.stop()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if kb_reindex_task and not kb_reindex_task.done():
                kb_reindex_task.cancel()
                try:
                    await kb_reindex_task
                except (asyncio.CancelledError, Exception):
                    pass
            if mcp_registry:
                await mcp_registry.close_all()
            if plugin_registry:
                await plugin_registry.run_shutdown()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Helpers for host-tools safety
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    (
        "shell",
        "run.py",
        [
            "rm -rf",
            "rmdir /s",
            "format",
            "del /f",
            "dd if=",
            "mkfs",
            "drop table",
            "drop database",
            "> /dev/",
            "shutdown",
            "reboot",
        ],
    ),
    ("filesystem", "fs.py", ["delete --recursive", "delete"]),
]


def _is_dangerous(skill: str, script: str, args: str) -> bool:
    args_lower = args.lower()
    for d_skill, d_script, patterns in _DANGEROUS_PATTERNS:
        if skill == d_skill and script == d_script:
            for p in patterns:
                if p in args_lower:
                    return True
    return False


async def _ask_confirm(skill: str, script: str, args: str) -> bool:
    console.print(
        f"\n[bold red]⚠  Dangerous operation requested:[/bold red]\n"
        f"  Skill: [cyan]{skill}[/cyan]  Script: [cyan]{script}[/cyan]\n"
        f"  Args:  [yellow]{args}[/yellow]\n"
    )
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, lambda: input("  Confirm? [y/N] ").strip().lower()
    )
    return answer in {"y", "yes"}


def _split_args(args: str) -> list[str]:
    import shlex

    try:
        return shlex.split(args)
    except ValueError:
        return args.split()


# ---------------------------------------------------------------------------
# eyetor run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_text")
@click.option("--provider", "-p", default=None)
@click.option("--model", "-m", default=None)
@click.option("--system", "-s", default=None)
@click.option("--stream", is_flag=True, default=False, help="Stream the response.")
@click.pass_context
def run(
    ctx: click.Context,
    input_text: str,
    provider: str | None,
    model: str | None,
    system: str | None,
    stream: bool,
) -> None:
    """Run a single agent call (one-shot)."""
    cfg = ctx.obj["cfg"]

    async def _run():
        from eyetor.agents.base import BaseAgent
        from eyetor.models.agents import AgentConfig
        from eyetor.tracking.context import current_channel
        from eyetor.tracking.usage import UsageTracker
        from eyetor.tracking.pricing import CostEstimator

        current_channel.set("cli")
        tracker = UsageTracker.from_config(cfg.tracking)
        cost_estimator = CostEstimator()
        prov = _resolve_provider(cfg, provider, model, tracker, cost_estimator)

        agent = BaseAgent(
            config=AgentConfig(
                name="one-shot",
                provider=provider or "fallback",
                model=prov.model,
                system_prompt=system or "You are a helpful assistant.",
                temperature=prov.temperature,
            ),
            provider=prov,
        )
        if stream:
            # BaseAgent.stream() is a coroutine returning a StreamingResponse;
            # it must be awaited before async-iterating it.
            response = await agent.stream(input_text)
            async for token in response:
                print(token, end="", flush=True)
            print()
        else:
            result = await agent.run(input_text)
            console.print(result.final_output)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# eyetor skills
# ---------------------------------------------------------------------------


@cli.group()
def skills() -> None:
    """Manage and inspect available skills."""


@skills.command("list")
@click.pass_context
def skills_list(ctx: click.Context) -> None:
    """List all discovered skills."""
    cfg = ctx.obj["cfg"]
    from eyetor.skills.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover(cfg.skills_dirs)
    names = reg.list_names()
    if not names:
        console.print("[yellow]No skills found.[/yellow]")
        return
    table = Table(title="Available Skills")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Author")
    table.add_column("Version")
    for name in names:
        m = reg.get_metadata(name)
        table.add_row(name, m.description[:80], m.author or "-", m.version or "-")
    console.print(table)


@skills.command("info")
@click.argument("name")
@click.pass_context
def skills_info(ctx: click.Context, name: str) -> None:
    """Show full info for a skill."""
    cfg = ctx.obj["cfg"]
    from eyetor.skills.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover(cfg.skills_dirs)
    try:
        info = reg.activate(name)
    except KeyError:
        console.print(f"[red]Skill '{name}' not found.[/red]")
        sys.exit(1)
    console.print(f"[bold]Skill: {name}[/bold]")
    console.print(f"Description: {info.metadata.description}")
    console.print(f"License: {info.metadata.license or 'N/A'}")
    console.print(f"Path: {info.metadata.path}")
    if info.scripts:
        console.print(f"Scripts: {', '.join(s.name for s in info.scripts)}")
    console.print("\n[bold]Instructions:[/bold]")
    from rich.markdown import Markdown

    console.print(Markdown(info.instructions))


# ---------------------------------------------------------------------------
# eyetor providers
# ---------------------------------------------------------------------------


@cli.group()
def providers() -> None:
    """Manage and inspect LLM providers."""


@providers.command("list")
@click.pass_context
def providers_list(ctx: click.Context) -> None:
    """List all configured providers."""
    cfg = ctx.obj["cfg"]
    chain = cfg.fallback.fallback_chain
    chain_index = {name: i + 1 for i, name in enumerate(chain)}
    table = Table(title="Configured Providers")
    table.add_column("Name", style="bold cyan")
    table.add_column("Type")
    table.add_column("Model")
    table.add_column("Base URL")
    table.add_column("Chain", style="green")
    for name, p in cfg.providers.items():
        position = str(chain_index[name]) if name in chain_index else ""
        table.add_row(name, p.type, p.model, p.base_url, position)
    console.print(table)


@providers.command("test")
@click.argument("name", required=False)
@click.pass_context
def providers_test(ctx: click.Context, name: str | None) -> None:
    """Health-check a provider by sending a simple message."""
    cfg = ctx.obj["cfg"]

    async def _test():
        from eyetor.providers import get_provider
        from eyetor.models.messages import Message

        target = name
        if target is None:
            chain = cfg.fallback.fallback_chain
            if not chain:
                console.print(
                    "[red]No provider name given and fallback.fallback_chain is empty.[/red]"
                )
                sys.exit(1)
            target = chain[0]
        try:
            prov = get_provider(cfg, target)
        except KeyError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        console.print(f"Testing provider [bold]{target}[/bold] ({prov.model})...")
        try:
            result = await prov.complete(
                messages=[Message(role="user", content="Reply with exactly: OK")],
            )
            console.print(f"[green]✓ Response:[/green] {result.message.content}")
        except Exception as exc:
            console.print(f"[red]✗ Failed: {exc}[/red]")
            sys.exit(1)

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# eyetor mcp
# ---------------------------------------------------------------------------


@cli.group()
def mcp() -> None:
    """Manage MCP (Model Context Protocol) servers."""


@mcp.command("list")
@click.pass_context
def mcp_list(ctx: click.Context) -> None:
    """List configured MCP servers and their connection status."""
    cfg = ctx.obj["cfg"]
    if not cfg.mcp_servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return

    async def _run():
        from eyetor.mcp.registry import McpRegistry

        registry = McpRegistry(cfg.mcp_servers)
        await registry.connect_all()
        report = registry.get_degraded_report()
        table = Table(title="MCP Servers")
        table.add_column("Name", style="bold cyan")
        table.add_column("Transport")
        table.add_column("Status")
        table.add_column("Tools", justify="right")
        table.add_column("Endpoint")
        for name, srv in cfg.mcp_servers.items():
            endpoint = srv.url or f"{srv.command} {' '.join(srv.args)}"
            if name in report.failed:
                status = f"[red]✗ Offline[/red]"
                tools_count = "—"
            else:
                status = f"[green]✓ Online[/green]"
                tools_count = str(len(registry.get_tools(name)))
            table.add_row(name, srv.transport, status, tools_count, endpoint)
        console.print(table)
        await registry.close_all()

    asyncio.run(_run())


@mcp.command("tools")
@click.argument("server_name")
@click.pass_context
def mcp_tools(ctx: click.Context, server_name: str) -> None:
    """List tools available from an MCP server."""
    cfg = ctx.obj["cfg"]

    async def _run():
        from eyetor.mcp.registry import McpRegistry

        if server_name not in cfg.mcp_servers:
            console.print(f"[red]MCP server '{server_name}' not found.[/red]")
            sys.exit(1)
        registry = McpRegistry({server_name: cfg.mcp_servers[server_name]})
        await registry.connect_all()
        tools = registry.get_tools(server_name)
        if not tools:
            console.print("[yellow]No tools found.[/yellow]")
        else:
            table = Table(title=f"Tools from {server_name}")
            table.add_column("Name", style="cyan")
            table.add_column("Description")
            for t in tools:
                table.add_row(t.name, t.description[:80])
            console.print(table)
        await registry.close_all()

    asyncio.run(_run())


@mcp.command("test")
@click.argument("server_name")
@click.pass_context
def mcp_test(ctx: click.Context, server_name: str) -> None:
    """Test connection to an MCP server."""
    cfg = ctx.obj["cfg"]

    async def _run():
        from eyetor.mcp.registry import McpRegistry

        if server_name not in cfg.mcp_servers:
            console.print(f"[red]MCP server '{server_name}' not found.[/red]")
            sys.exit(1)
        console.print(f"Connecting to MCP server [bold]{server_name}[/bold]...")
        registry = McpRegistry({server_name: cfg.mcp_servers[server_name]})
        await registry.connect_all()
        if registry.is_connected(server_name):
            tools = registry.get_tools(server_name)
            console.print(f"[green]✓ Connected. {len(tools)} tools available.[/green]")
        else:
            console.print("[red]✗ Failed to connect.[/red]")
            sys.exit(1)
        await registry.close_all()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# eyetor usage
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--period", default="day", type=click.Choice(["day", "week", "month"]))
@click.option("--provider", default=None)
@click.option(
    "--detail",
    is_flag=True,
    default=False,
    help="Show individual calls instead of summary.",
)
@click.option(
    "--limit", "-n", default=20, help="Number of recent calls (with --detail)."
)
@click.option("--agent", default=None, help="Filter by agent name.")
@click.option(
    "--phase",
    default=None,
    type=click.Choice(
        [
            "main",
            "loop_break",
            "chain_plan",
            "chain_execute",
            "chain_synthesize",
            "compaction",
            "routing",
            "agent",
        ]
    ),
    help="Filter by call phase.",
)
@click.option(
    "--by",
    default="provider",
    type=click.Choice(["provider", "agent", "phase"]),
    help="Group the summary by this dimension.",
)
@click.pass_context
def usage(
    ctx: click.Context,
    period: str,
    provider: str | None,
    detail: bool,
    limit: int,
    agent: str | None,
    phase: str | None,
    by: str,
) -> None:
    """Show LLM usage and cost statistics."""
    cfg = ctx.obj["cfg"]
    from eyetor.tracking.usage import UsageTracker

    tracker = UsageTracker.from_config(cfg.tracking)

    if detail:
        records = tracker.get_recent(limit=limit, provider=provider)
        if agent is not None:
            records = [r for r in records if r.agent == agent]
        if phase is not None:
            records = [r for r in records if r.phase == phase]
        if not records:
            console.print("[yellow]No usage records found.[/yellow]")
            return
        table = Table(title="Recent LLM Calls")
        table.add_column("Timestamp", style="dim")
        table.add_column("Provider", style="cyan")
        table.add_column("Model")
        table.add_column("Agent")
        table.add_column("Phase")
        table.add_column("Chan", style="dim")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Speed", justify="right")
        table.add_column("Tools", justify="right")
        table.add_column("Msgs", justify="right")
        table.add_column("Trace", style="dim")
        for r in records:
            ts = r.timestamp[:19].replace("T", " ")
            tokens = f"{r.prompt_tokens} → {r.completion_tokens}"
            speed = f"{r.speed_tps:.1f} tps" if r.speed_tps else "—"
            cost = f"{r.estimated_cost:.4f}" if r.estimated_cost else "0"
            table.add_row(
                ts,
                r.provider,
                r.model,
                r.agent or "—",
                r.phase or "—",
                r.channel or "—",
                tokens,
                cost,
                speed,
                str(r.tool_count),
                str(r.msg_count),
                (r.trace_id or "")[:8] or "—",
            )
        console.print(table)
        return

    summaries = tracker.get_summary(
        period=period,
        provider=provider,
        agent=agent,
        phase=phase,
        group_by_agent=(by == "agent"),
        group_by_phase=(by == "phase"),
    )
    if not summaries:
        console.print(f"[yellow]No usage data for period '{period}'.[/yellow]")
        return
    table = Table(title=f"Usage ({period}, by {by})")
    table.add_column("Provider", style="cyan")
    table.add_column("Model")
    if by == "agent":
        table.add_column("Agent")
    if by == "phase":
        table.add_column("Phase")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Est. Cost ($)", justify="right")
    for s in summaries:
        row = [s.provider, s.model]
        if by == "agent":
            row.append(s.agent or "—")
        if by == "phase":
            row.append(s.phase or "—")
        row += [
            str(s.calls),
            str(s.prompt_tokens),
            str(s.completion_tokens),
            str(s.total_tokens),
            f"{s.estimated_cost:.4f}",
        ]
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# eyetor kb
# ---------------------------------------------------------------------------


@cli.group()
def kb() -> None:
    """Manage the workspace knowledge base (hybrid BM25 + semantic retrieval)."""


def _require_kb(ctx: click.Context):
    cfg = ctx.obj["cfg"]
    if not cfg.knowledge or not cfg.knowledge.enabled:
        console.print(
            "[yellow]Knowledge base disabled. Set knowledge.enabled=true in config.[/yellow]"
        )
        sys.exit(1)
    if not cfg.knowledge.workspaces:
        console.print(
            "[yellow]No knowledge workspaces configured. Add entries under knowledge.workspaces.[/yellow]"
        )
        sys.exit(1)
    return cfg


@kb.command("index")
@click.option("--workspace", "-w", default=None, help="Workspace name (default: all).")
@click.option("--force", is_flag=True, default=False, help="Ignore sha1 skip and reindex all files.")
@click.option("--prune/--no-prune", default=True, help="Delete docs that no longer match globs.")
@click.pass_context
def kb_index(ctx: click.Context, workspace: str | None, force: bool, prune: bool) -> None:
    """Index (or reindex) one or all workspaces."""
    cfg = _require_kb(ctx)
    from eyetor.knowledge.manager import KnowledgeManager

    async def _run():
        km = KnowledgeManager.from_config(cfg.knowledge)
        if workspace:
            report = await km.index_workspace(workspace, force=force, prune=prune)
            reports = {workspace: report}
        else:
            reports = await km.index_all(force=force, prune=prune)
        table = Table(title="Indexing results")
        table.add_column("Workspace", style="bold cyan")
        table.add_column("Scanned", justify="right")
        table.add_column("Indexed", justify="right", style="green")
        table.add_column("Updated", justify="right", style="yellow")
        table.add_column("Skipped", justify="right", style="dim")
        table.add_column("Pruned", justify="right")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Chunks", justify="right")
        table.add_column("Time (s)", justify="right")
        for name, r in reports.items():
            table.add_row(
                name,
                str(r.scanned),
                str(r.indexed),
                str(r.updated),
                str(r.skipped),
                str(r.pruned),
                str(r.errors),
                str(r.chunks_written),
                f"{r.duration_s:.2f}",
            )
        console.print(table)

    asyncio.run(_run())


@kb.command("search")
@click.argument("query")
@click.option("--workspace", "-w", default=None, help="Filter by workspace.")
@click.option("--top-k", "-k", default=5, help="Number of results.")
@click.option(
    "--bench",
    default=0,
    type=int,
    help="Run the query N times and report p50/p95/p99 latency (skips result table).",
)
@click.pass_context
def kb_search(
    ctx: click.Context, query: str, workspace: str | None, top_k: int, bench: int
) -> None:
    """Run a hybrid retrieval query against the knowledge base."""
    cfg = _require_kb(ctx)
    from eyetor.knowledge.manager import KnowledgeManager

    async def _run():
        km = KnowledgeManager.from_config(cfg.knowledge)
        if bench > 0:
            import time

            # Warm-up so the first call's import/open cost doesn't skew stats.
            await km.search(query, workspace=workspace, top_k=top_k)
            samples: list[float] = []
            for _ in range(bench):
                t0 = time.perf_counter()
                await km.search(query, workspace=workspace, top_k=top_k)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            n = len(samples)
            p50 = samples[int(n * 0.50)]
            p95 = samples[min(int(n * 0.95), n - 1)]
            p99 = samples[min(int(n * 0.99), n - 1)]
            table = Table(title=f"kb_search benchmark ({n} runs)")
            table.add_column("Metric", style="bold cyan")
            table.add_column("ms", justify="right")
            table.add_row("min", f"{samples[0]:.2f}")
            table.add_row("p50", f"{p50:.2f}")
            table.add_row("p95", f"{p95:.2f}")
            table.add_row("p99", f"{p99:.2f}")
            table.add_row("max", f"{samples[-1]:.2f}")
            table.add_row("mean", f"{sum(samples) / n:.2f}")
            console.print(table)
            return
        hits = await km.search(query, workspace=workspace, top_k=top_k)
        if not hits:
            console.print(f"[yellow]No results for '{query}'.[/yellow]")
            return
        table = Table(title=f"Search: {query}")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Workspace", style="cyan")
        table.add_column("Path")
        table.add_column("Heading", style="dim")
        table.add_column("Sources", style="magenta")
        table.add_column("Snippet")
        for i, h in enumerate(hits, start=1):
            table.add_row(
                str(i),
                h.workspace,
                h.path,
                (h.heading or "")[:40],
                "+".join(h.sources) or "-",
                h.snippet[:120].replace("\n", " "),
            )
        console.print(table)
        console.print(
            "[dim]Use 'eyetor kb read DOC_ID' to read a full document (doc_ids: "
            + ", ".join(str(h.doc_id) for h in hits)
            + ")[/dim]"
        )

    asyncio.run(_run())


@kb.command("read")
@click.argument("doc_id", type=int)
@click.option("--section", "-s", default=None, help="Section heading prefix filter.")
@click.option("--max-chars", default=1800, help="Maximum characters to return.")
@click.pass_context
def kb_read(ctx: click.Context, doc_id: int, section: str | None, max_chars: int) -> None:
    """Read a document (or a section) from the knowledge base."""
    cfg = _require_kb(ctx)
    from eyetor.knowledge.manager import KnowledgeManager

    km = KnowledgeManager.from_config(cfg.knowledge)
    result = km.read_doc(doc_id, section=section, max_chars=max_chars)
    if not result:
        console.print(f"[red]Document {doc_id} not found.[/red]")
        sys.exit(1)
    if not result.section_matched:
        console.print(
            f"[yellow]Section '{section}' not found in document {doc_id}.[/yellow]"
        )
        if result.available_sections:
            console.print("[dim]Available sections:[/dim]")
            for s in result.available_sections:
                console.print(f"  - {s}")
        sys.exit(1)
    console.print(f"[bold]{result.title or result.path}[/bold] ({result.path})")
    if result.section:
        console.print(f"[dim]Section: {result.section}[/dim]")
    console.print()
    console.print(result.content)
    if result.truncated:
        console.print(
            f"\n[yellow]…truncated (total {result.total_chars} chars)[/yellow]"
        )


@kb.command("list")
@click.option("--workspace", "-w", default=None, help="Filter by workspace.")
@click.option("--limit", default=50, help="Maximum rows to show.")
@click.pass_context
def kb_list(ctx: click.Context, workspace: str | None, limit: int) -> None:
    """List indexed documents in the knowledge base."""
    cfg = _require_kb(ctx)
    from eyetor.knowledge.manager import KnowledgeManager

    km = KnowledgeManager.from_config(cfg.knowledge)
    sources = km.list_sources(workspace=workspace, limit=limit)
    if not sources["docs"]:
        console.print("[yellow]No documents indexed.[/yellow]")
        return
    table = Table(title="Indexed documents")
    table.add_column("doc_id", justify="right", style="dim")
    table.add_column("Workspace", style="cyan")
    table.add_column("Path")
    table.add_column("Title")
    for d in sources["docs"]:
        table.add_row(
            str(d["doc_id"]),
            d["workspace"],
            d["path"],
            (d["title"] or "")[:50],
        )
    console.print(table)
    console.print(f"[dim]Workspaces: {', '.join(sources['workspaces'])}[/dim]")


@kb.command("status")
@click.pass_context
def kb_status(ctx: click.Context) -> None:
    """Show knowledge base statistics."""
    cfg = _require_kb(ctx)
    from eyetor.knowledge.manager import KnowledgeManager

    km = KnowledgeManager.from_config(cfg.knowledge)
    stats = km.stats()
    table = Table(title="Knowledge base status")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[dim]Workspaces: {', '.join(km.list_workspaces()) or '-'}[/dim]")


# ---------------------------------------------------------------------------
# eyetor sessions
# ---------------------------------------------------------------------------


@cli.group()
def sessions() -> None:
    """Manage persistent CLI/agent sessions (JSONL on disk)."""


def _sessions_dir(cfg) -> Path:
    return Path(cfg.sessions.dir).expanduser()


def _iter_session_files(cfg) -> list[Path]:
    d = _sessions_dir(cfg)
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"))


def _ppid_alive(ppid: int) -> bool:
    """True if /proc/<ppid> still exists (Linux); falls back to kill(0) elsewhere."""
    proc = Path(f"/proc/{ppid}")
    if proc.exists():
        return True
    if proc.parent.exists():  # /proc exists, just no entry — dead
        return False
    try:
        os.kill(ppid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


@sessions.command("list")
@click.pass_context
def sessions_list(ctx: click.Context) -> None:
    """List persisted sessions in the configured sessions directory."""
    cfg = ctx.obj["cfg"]
    files = _iter_session_files(cfg)
    if not files:
        console.print(
            f"[yellow]No sessions in {_sessions_dir(cfg)}.[/yellow]"
        )
        return
    current = _default_cli_session_id()
    table = Table(title=f"Sessions in {_sessions_dir(cfg)}")
    table.add_column("", width=1)
    table.add_column("Name", style="bold cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")
    for f in files:
        name = f.stem
        marker = "*" if name == current else " "
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if size >= 1024 * 1024:
            size_s = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            size_s = f"{size / 1024:.1f} KB"
        else:
            size_s = f"{size} B"
        table.add_row(marker, name, size_s, mtime)
    console.print(table)
    console.print(
        "[dim]'*' = sesión que este proceso usaría por defecto "
        f"({current}).[/dim]"
    )


@sessions.command("rm")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
@click.pass_context
def sessions_rm(ctx: click.Context, name: str, yes: bool) -> None:
    """Delete a persisted session JSONL by name (no extension)."""
    cfg = ctx.obj["cfg"]
    path = _sessions_dir(cfg) / f"{name}.jsonl"
    if not path.exists():
        console.print(f"[red]Session '{name}' not found ({path}).[/red]")
        sys.exit(1)
    if not yes:
        if not click.confirm(f"Delete {path}?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return
    path.unlink()
    console.print(f"[green]Deleted {path}.[/green]")


@sessions.command("prune")
@click.option("--dry-run", is_flag=True, default=False, help="Only list, do not delete.")
@click.pass_context
def sessions_prune(ctx: click.Context, dry_run: bool) -> None:
    """Delete `cli-<user>-<ppid>` sessions whose ppid no longer exists."""
    cfg = ctx.obj["cfg"]
    user = getpass.getuser()
    prefix = f"cli-{user}-"
    candidates: list[tuple[Path, int]] = []
    for f in _iter_session_files(cfg):
        name = f.stem
        if not name.startswith(prefix):
            continue
        ppid_part = name[len(prefix):]
        if not ppid_part.isdigit():
            continue
        ppid = int(ppid_part)
        if not _ppid_alive(ppid):
            candidates.append((f, ppid))
    if not candidates:
        console.print("[dim]No orphaned sessions found.[/dim]")
        return
    table = Table(title="Orphaned sessions" + (" (dry-run)" if dry_run else ""))
    table.add_column("File")
    table.add_column("ppid", justify="right")
    for path, ppid in candidates:
        table.add_row(str(path), str(ppid))
    console.print(table)
    if dry_run:
        return
    for path, _ in candidates:
        path.unlink(missing_ok=True)
    console.print(f"[green]Deleted {len(candidates)} session(s).[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main_sync() -> None:
    """Synchronous entry point for the 'eyetor' command."""
    cli(obj={})
