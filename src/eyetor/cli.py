"""Eyetor CLI — command-line interface for the multi-agent system."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
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
@click.pass_context
def start(
    ctx: click.Context, provider: str | None, model: str | None, host_tools: bool | None
) -> None:
    """Start the agent — launches all configured channels (CLI and/or Telegram)."""
    import sys

    cfg = ctx.obj["cfg"]

    # --host-tools flag overrides config; config default is True
    use_host_tools = (
        host_tools if host_tools is not None else cfg.channels.cli.host_tools
    )
    interactive = sys.stdin.isatty()

    if interactive:
        # Reconfigure logging: suppress INFO noise in interactive CLI mode
        _setup_logging(cfg.log_level, interactive=True)

    async def _run():
        from eyetor.providers import get_provider
        from eyetor.models.agents import AgentConfig
        from eyetor.chat.manager import SessionManager
        from eyetor.models.tools import ToolRegistry, ToolDefinition
        from eyetor.memory.manager import MemoryManager
        from eyetor.skills.registry import SkillRegistry
        from eyetor.skills.executor import run_script, DEFAULT_TIMEOUT
        from eyetor.tracking.usage import UsageTracker
        from eyetor.tracking.pricing import CostEstimator

        tracker = UsageTracker.from_config(cfg.tracking)
        cost_estimator = CostEstimator()
        prov = get_provider(
            cfg, provider, tracker=tracker, cost_estimator=cost_estimator
        )
        if model:
            prov.model = model
            if hasattr(prov, "_inner"):
                prov._inner.model = model

        memory = MemoryManager.from_path(cfg.memory_db_path)

        # Shared skill registry
        skill_reg = SkillRegistry()
        skill_reg.discover(cfg.skills_dirs)
        skill_names = skill_reg.list_names()

        # System prompt
        if use_host_tools:
            base_system = (
                "Eres Eyetor (suena como Aitor), un asistente de IA con acceso a herramientas que pueden actuar en el ordenador del usuario. "
                "Puedes ejecutar comandos de shell, gestionar ficheros, abrir URLs y buscar en internet. "
                "Responde siempre en español de España. "
                "Explica brevemente lo que vas a hacer antes de hacerlo. "
                "Pide confirmación antes de operaciones destructivas (borrar, sobreescribir, formatear)."
            )
        else:
            base_system = (
                "Eres Eyetor (suena como Aitor), un asistente de IA útil. "
                "Responde siempre en español de España."
            )

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

            async def run_skill_script_handler(
                skill: str, script: str, args: str = ""
            ) -> str:
                scripts = skill_reg.list_scripts(skill)
                script_path = next((s for s in scripts if s.name == script), None)
                if not script_path:
                    return json.dumps(
                        {"error": f"Script '{script}' not found in skill '{skill}'"}
                    )
                arg_list = _split_args(args)
                if use_host_tools and _is_dangerous(skill, script, args):
                    confirmed = await _ask_confirm(skill, script, args)
                    if not confirmed:
                        return json.dumps({"error": "Operation cancelled by user."})
                meta = skill_reg.get_metadata(skill)
                timeout = meta.timeout if meta.timeout is not None else DEFAULT_TIMEOUT
                return await run_script(script_path, arg_list, timeout=timeout)

            # Build dynamic skill list for tool description
            _skill_summaries = []
            for _sn in skill_names:
                _sm = skill_reg.get_metadata(_sn)
                if _sm:
                    _skill_summaries.append(f"{_sm.name} ({_sm.description[:80]})")
            _skills_desc = ", ".join(_skill_summaries) if _skill_summaries else "none"

            tool_registry.register(
                ToolDefinition(
                    name="run_skill_script",
                    description=(
                        "Execute a script from an available skill. "
                        f"Available skills: {_skills_desc}"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "skill": {
                                "type": "string",
                                "description": f"Skill name. One of: {', '.join(skill_names)}",
                            },
                            "script": {
                                "type": "string",
                                "description": "Script filename (e.g. run.py, fs.py, browser.py, search.py)",
                            },
                            "args": {
                                "type": "string",
                                "description": "CLI arguments as a single string (e.g. '--cmd \"ls -la\"')",
                            },
                        },
                        "required": ["skill", "script"],
                    },
                    handler=run_skill_script_handler,
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

        agent_cfg = AgentConfig(
            name="eyetor",
            provider=provider or cfg.default_provider,
            model=model or prov.model,
            system_prompt=base_system,
            temperature=prov.temperature,
        )

        # Scheduler (shared across all channels)
        scheduler = None
        sched_cfg = cfg.scheduler
        if sched_cfg.enabled:
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
                root_config=cfg,
                tracker=tracker,
                cost_estimator=cost_estimator,
            )
            channels.append(CliChannel(session_mgr_cli, skill_reg=skill_reg))

        tg_cfg = cfg.channels.telegram
        if tg_cfg.enabled and tg_cfg.bot_token and not interactive:
            from eyetor.channels.telegram import TelegramChannel

            session_mgr_tg = SessionManager(
                agent_cfg,
                prov,
                tool_registry=tool_registry,
                memory_manager=memory,
                scheduler=scheduler,
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
                )
            )

        if scheduler:
            channels.append(scheduler)

        if not channels or (len(channels) == 1 and scheduler in channels):
            console.print(
                "[red]No channels available. Run interactively or configure Telegram.[/red]"
            )
            return

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
                await asyncio.gather(*[ch.start() for ch in channels])
        finally:
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
        from eyetor.providers import get_provider
        from eyetor.agents.base import BaseAgent
        from eyetor.models.agents import AgentConfig
        from eyetor.tracking.usage import UsageTracker
        from eyetor.tracking.pricing import CostEstimator

        tracker = UsageTracker.from_config(cfg.tracking)
        cost_estimator = CostEstimator()
        prov = get_provider(
            cfg, provider, tracker=tracker, cost_estimator=cost_estimator
        )
        if model:
            prov.model = model
            if hasattr(prov, "_inner"):
                prov._inner.model = model

        agent = BaseAgent(
            config=AgentConfig(
                name="one-shot",
                provider=provider or cfg.default_provider,
                model=prov.model,
                system_prompt=system or "You are a helpful assistant.",
                temperature=prov.temperature,
            ),
            provider=prov,
        )
        if stream:
            async for token in agent.stream(input_text):
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
    table = Table(title="Configured Providers")
    table.add_column("Name", style="bold cyan")
    table.add_column("Type")
    table.add_column("Model")
    table.add_column("Base URL")
    table.add_column("Default", style="green")
    for name, p in cfg.providers.items():
        is_default = "*" if name == cfg.default_provider else ""
        table.add_row(name, p.type, p.model, p.base_url, is_default)
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

        target = name or cfg.default_provider
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
@click.pass_context
def usage(
    ctx: click.Context, period: str, provider: str | None, detail: bool, limit: int
) -> None:
    """Show LLM usage and cost statistics."""
    cfg = ctx.obj["cfg"]
    from eyetor.tracking.usage import UsageTracker

    tracker = UsageTracker.from_config(cfg.tracking)

    if detail:
        records = tracker.get_recent(limit=limit, provider=provider)
        if not records:
            console.print("[yellow]No usage records found.[/yellow]")
            return
        table = Table(title="Recent LLM Calls")
        table.add_column("Timestamp", style="dim")
        table.add_column("Provider", style="cyan")
        table.add_column("Model")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Speed", justify="right")
        table.add_column("Finish")
        table.add_column("Session", style="dim")
        for r in records:
            ts = r.timestamp[:19].replace("T", " ")
            tokens = f"{r.prompt_tokens} → {r.completion_tokens}"
            speed = f"{r.speed_tps:.1f} tps" if r.speed_tps else "—"
            cost = f"{r.estimated_cost:.4f}" if r.estimated_cost else "0"
            table.add_row(
                ts,
                r.provider,
                r.model,
                tokens,
                cost,
                speed,
                r.finish_reason or "—",
                r.session_id,
            )
        console.print(table)
        return

    summaries = tracker.get_summary(period=period, provider=provider)
    if not summaries:
        console.print(f"[yellow]No usage data for period '{period}'.[/yellow]")
        return
    table = Table(title=f"Usage ({period})")
    table.add_column("Provider", style="cyan")
    table.add_column("Model")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Est. Cost ($)", justify="right")
    for s in summaries:
        table.add_row(
            s.provider,
            s.model,
            str(s.calls),
            str(s.prompt_tokens),
            str(s.completion_tokens),
            str(s.total_tokens),
            f"{s.estimated_cost:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main_sync() -> None:
    """Synchronous entry point for the 'eyetor' command."""
    cli(obj={})
