"""Eyetor CLI — command-line interface for the multi-agent system."""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
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
    """Default CLI chat identity: stable per user.

    Each ``eyetor start`` opens a NEW conversation within this chat; past
    conversations are listed/resumed with /chats and /chat N in the REPL.
    """
    return f"cli-{getpass.getuser()}"


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
# eyetor setup
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--refresh-host",
    is_flag=True,
    default=False,
    help="Regenerate the persisted host profile.",
)
@click.option(
    "--print-json",
    "print_json",
    is_flag=True,
    default=False,
    help="Print the host profile as JSON.",
)
@click.option(
    "--install-helper",
    is_flag=True,
    default=False,
    help="Install the restricted privileged package-install helper.",
)
@click.option(
    "--service-user",
    default=None,
    help="User account that runs the Eyetor service (default: current user or SUDO_USER).",
)
def setup(
    refresh_host: bool,
    print_json: bool,
    install_helper: bool,
    service_user: str | None,
) -> None:
    """Run first-time setup tasks for this machine."""
    from eyetor.host_info import (
        detect_host_profile,
        ensure_host_profile,
        host_profile_path,
        read_host_profile,
    )

    if install_helper:
        from eyetor.install_helper import install_privileged_helper

        service_user = _resolve_setup_service_user(service_user)
        target = _host_profile_path_for_user(service_user)
        if os.geteuid() != 0:
            import shlex

            exe = shlex.quote(str(Path(sys.argv[0]).resolve()))
            user_arg = shlex.quote(service_user)
            console.print(
                "[red]Installing the privileged helper requires root.[/red]\n"
                "Run this one-time bootstrap command:\n"
                f"  sudo {exe} setup --install-helper --service-user {user_arg}"
            )
            sys.exit(1)
        profile = detect_host_profile() if refresh_host else (read_host_profile(target) or detect_host_profile())
        profile = install_privileged_helper(
            service_user=service_user,
            host_profile=profile,
            host_path=target,
        )
        existed = True
    else:
        target = host_profile_path()
        existed = target.exists()
        profile = ensure_host_profile(refresh=refresh_host)

    if print_json:
        console.print_json(data=profile)
        return

    if install_helper:
        status = "Installed package helper and updated"
    elif refresh_host:
        status = "Regenerated"
    elif existed:
        status = "Existing"
    else:
        status = "Created"

    console.print(f"[green]{status} host profile:[/green] {target}")
    table = Table(title="Host Profile")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("OS", str(profile.get("os_name") or "-"))
    table.add_row("OS ID", str(profile.get("os_id") or "-"))
    table.add_row("OS Like", ", ".join(profile.get("os_like") or []) or "-")
    table.add_row("Platform", str(profile.get("platform") or "-"))
    table.add_row("Machine", str(profile.get("machine") or "-"))
    table.add_row(
        "Package managers",
        ", ".join(profile.get("package_managers") or []) or "-",
    )
    table.add_row(
        "Preferred manager",
        str(profile.get("preferred_package_manager") or "-"),
    )
    table.add_row(
        "Autonomous installs",
        "yes" if profile.get("can_install_system_packages") else "no",
    )
    if profile.get("install_helper"):
        table.add_row("Install helper", str(profile.get("install_helper")))
        table.add_row("Install strategy", str(profile.get("install_strategy") or "auto"))
    console.print(table)


def _resolve_setup_service_user(service_user: str | None) -> str:
    from eyetor.install_helper import validate_service_user

    resolved = service_user or os.environ.get("SUDO_USER") or getpass.getuser()
    if resolved == "root" and not service_user:
        raise click.UsageError("Pass --service-user with the account that runs Eyetor.")
    return validate_service_user(resolved)


def _host_profile_path_for_user(service_user: str) -> Path:
    import pwd

    info = pwd.getpwnam(service_user)
    return Path(info.pw_dir) / ".eyetor" / "host.json"


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
        "CLI chat identity override. Each start opens a new conversation; "
        "resume past ones with /chats and /chat N. "
        "Precedence: --session > $EYETOR_SESSION > cli-<user>."
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
        from eyetor.app import build_app, build_channels, serve
        from eyetor.tracking.pricing import CostEstimator
        from eyetor.tracking.usage import UsageTracker

        tracker = UsageTracker.from_config(cfg.tracking)
        cost_estimator = CostEstimator()
        prov = _resolve_provider(cfg, provider, model, tracker, cost_estimator)
        app = await build_app(
            cfg,
            prov=prov,
            tracker=tracker,
            cost_estimator=cost_estimator,
            use_host_tools=use_host_tools,
            interactive=interactive,
            agent_provider=provider or "fallback",
            agent_model=model or prov.model,
        )
        channels, scheduler = build_channels(
            app, interactive=interactive, cli_session_id=resolved_session_id
        )
        if not channels or (len(channels) == 1 and scheduler in channels):
            console.print(
                "[red]No channels available. Run interactively or configure Telegram.[/red]"
            )
            return
        await serve(app, channels, interactive=interactive)

    asyncio.run(_run())


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
        # Conversation files are named <chat_id>.<conv_id>.jsonl — match both.
        marker = "*" if name == current or name.startswith(current + ".") else " "
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
        "[dim]'*' = conversaciones del chat CLI por defecto "
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
    """Delete LEGACY `cli-<user>-<ppid>` session files whose ppid is gone.

    Current CLI sessions use per-conversation files (`cli-<user>.<conv>.jsonl`)
    which this command never touches.
    """
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
