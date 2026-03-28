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


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


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
@click.option("--provider", "-p", default=None, help="Provider name (default from config).")
@click.option("--model", "-m", default=None, help="Model override.")
@click.option(
    "--host-tools/--no-host-tools", default=None,
    help="Enable/disable host skills (shell, filesystem, browser). Default from config."
)
@click.pass_context
def start(ctx: click.Context, provider: str | None, model: str | None, host_tools: bool | None) -> None:
    """Start the agent — launches all configured channels (CLI and/or Telegram)."""
    import sys
    cfg = ctx.obj["cfg"]

    # --host-tools flag overrides config; config default is True
    use_host_tools = host_tools if host_tools is not None else cfg.channels.cli.host_tools
    interactive = sys.stdin.isatty()

    async def _run():
        from eyetor.providers import get_provider
        from eyetor.models.agents import AgentConfig
        from eyetor.chat.manager import SessionManager
        from eyetor.models.tools import ToolRegistry, ToolDefinition
        from eyetor.memory.manager import MemoryManager
        from eyetor.skills.registry import SkillRegistry
        from eyetor.skills.executor import run_script

        prov = get_provider(cfg, provider)
        if model:
            prov.model = model

        memory = MemoryManager.from_path(cfg.memory_db_path)

        # Shared skill registry
        skill_reg = SkillRegistry()
        skill_reg.discover(cfg.skills_dirs)
        skill_names = skill_reg.list_names()

        # System prompt
        if use_host_tools:
            base_system = (
                "You are Eyetor, a helpful AI assistant with access to tools that can act on the user's computer. "
                "You can run shell commands, manage files, open URLs, and search the web. "
                "Always explain what you are about to do before doing it. "
                "Ask for confirmation before destructive operations (delete, overwrite, format)."
            )
        else:
            base_system = "You are Eyetor, a helpful AI assistant."

        skills_context = skill_reg.build_skills_context(skill_names)
        if skills_context:
            base_system = f"{base_system}\n\n{skills_context}"

        # Shared tool registry
        tool_registry = ToolRegistry()
        if skill_names:
            async def run_skill_script_handler(skill: str, script: str, args: str = "") -> str:
                scripts = skill_reg.list_scripts(skill)
                script_path = next((s for s in scripts if s.name == script), None)
                if not script_path:
                    return json.dumps({"error": f"Script '{script}' not found in skill '{skill}'"})
                arg_list = _split_args(args)
                if use_host_tools and _is_dangerous(skill, script, args):
                    confirmed = await _ask_confirm(skill, script, args)
                    if not confirmed:
                        return json.dumps({"error": "Operation cancelled by user."})
                return await run_script(script_path, arg_list)

            tool_registry.register(ToolDefinition(
                name="run_skill_script",
                description=(
                    "Execute a script from an available skill. "
                    "Skills: shell (run commands), filesystem (read/write/list/search files), "
                    "browser (open URLs), web-search (search the web)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "description": "Skill name (shell, filesystem, browser, web-search)"},
                        "script": {"type": "string", "description": "Script filename (e.g. run.py, fs.py, browser.py, search.py)"},
                        "args": {"type": "string", "description": "CLI arguments as a single string (e.g. '--cmd \"ls -la\"')"},
                    },
                    "required": ["skill", "script"],
                },
                handler=run_skill_script_handler,
            ))

        agent_cfg = AgentConfig(
            name="eyetor",
            provider=provider or cfg.default_provider,
            model=model or prov.model,
            system_prompt=base_system,
            temperature=prov.temperature,
        )

        # Build channels
        channels = []

        if interactive:
            from eyetor.channels.cli_channel import CliChannel
            session_mgr_cli = SessionManager(agent_cfg, prov, tool_registry=tool_registry, memory_manager=memory)
            channels.append(CliChannel(session_mgr_cli, skill_reg=skill_reg))

        tg_cfg = cfg.channels.telegram
        if tg_cfg.enabled and tg_cfg.bot_token:
            from eyetor.channels.telegram import TelegramChannel
            session_mgr_tg = SessionManager(agent_cfg, prov, tool_registry=tool_registry, memory_manager=memory)
            channels.append(TelegramChannel(session_mgr_tg, tg_cfg, skill_reg=skill_reg))

        if not channels:
            console.print("[red]No channels available. Run interactively or configure Telegram.[/red]")
            return

        await asyncio.gather(*[ch.start() for ch in channels])

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Helpers for host-tools safety
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    ("shell", "run.py", ["rm -rf", "rmdir /s", "format", "del /f", "dd if=", "mkfs",
                          "drop table", "drop database", "> /dev/", "shutdown", "reboot"]),
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
        None,
        lambda: input("  Confirm? [y/N] ").strip().lower()
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
def run(ctx: click.Context, input_text: str, provider: str | None, model: str | None, system: str | None, stream: bool) -> None:
    """Run a single agent call (one-shot)."""
    cfg = ctx.obj["cfg"]

    async def _run():
        from eyetor.providers import get_provider
        from eyetor.agents.base import BaseAgent
        from eyetor.models.agents import AgentConfig

        prov = get_provider(cfg, provider)
        if model:
            prov.model = model

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
            msg = await prov.complete(
                messages=[Message(role="user", content="Reply with exactly: OK")],
            )
            console.print(f"[green]✓ Response:[/green] {msg.content}")
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
    """List configured MCP servers."""
    cfg = ctx.obj["cfg"]
    if not cfg.mcp_servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return
    table = Table(title="MCP Servers")
    table.add_column("Name", style="bold cyan")
    table.add_column("Transport")
    table.add_column("Command / URL")
    for name, srv in cfg.mcp_servers.items():
        endpoint = srv.url or f"{srv.command} {' '.join(srv.args)}"
        table.add_row(name, srv.transport, endpoint)
    console.print(table)


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
@click.pass_context
def usage(ctx: click.Context, period: str, provider: str | None) -> None:
    """Show LLM usage and cost statistics."""
    cfg = ctx.obj["cfg"]
    from eyetor.tracking.usage import UsageTracker
    tracker = UsageTracker.from_config(cfg.tracking)
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
            s.provider, s.model, str(s.calls),
            str(s.prompt_tokens), str(s.completion_tokens),
            str(s.total_tokens), f"{s.estimated_cost:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_sync() -> None:
    """Synchronous entry point for the 'eyetor' command."""
    cli(obj={})
