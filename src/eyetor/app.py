"""Application assembly for ``eyetor start``.

Builds the shared pieces (memory, knowledge base, registries, tools, system
prompt), the channels that use them, and runs those channels until shutdown.
``cli.start`` only parses flags and calls into here.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from eyetor.chat.manager import SessionManager
from eyetor.models.agents import AgentConfig
from eyetor.models.tools import ToolRegistry

logger = logging.getLogger(__name__)

_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

_TOOL_USE_RULES = (
    "## Tool use\n\n"
    "When the user asks for something that needs several tool calls, chain them yourself without asking for unnecessary clarification. "
    "Do not ask the user to choose which command to run or show them script syntax: run them directly. "
    "If you need several steps (e.g. list stores then check each store's prices), do them all in sequence. "
    "Only ask the user when there is a real ambiguity you cannot resolve with the available data."
)


@dataclass
class App:
    """Shared runtime objects every channel's SessionManager is built from."""

    cfg: Any
    prov: Any
    agent_cfg: AgentConfig
    tool_registry: ToolRegistry
    memory: Any
    knowledge: Any
    chatlog: Any
    skill_reg: Any
    agent_reg: Any
    tracker: Any
    cost_estimator: Any
    plugin_registry: Any = None
    mcp_registry: Any = None

    def session_manager(
        self,
        *,
        scheduler=None,
        system_prompt_suffix: str = "",
        enable_conversations: bool = False,
    ) -> SessionManager:
        return SessionManager(
            self.agent_cfg,
            self.prov,
            tool_registry=self.tool_registry,
            system_prompt_suffix=system_prompt_suffix,
            memory_manager=self.memory,
            scheduler=scheduler,
            knowledge=self.knowledge,
            root_config=self.cfg,
            tracker=self.tracker,
            cost_estimator=self.cost_estimator,
            enable_conversations=enable_conversations,
        )


def build_system_prompt(cfg, *, use_host_tools: bool, host_profile: dict, skill_reg) -> str:
    """Base system prompt: identity, tool-use rules, date, host, skills, AGENTS.md."""
    from eyetor.host_info import format_host_prompt

    if use_host_tools:
        identity = (
            "You are Eyetor (sounds like Aitor), an AI assistant with tools that can act on the user's computer. "
            "You can run shell commands, manage files, open URLs and search the internet. "
            "Always reply in Spanish (Spain). "
            "Briefly explain what you are doing. "
            "Tools that delete or wipe data ask the user for confirmation on their own, "
            "so when the user asks you to delete something, just use the tool. "
            "Ask first only before destructive operations the user did not request.\n\n"
        )
    else:
        identity = (
            "You are Eyetor (sounds like Aitor), a helpful AI assistant. "
            "Always reply in Spanish (Spain).\n\n"
        )
    base_system = identity + _TOOL_USE_RULES

    # Inject current date/time so the model knows "today"
    now = datetime.now(ZoneInfo(cfg.scheduler.default_timezone))
    now_str = (
        f"{_DAYS_ES[now.weekday()]} {now.day} de {_MONTHS_ES[now.month - 1]} "
        f"de {now.year}, {now.strftime('%H:%M')} ({now.tzname()})"
    )
    base_system = f"{base_system}\n\nCurrent date/time: {now_str}"

    host_context = format_host_prompt(host_profile)
    if host_context:
        base_system = f"{base_system}\n\n{host_context}"

    skills_context = skill_reg.build_skills_context(skill_reg.list_names())
    if skills_context:
        base_system = f"{base_system}\n\n{skills_context}"

    # Agent instructions (user-managed file with custom behavior rules)
    instructions_path = Path(cfg.agent_instructions).expanduser()
    if instructions_path.exists():
        instructions_text = instructions_path.read_text(encoding="utf-8").strip()
        if instructions_text:
            base_system = f"{base_system}\n\n## Agent Instructions\n\n{instructions_text}"
            logger.info("Loaded agent instructions from %s", instructions_path)

    return base_system


async def build_app(
    cfg,
    *,
    prov,
    tracker,
    cost_estimator,
    use_host_tools: bool,
    interactive: bool,
    agent_provider: str,
    agent_model: str,
) -> App:
    """Create shared services, register every tool and build the agent config."""
    from eyetor.agents.registry import AgentRegistry
    from eyetor.host_info import ensure_host_profile
    from eyetor.memory.manager import MemoryManager
    from eyetor.runtime import runtime_path, write_snapshot
    from eyetor.skills.registry import SkillRegistry
    from eyetor.tools.chat_history import register_chat_history_tools
    from eyetor.tools.delegate import register_delegate_tool
    from eyetor.tools.host import register_install_tool
    from eyetor.tools.images import register_image_tool
    from eyetor.tools.knowledge import register_kb_tools
    from eyetor.tools.skills import register_skill_tools

    host_profile = ensure_host_profile()

    # In interactive (CLI) mode the systemd service is usually the snapshot
    # owner — only write it if no snapshot exists yet, to avoid races.
    if not interactive or not runtime_path().exists():
        write_snapshot(cfg, host_profile=host_profile)

    memory = MemoryManager.from_path(cfg.memory_db_path)

    # Per-day, per-chat conversation archive (group chats). Out of context;
    # queried on demand via the chat_history_* tools.
    chatlog = None
    if cfg.chatlog_enabled:
        from eyetor.chatlog.manager import ChatLogManager

        chatlog = ChatLogManager.from_path(
            cfg.chatlog_db_path, retention_days=cfg.chatlog_retention_days
        )

    # Knowledge base (optional, hybrid BM25 + semantic retrieval).
    # Created when enabled even without static workspaces: Telegram document
    # uploads register their own per-chat ephemeral workspaces on the fly.
    knowledge = None
    if cfg.knowledge and cfg.knowledge.enabled:
        from eyetor.knowledge.manager import KnowledgeManager

        knowledge = KnowledgeManager.from_config(cfg.knowledge)
        if interactive and cfg.knowledge.auto_cwd_workspace:
            knowledge.register_cwd_workspace(Path.cwd())

    skill_reg = SkillRegistry()
    skill_reg.discover(cfg.skills_dirs)

    # Subagent registry (Markdown files under agents_dirs)
    agent_reg = AgentRegistry()
    agent_reg.discover(cfg.agents_dirs)

    base_system = build_system_prompt(
        cfg, use_host_tools=use_host_tools, host_profile=host_profile, skill_reg=skill_reg
    )

    plugin_registry = None
    if cfg.plugins_dirs:
        from eyetor.plugins.registry import PluginRegistry

        plugin_registry = PluginRegistry()
        plugin_registry.load_all(cfg.plugins_dirs)
        await plugin_registry.run_init()

    # Shared tool registry. Per-session tools (memory, scheduler) are added by
    # each ChatSession on its own copy.
    tool_registry = ToolRegistry(plugin_registry=plugin_registry)
    if use_host_tools:
        register_install_tool(tool_registry, host_profile)
    register_skill_tools(
        tool_registry, skill_reg, use_host_tools=use_host_tools, host_profile=host_profile
    )
    base_system += register_image_tool(tool_registry, cfg, tracker, cost_estimator)
    if knowledge is not None:
        register_kb_tools(tool_registry, knowledge)
    if chatlog is not None:
        register_chat_history_tools(tool_registry, chatlog)

    # MCP servers — connect and register tools
    mcp_registry = None
    if cfg.mcp_servers:
        from eyetor.mcp.registry import McpRegistry

        mcp_registry = McpRegistry(cfg.mcp_servers)
        await mcp_registry.connect_all()
        mcp_registry.register_all_into(tool_registry)
        report = mcp_registry.get_degraded_report()
        if report.is_degraded:
            base_system = f"{base_system}\n\n{report.format_for_prompt()}"
            logger.warning("MCP degraded — failed servers: %s", list(report.failed.keys()))

    base_system += register_delegate_tool(tool_registry, cfg, agent_reg, prov)

    agent_cfg = AgentConfig(
        name="eyetor",
        provider=agent_provider,
        model=agent_model,
        system_prompt=base_system,
        temperature=prov.temperature,
    )

    return App(
        cfg=cfg,
        prov=prov,
        agent_cfg=agent_cfg,
        tool_registry=tool_registry,
        memory=memory,
        knowledge=knowledge,
        chatlog=chatlog,
        skill_reg=skill_reg,
        agent_reg=agent_reg,
        tracker=tracker,
        cost_estimator=cost_estimator,
        plugin_registry=plugin_registry,
        mcp_registry=mcp_registry,
    )


def build_channels(app: App, *, interactive: bool, cli_session_id: str) -> tuple[list, Any]:
    """Create the enabled channels; return (channels, scheduler or None)."""
    cfg = app.cfg

    # Scheduler (shared across all channels). Only the non-interactive
    # "server" instance owns APScheduler — interactive CLIs would
    # duplicate every job otherwise.
    scheduler = None
    sched_cfg = cfg.scheduler
    if sched_cfg.enabled and not interactive:
        from eyetor.scheduler.channel import SchedulerChannel
        from eyetor.scheduler.store import SchedulerStore

        tg_token = cfg.channels.telegram.bot_token if cfg.channels.telegram.enabled else None
        # Scheduler needs a SessionManager; a dedicated one without scheduler
        # tools to avoid recursion.
        scheduler = SchedulerChannel(
            store=SchedulerStore(sched_cfg.db_path),
            session_manager=app.session_manager(),
            bot_token=tg_token,
            default_timezone=sched_cfg.default_timezone,
        )

    channels = []

    if interactive:
        from eyetor.channels.cli_channel import CliChannel

        channels.append(
            CliChannel(
                app.session_manager(scheduler=scheduler, enable_conversations=True),
                session_id=cli_session_id,
                skill_reg=app.skill_reg,
                agent_reg=app.agent_reg,
            )
        )

    tg_cfg = cfg.channels.telegram
    if tg_cfg.enabled and tg_cfg.bot_token and not interactive:
        from eyetor.channels.telegram import TelegramChannel

        channels.append(
            TelegramChannel(
                app.session_manager(scheduler=scheduler, enable_conversations=True),
                tg_cfg,
                skill_reg=app.skill_reg,
                scheduler=scheduler,
                tracker=app.tracker,
                full_config=cfg,
                agent_reg=app.agent_reg,
                chatlog=app.chatlog,
                knowledge=app.knowledge,
            )
        )

    voice_cfg = cfg.channels.voice
    if voice_cfg.enabled and not interactive:
        from eyetor.channels.voice import VOICE_PROMPT_SUFFIX, VoiceChannel

        channels.append(
            VoiceChannel(
                app.session_manager(
                    scheduler=scheduler,
                    system_prompt_suffix=VOICE_PROMPT_SUFFIX,
                    enable_conversations=True,
                ),
                voice_cfg,
            )
        )

    if scheduler:
        channels.append(scheduler)

    return channels, scheduler


async def serve(app: App, channels: list, *, interactive: bool) -> None:
    """Run channels until /exit (interactive) or SIGTERM/SIGINT, then clean up."""
    cfg = app.cfg
    knowledge = app.knowledge

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
            cli_channel = next(ch for ch in channels if type(ch).__name__ == "CliChannel")
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
        if app.mcp_registry:
            await app.mcp_registry.close_all()
        if app.plugin_registry:
            await app.plugin_registry.run_shutdown()
