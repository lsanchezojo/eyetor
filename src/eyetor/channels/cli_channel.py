"""CLI channel — interactive terminal chat with rich formatting."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from eyetor.channels.base import BaseChannel
from eyetor.chat.manager import SessionManager
from eyetor.tracking.context import current_channel

logger = logging.getLogger(__name__)


_IMAGE_MARKER_RE = re.compile(r"\[IMAGE:(.*?)\]")

_HELP = """
[bold]Eyetor CLI Commands[/bold]
  /help              — show this help
  /reset             — start a new conversation (old one is kept)
  /chats             — list recent conversations
  /chat N            — resume conversation N from /chats
  /history           — show conversation history
  /skills            — list available skills with descriptions
  /agents            — list loaded subagent definitions
  /tools             — list registered tools
  /model [name] [m]  — list or change provider (and optionally model)
  /exit              — quit
"""


class CliChannel(BaseChannel):
    """Interactive CLI channel using rich for formatted output."""

    def __init__(
        self,
        session_manager: SessionManager,
        session_id: str,
        skill_reg=None,
        agent_reg=None,
    ) -> None:
        self._manager = session_manager
        self._session_id = session_id
        self._skill_reg = skill_reg
        self._agent_reg = agent_reg
        self._console = Console()
        self._running = False

    async def start(self) -> None:
        """Start the interactive chat loop."""
        self._running = True
        self._console.print("[bold green]Eyetor[/bold green] — Multi-agent AI system")
        self._console.print(
            f"Sesión: [cyan]{self._session_id}[/cyan]   "
            "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.\n"
        )

        # Each `eyetor start` opens a fresh conversation (the index reuses an
        # empty one, so quick start-and-quit runs don't accumulate entries).
        self._manager.new_conversation(self._session_id)
        session = self._manager.get_or_create(self._session_id)

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: Prompt.ask("[bold blue]You[/bold blue]"),
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "/exit":
                self._console.print("[dim]Goodbye.[/dim]")
                break
            elif user_input.lower() == "/reset":
                conv_id = self._manager.new_conversation(self._session_id)
                session = self._manager.get_or_create(self._session_id)
                from eyetor.skills.lifecycle import run_reset_hooks
                await run_reset_hooks(self._skill_reg, self._session_id, conversation_id=conv_id)
                self._console.print("[dim]Nueva conversación iniciada.[/dim]")
                continue
            elif user_input.lower() == "/chats":
                self._console.print(self._format_chats())
                continue
            elif user_input.lower().startswith("/chat "):
                switched = self._switch_chat(user_input.split(maxsplit=1)[1])
                if switched:
                    session = self._manager.get_or_create(self._session_id)
                    title = switched.get("title") or "(sin título)"
                    self._console.print(
                        f"[dim]Conversación retomada: «{title}» "
                        f"({len(session.get_history())} mensajes).[/dim]"
                    )
                continue
            elif user_input.lower() == "/history":
                self._show_history(session)
                continue
            elif user_input.lower() == "/skills":
                self._console.print(_format_skills(self._skill_reg))
                continue
            elif user_input.lower() == "/agents":
                self._console.print(_format_agents(self._agent_reg))
                continue
            elif user_input.lower() == "/tools":
                self._console.print(_format_tools(session.tool_registry))
                continue
            elif user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) == 1:
                    # List available providers
                    providers = self._manager.list_providers()
                    current = session.provider
                    current_model = getattr(current, "model", "?")
                    provider_name = _describe_provider(current)
                    lines = [
                        f"[bold]Proveedor actual:[/bold] [cyan]{provider_name}[/cyan] (modelo: {current_model})\n"
                    ]
                    lines.append("[bold]Proveedores disponibles:[/bold]")
                    for name, model in providers.items():
                        lines.append(f"  [cyan]{name}[/cyan] — {model}")
                    lines.append("\n[dim]Uso: /model <provider> [model][/dim]")
                    self._console.print("\n".join(lines))
                else:
                    provider_name = parts[1]
                    model_override = parts[2] if len(parts) > 2 else None
                    try:
                        msg = session.change_provider(provider_name, model_override)
                        self._console.print(f"[green]{msg}[/green]")
                    except Exception as exc:
                        self._console.print(f"[red]Error: {exc}[/red]")
                continue
            elif user_input.lower() == "/help":
                self._console.print(_HELP)
                continue

            # Send to agent and stream response
            self._console.print("\n[bold green]Assistant[/bold green]:", end=" ")
            response_text = ""
            try:
                current_channel.set("cli")
                with self._console.status("", spinner="dots"):
                    async for chunk in session.send(user_input):
                        response_text += chunk
                # Show reasoning/thinking block if present
                if session.last_reasoning:
                    self._console.print(
                        f"[dim]💭 Thinking:\n{session.last_reasoning.strip()}[/dim]\n"
                    )
                # Strip image markers from text
                clean_text = _IMAGE_MARKER_RE.sub("", response_text).strip()
                if clean_text:
                    self._console.print(Markdown(clean_text))
                # Show generated images from markers + tool results
                for img_path in _collect_image_paths(response_text, session):
                    self._console.print(f"[green]Imagen guardada:[/green] {img_path}")
            except Exception as exc:
                self._console.print(f"[red]Error: {exc}[/red]")
                logger.error("Chat error: %s", exc)
            self._console.print()

    async def stop(self) -> None:
        self._running = False

    def _format_chats(self) -> str:
        """Numbered list of this chat's conversations, newest first."""
        items = self._manager.list_conversations(self._session_id)
        if not items:
            return "[dim]No hay conversaciones guardadas (¿persistencia desactivada?).[/dim]"
        lines = ["[bold]Conversaciones[/bold]"]
        for n, it in enumerate(items[:15], start=1):
            mark = "→" if it.get("active") else " "
            title = it.get("title") or "(sin título)"
            date = (it.get("updated_at") or "")[:16].replace("T", " ")
            lines.append(f" {mark} {n}. {title}  [dim]{date}[/dim]")
        lines.append("[dim]Usa /chat N para retomar una.[/dim]")
        return "\n".join(lines)

    def _switch_chat(self, arg: str) -> dict | None:
        """Resolve /chat N against the same ordering /chats prints."""
        items = self._manager.list_conversations(self._session_id)
        try:
            n = int(arg.strip())
            target = items[n - 1] if 1 <= n <= len(items) else None
        except ValueError:
            target = None
        if target is None:
            self._console.print("[dim]Uso: /chat N (ver números con /chats).[/dim]")
            return None
        if not self._manager.switch_conversation(self._session_id, target["id"]):
            self._console.print("[red]No se pudo cambiar de conversación.[/red]")
            return None
        return target

    def _show_history(self, session) -> None:
        history = session.get_history()
        if not history:
            self._console.print("[dim]No conversation history.[/dim]")
            return
        for msg in history:
            role_color = {"user": "blue", "assistant": "green", "tool": "yellow"}.get(
                msg.role, "white"
            )
            self._console.print(
                f"[{role_color}]{msg.role}[/{role_color}]: {msg.content or '[tool call]'}"
            )


def _describe_provider(provider) -> str:
    """Best-effort human-readable name for the active provider.

    - TrackingProvider exposes ``_provider_name``.
    - FallbackProvider exposes ``_providers`` (each wrapped); list them as a chain.
    - Anything else falls back to the class name.
    """
    name = getattr(provider, "_provider_name", None)
    if name:
        return name
    inner = getattr(provider, "_providers", None)
    if inner:
        names = [
            getattr(p, "_provider_name", None) or type(p).__name__
            for p in inner
        ]
        return "fallback(" + " → ".join(names) + ")"
    return type(provider).__name__


def _collect_image_paths(buffer: str, session) -> list[str]:
    """Collect image paths from [IMAGE:] markers and generate_image tool results.

    Only scans tool results from the latest turn (after the last user message).
    """
    paths: set[str] = set()
    for p in _IMAGE_MARKER_RE.findall(buffer):
        paths.add(p.strip())
    # Scan only messages after the last user message
    history = session.get_history()
    for msg in reversed(history):
        if msg.role == "user":
            break
        if msg.role == "tool" and msg.content:
            try:
                data = json.loads(msg.content)
                if (
                    isinstance(data, dict)
                    and data.get("status") == "ok"
                    and "image_path" in data
                ):
                    paths.add(data["image_path"])
            except (json.JSONDecodeError, TypeError):
                pass
    return list(paths)


def _format_skills(skill_reg) -> str:
    """Return a formatted skills list with descriptions for display in any channel."""
    if skill_reg is None:
        return "[dim]No skills configured.[/dim]"
    metadata = skill_reg.all_metadata()
    if not metadata:
        return "[dim]No skills configured.[/dim]"
    lines = ["[bold]Available skills:[/bold]"]
    for m in metadata:
        lines.append(f"  [cyan]{m.name}[/cyan] — {m.description}")
    return "\n".join(lines)


def _format_agents(agent_reg) -> str:
    """Return a formatted subagent list with descriptions for display."""
    if agent_reg is None:
        return "[dim]No agents configured.[/dim]"
    agents = agent_reg.all()
    if not agents:
        return "[dim]No agents configured.[/dim]"
    lines = ["[bold]Available agents:[/bold]"]
    for a in agents:
        lines.append(f"  [cyan]{a.name}[/cyan] — {a.description}")
    return "\n".join(lines)


def _format_tools(tool_registry) -> str:
    """Return a formatted list of registered tools for display."""
    if tool_registry is None:
        return "[dim]No tools registered.[/dim]"
    tools = tool_registry._tools
    if not tools:
        return "[dim]No tools registered.[/dim]"
    lines = [f"[bold]Registered tools ({len(tools)}):[/bold]"]
    for name, defn in tools.items():
        lines.append(f"  [cyan]{name}[/cyan] — {defn.description}")
    return "\n".join(lines)
