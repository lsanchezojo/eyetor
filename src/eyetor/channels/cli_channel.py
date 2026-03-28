"""CLI channel — interactive terminal chat with rich formatting."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from eyetor.channels.base import BaseChannel
from eyetor.chat.manager import SessionManager

logger = logging.getLogger(__name__)

_SESSION_ID = "cli-local"

_HELP = """
[bold]Eyetor CLI Commands[/bold]
  /help     — show this help
  /reset    — clear conversation history
  /history  — show conversation history
  /skills   — list available skills with descriptions
  /exit     — quit
"""


class CliChannel(BaseChannel):
    """Interactive CLI channel using rich for formatted output."""

    def __init__(
        self,
        session_manager: SessionManager,
        skill_reg=None,
    ) -> None:
        self._manager = session_manager
        self._skill_reg = skill_reg
        self._console = Console()
        self._running = False

    async def start(self) -> None:
        """Start the interactive chat loop."""
        self._running = True
        self._console.print("[bold green]Eyetor[/bold green] — Multi-agent AI system")
        self._console.print("Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.\n")

        session = self._manager.get_or_create(_SESSION_ID)

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
                session.reset()
                self._console.print("[dim]Conversation reset.[/dim]")
                continue
            elif user_input.lower() == "/history":
                self._show_history(session)
                continue
            elif user_input.lower() == "/skills":
                self._console.print(_format_skills(self._skill_reg))
                continue
            elif user_input.lower() == "/help":
                self._console.print(_HELP)
                continue

            # Send to agent and stream response
            self._console.print("\n[bold green]Assistant[/bold green]:", end=" ")
            response_text = ""
            try:
                with self._console.status("", spinner="dots"):
                    async for chunk in session.send(user_input):
                        response_text += chunk
                self._console.print(Markdown(response_text))
            except Exception as exc:
                self._console.print(f"[red]Error: {exc}[/red]")
                logger.error("Chat error: %s", exc)
            self._console.print()

    async def stop(self) -> None:
        self._running = False

    def _show_history(self, session) -> None:
        history = session.get_history()
        if not history:
            self._console.print("[dim]No conversation history.[/dim]")
            return
        for msg in history:
            role_color = {"user": "blue", "assistant": "green", "tool": "yellow"}.get(msg.role, "white")
            self._console.print(f"[{role_color}]{msg.role}[/{role_color}]: {msg.content or '[tool call]'}")


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

