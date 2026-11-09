"""Ask the user to confirm a dangerous operation through the channel they use.

Each channel installs a confirmer for the turn it is running
(``current_confirmer.set(...)``): Telegram shows Sí/No buttons, voice asks out
loud. The interactive CLI falls back to the terminal. Where nobody can be
asked (scheduled tasks, a service without a confirmer) the operation is
refused instead of hanging on ``input()``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import sys
from collections.abc import Awaitable, Callable

from eyetor.tracking.context import current_channel

logger = logging.getLogger(__name__)

#: Receives a human-readable description of the operation; returns the answer.
Confirmer = Callable[[str], Awaitable[bool]]

current_confirmer: contextvars.ContextVar[Confirmer | None] = contextvars.ContextVar(
    "current_confirmer", default=None
)

CONFIRM_TIMEOUT = 120.0  # seconds to wait for the user's answer

CANCELLED = json.dumps({"error": "Operation cancelled by user."})
UNAVAILABLE = json.dumps(
    {
        "ok": False,
        "error": (
            "This operation needs the user's explicit confirmation, which cannot be "
            "asked on this channel. It was NOT executed. Tell the user to request it "
            "from Telegram or the interactive CLI."
        ),
    }
)

_YES_RE = re.compile(
    r"^\W*(s[ií]+|vale|ok(ay)?|adelante|hazlo|dale|confirmo|confirmado|de acuerdo|"
    r"claro|por supuesto|correcto|yes|y)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(r"\b(no|cancela\w*|para|espera|nada)\b", re.IGNORECASE)


def is_yes(text: str) -> bool:
    """Whether a free-text answer is an unambiguous yes."""
    text = (text or "").strip()
    return bool(_YES_RE.match(text)) and not _NO_RE.search(text)


async def confirm(description: str) -> bool | None:
    """The user's answer, or ``None`` if nobody can be asked on this channel."""
    confirmer = current_confirmer.get()
    if confirmer is not None:
        try:
            return bool(await confirmer(description))
        except Exception:
            logger.exception("Confirmation failed; treating it as a no")
            return False
    if current_channel.get() in ("", "cli") and sys.stdin.isatty():
        return await _confirm_in_terminal(description)
    return None


async def guard(description: str) -> str | None:
    """``None`` to go ahead, or the JSON error the tool should return."""
    answer = await confirm(description)
    if answer:
        return None
    if answer is None:
        logger.warning(
            "Dangerous operation refused: no way to confirm on channel %r: %s",
            current_channel.get() or "?",
            description,
        )
        return UNAVAILABLE
    return CANCELLED


async def _confirm_in_terminal(description: str) -> bool:
    from rich.console import Console

    Console().print(
        f"\n[bold red]⚠  Dangerous operation requested:[/bold red]\n"
        f"  [yellow]{description}[/yellow]\n"
    )
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, lambda: input("  Confirm? [y/N] "))
    return answer.strip().lower() in {"y", "yes", "s", "si", "sí"}
