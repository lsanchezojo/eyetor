"""Tracking context — contextvars threaded into every recorded LLM call.

These vars let the ``TrackingProvider`` tag each ``usage`` row with the agent,
phase, channel and a per-turn trace id, without changing every call signature.
``tracking_context`` is a context manager that only sets what it is given and
``reset()``s on exit, so a value cannot leak into whatever runs next on the
same task (the previous code used bare ``.set()`` which did leak).

``current_session_id`` stays defined in ``providers.tracking`` (it has external
importers) and is only re-exported here for a single import site.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import uuid
from typing import Iterator

from eyetor.providers.tracking import current_session_id  # re-export

__all__ = [
    "current_session_id",
    "current_agent",
    "current_phase",
    "current_channel",
    "current_trace_id",
    "skip_limit",
    "new_trace_id",
    "make_digest",
    "effective_phase",
    "tracking_context",
]

current_agent: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent", default=""
)
current_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_phase", default=""
)
current_channel: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_channel", default=""
)
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default=""
)
skip_limit: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "skip_limit", default=False
)


def new_trace_id() -> str:
    """A short correlation id for all model calls within one user turn."""
    return uuid.uuid4().hex[:16]


def make_digest(text: str | None, preview_chars: int = 120) -> str:
    """Privacy-preserving fingerprint of a prompt/response.

    Format: ``sha256:<first 16 hex>|<whitespace-collapsed first N chars>``.
    Never stores the full content; the preview gives human context in the CLI.
    """
    s = text or ""
    digest = hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]
    preview = " ".join(s.split())[:preview_chars]
    return f"sha256:{digest}|{preview}"


def effective_phase(default: str) -> str:
    """Return the already-set phase if any, else ``default``.

    Lets an outer phase (e.g. ``routing``) win over an inner agent's
    ``agent`` phase instead of being overwritten.
    """
    return current_phase.get() or default


@contextlib.contextmanager
def tracking_context(
    *,
    session_id: str | None = None,
    agent: str | None = None,
    phase: str | None = None,
    channel: str | None = None,
    trace_id: str | None = None,
    skip_limit_flag: bool | None = None,
) -> Iterator[None]:
    """Set only the provided tracking vars; reset all of them on exit.

    Vars left as ``None`` inherit the surrounding context unchanged.
    """
    tokens: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
    if session_id is not None:
        tokens.append((current_session_id, current_session_id.set(session_id)))
    if agent is not None:
        tokens.append((current_agent, current_agent.set(agent)))
    if phase is not None:
        tokens.append((current_phase, current_phase.set(phase)))
    if channel is not None:
        tokens.append((current_channel, current_channel.set(channel)))
    if trace_id is not None:
        tokens.append((current_trace_id, current_trace_id.set(trace_id)))
    if skip_limit_flag is not None:
        tokens.append((skip_limit, skip_limit.set(skip_limit_flag)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
