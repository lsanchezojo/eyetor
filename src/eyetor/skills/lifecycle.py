"""Generic skill lifecycle hooks (no knowledge of any specific skill).

Currently: session-reset propagation. When an eyetor conversation is reset, any
skill that declares an ``on_reset`` script in its SKILL.md frontmatter gets it
run as ``<script> --session-reset`` with ``EYETOR_SESSION_ID`` in the env, so it
can clear its own per-session state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run_reset_hooks(
    skill_reg, session_id: str, conversation_id: str | None = None
) -> None:
    """Run every skill's ``on_reset`` hook for a reset session (best effort)."""
    if skill_reg is None:
        return
    from eyetor.skills.executor import run_script

    env = {"EYETOR_SESSION_ID": session_id}
    if conversation_id:
        env["EYETOR_CONVERSATION_ID"] = conversation_id

    for meta in skill_reg.all_metadata():
        if not meta.on_reset:
            continue
        script = meta.path / "scripts" / meta.on_reset
        if not script.exists():
            continue
        try:
            await run_script(
                script,
                ["--session-reset"],
                timeout=30.0,
                env=env,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Skill '%s' on_reset hook failed for %s: %s", meta.name, session_id, exc)
