"""Per-chat access control: which tools and skills a session may use.

Restriction is configured under ``access`` in the config, keyed by session_id
(e.g. ``cli``, ``telegram-<chat_id>``, or the special ``default``). Each skill
is exposed to the model as a tool named ``skill_<name>``, so restricting tools
and skills is ultimately a single allowlist of tool names per chat.

Semantics:
- No ``access`` config at all  -> unrestricted (backwards compatible).
- ``access`` present: a session uses its own entry. Lookup precedence is
  exact session_id > glob pattern (e.g. ``cli-*``, longest match wins) >
  ``default`` > nothing allowed (restrictive default).
- ``"*"`` in a list means "all tools" / "all skills".
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eyetor.config import VectorConfig

_SKILL_PREFIX = "skill_"


def _norm_skill(name: str) -> str:
    """Normalize a skill name to match the ``skill_<name>`` tool naming."""
    return name.replace("-", "_")


@dataclass
class Access:
    """Resolved tool/skill allowlist for a single chat session."""

    all_tools: bool = False
    all_skills: bool = False
    tools: set[str] = field(default_factory=set)  # exact tool names
    skills: set[str] = field(default_factory=set)  # normalized skill names
    unrestricted: bool = False  # True when no access config exists at all

    def allows_tool(self, name: str) -> bool:
        """Whether the given tool name (incl. ``skill_*``) is allowed."""
        if self.unrestricted:
            return True
        if name.startswith(_SKILL_PREFIX):
            sk = name[len(_SKILL_PREFIX):]
            return self.all_skills or sk in self.skills
        return self.all_tools or name in self.tools

    def allowed_summary(self) -> str:
        """Human-readable summary for a system-prompt note (restricted only)."""
        if self.unrestricted or (self.all_tools and self.all_skills):
            return ""
        parts: list[str] = []
        if self.all_tools:
            parts.append("todas las herramientas")
        elif self.tools:
            parts.append("herramientas: " + ", ".join(sorted(self.tools)))
        if self.all_skills:
            parts.append("todas las skills")
        elif self.skills:
            parts.append("skills: " + ", ".join(sorted(self.skills)))
        allowed = "; ".join(parts) if parts else "ninguna herramienta ni skill"
        return (
            "En este chat tu acceso está restringido por configuración: "
            f"SOLO puedes usar {allowed}. Cualquier otra herramienta o skill no "
            "está disponible aquí; no intentes usarla."
        )


def resolve(config: "VectorConfig | None", session_id: str) -> Access:
    """Resolve the :class:`Access` policy for a session id."""
    access_cfg = getattr(config, "access", None) if config else None
    if not access_cfg:
        return Access(unrestricted=True)

    # Precedence: exact session_id > glob pattern (longest wins) > "default".
    policy = access_cfg.get(session_id)
    if policy is None:
        globs = [
            key
            for key in access_cfg
            if key != "default"
            and any(c in key for c in "*?[")
            and fnmatch.fnmatchcase(session_id, key)
        ]
        if globs:
            policy = access_cfg[max(globs, key=len)]
    if policy is None:
        policy = access_cfg.get("default")
    if policy is None:
        # Restrictive default: nothing allowed when unlisted and no 'default'.
        return Access()

    tools = set(policy.tools or [])
    skills = {_norm_skill(s) for s in (policy.skills or [])}
    return Access(
        all_tools="*" in tools,
        all_skills="*" in skills,
        tools={t for t in tools if t != "*"},
        skills={s for s in skills if s != "*"},
    )
