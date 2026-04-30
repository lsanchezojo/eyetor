"""Global skill registry — discovery, activation, and instruction retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.skills.loader import SkillCommand, SkillInfo, SkillMetadata, discover_skills, load_skill_info

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skills.

    Progressive disclosure:
    1. Metadata (name + description) — always loaded at startup
    2. Instructions (full SKILL.md body) — loaded when skill is activated
    3. Scripts — listed via list_scripts(); loaded on demand by executor
    """

    def __init__(self) -> None:
        self._metadata: dict[str, SkillMetadata] = {}
        self._loaded: dict[str, SkillInfo] = {}

    def discover(self, skills_dirs: list[str | Path]) -> None:
        """Scan skills directories and register metadata."""
        found = discover_skills(skills_dirs)
        for meta in found:
            self._metadata[meta.name] = meta
            logger.debug("Discovered skill: %s (%s)", meta.name, meta.path)
        logger.info("Skills discovered: %d", len(found))

    def list_names(self) -> list[str]:
        """Names of all discovered skills."""
        return list(self._metadata.keys())

    def get_metadata(self, name: str) -> SkillMetadata:
        """Return metadata for a skill by name."""
        if name not in self._metadata:
            raise KeyError(f"Skill not found: {name!r}")
        return self._metadata[name]

    def all_metadata(self) -> list[SkillMetadata]:
        """All discovered skill metadata objects."""
        return list(self._metadata.values())

    def activate(self, name: str) -> SkillInfo:
        """Load and cache the full skill info (instructions + scripts)."""
        if name in self._loaded:
            return self._loaded[name]
        meta = self.get_metadata(name)
        info = load_skill_info(meta)
        self._loaded[name] = info
        logger.debug("Activated skill: %s", name)
        return info

    def get_instructions(self, name: str) -> str:
        """Return the skill instructions (activates if needed)."""
        return self.activate(name).instructions

    def list_scripts(self, name: str) -> list[Path]:
        """Return list of executable script paths for a skill."""
        return self.activate(name).scripts

    def build_skills_context(self, skill_names: list[str]) -> str:
        """Build a system prompt section for the given activated skills."""
        if not skill_names:
            return ""
        parts = [
            "## Available Skills",
            (
                "> **Skill tool call format:** pass only subcommands and flags in `args`. "
                "Code blocks in skill docs show manual shell usage (with script paths or "
                "variables like `$PWCLI`) — omit those prefixes when calling the tool."
            ),
        ]
        for name in skill_names:
            try:
                info = self.activate(name)
                parts.append(f"\n### Skill: {name}")
                parts.append(info.instructions)
            except KeyError:
                logger.warning("Skill not found in registry: %s", name)
        return "\n".join(parts)

    def build_skills_summary_context(
        self,
        skill_names: list[str],
        *,
        expanded_skill_names: list[str] | None = None,
    ) -> str:
        """Build a compact system prompt skill section for small local models.

        The summary intentionally avoids SKILL.md bodies. It includes only
        name, description and public script names, while still allowing callers
        to opt in to full instructions for a small set of relevant skills.
        """
        if not skill_names:
            return ""
        expanded = set(expanded_skill_names or [])
        parts = [
            "## Available Skills (summary)",
            (
                "Call a skill via its `skill_<name>` tool. Pass only subcommands "
                "and flags in `args`; do not include interpreter, script path or wrapper."
            ),
        ]
        for name in skill_names:
            try:
                meta = self.get_metadata(name)
            except KeyError:
                logger.warning("Skill not found in registry: %s", name)
                continue
            script_names: list[str] = []
            scripts_dir = meta.path / "scripts"
            if scripts_dir.exists():
                script_names = [
                    p.stem
                    for p in sorted(scripts_dir.iterdir())
                    if p.is_file() and not p.name.startswith(("_", "."))
                ]
            line = f"- **{meta.name}**: {meta.description}"
            if script_names:
                line += f" Public scripts: {', '.join(script_names)}."
            parts.append(line)
            if name in expanded:
                info = self.activate(name)
                parts.append(f"\n### Expanded Skill: {name}")
                parts.append(info.instructions)
        return "\n".join(parts)

    def get_all_commands(self) -> list[tuple[SkillMetadata, SkillCommand]]:
        """Return all skill-declared channel commands with their parent metadata."""
        result = []
        for meta in self._metadata.values():
            for cmd in meta.commands:
                result.append((meta, cmd))
        return result

    def available_skills_summary(self) -> str:
        """One-line summary of all skills for system prompts (metadata only)."""
        if not self._metadata:
            return ""
        lines = ["## Available Skills (summary)"]
        for meta in self._metadata.values():
            lines.append(f"- **{meta.name}**: {meta.description}")
        return "\n".join(lines)
