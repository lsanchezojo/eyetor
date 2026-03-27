"""Skill discovery and loading from SKILL.md files (agentskills.io format)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Matches YAML frontmatter between --- delimiters
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)

# Valid skill name: 1-64 chars, lowercase alphanumeric + hyphens
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")


@dataclass
class SkillMetadata:
    """Lightweight metadata loaded from SKILL.md frontmatter.

    Loaded at startup for all skills — kept small (~100 tokens).
    """

    name: str
    description: str
    path: Path  # Directory containing SKILL.md
    license: str = ""
    compatibility: str = ""
    author: str = ""
    version: str = ""


@dataclass
class SkillInfo:
    """Full skill information including SKILL.md instructions."""

    metadata: SkillMetadata
    instructions: str  # Full SKILL.md body (Markdown)
    scripts: list[Path] = field(default_factory=list)


def load_skill_metadata(skill_dir: Path) -> SkillMetadata | None:
    """Parse SKILL.md in skill_dir and return metadata, or None if invalid."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    text = skill_md.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None

    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None

    name = fm.get("name", "")
    description = fm.get("description", "")
    if not name or not description:
        return None

    # Validate name format
    if not _SKILL_NAME_RE.match(name):
        return None

    # name must match directory name
    if name != skill_dir.name:
        return None

    meta_block = fm.get("metadata", {}) or {}
    return SkillMetadata(
        name=name,
        description=description,
        path=skill_dir,
        license=fm.get("license", ""),
        compatibility=fm.get("compatibility", ""),
        author=meta_block.get("author", ""),
        version=str(meta_block.get("version", "")),
    )


def load_skill_info(metadata: SkillMetadata) -> SkillInfo:
    """Load full SKILL.md instructions and discover scripts."""
    skill_md = metadata.path / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    instructions = match.group(2).strip() if match else text

    scripts_dir = metadata.path / "scripts"
    scripts: list[Path] = []
    if scripts_dir.exists():
        scripts = sorted(scripts_dir.iterdir())

    return SkillInfo(metadata=metadata, instructions=instructions, scripts=scripts)


def discover_skills(skills_dirs: list[str | Path]) -> list[SkillMetadata]:
    """Scan all skills directories and return valid SkillMetadata objects.

    Skills dirs are searched in order; later dirs override earlier ones
    if there is a name conflict.
    """
    found: dict[str, SkillMetadata] = {}
    for raw_dir in skills_dirs:
        base = Path(raw_dir).expanduser().resolve()
        if not base.exists():
            continue
        for candidate in sorted(base.iterdir()):
            if not candidate.is_dir():
                continue
            meta = load_skill_metadata(candidate)
            if meta is not None:
                found[meta.name] = meta
    return list(found.values())
