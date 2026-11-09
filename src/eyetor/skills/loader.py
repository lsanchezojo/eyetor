"""Skill discovery and loading from SKILL.md files (agentskills.io format)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Matches YAML frontmatter between --- delimiters
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)

# Valid skill name: 1-64 chars, lowercase alphanumeric + hyphens
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")

# Valid Telegram command name: lowercase alphanumeric + underscores
_COMMAND_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# Core commands that skills cannot override
_RESERVED_COMMANDS = {"start", "reset", "skills", "tasks", "help", "chats", "chat"}


@dataclass
class SkillCommand:
    """A channel command declared by a skill in SKILL.md frontmatter."""

    name: str  # without "/", e.g. "compra"
    description: str
    action: str  # "script" or "prompt"
    script: str = ""  # relative to skill's scripts/ dir
    args: list[str] = field(default_factory=list)
    prompt: str = ""  # template with {args} placeholder
    parse_mode: str = "HTML"  # Telegram parse mode for script output
    # If true (script commands only), replying to a result message re-invokes
    # the script to continue it. The script may emit a "reply_token" in its JSON
    # output (opaque to the core) that is handed back via EYETOR_REPLY_TOKEN.
    reply_continue: bool = False


@dataclass
class SkillTool:
    """A typed tool declared by a skill in SKILL.md frontmatter.

    Calling it runs ``scripts/<script> <argv...>``, then the values of the
    ``positional`` parameters in that order, then one ``--<param-name>=<value>``
    per other argument given (underscores become dashes; booleans become a bare
    flag when true; lists/objects are passed as JSON).
    """

    name: str
    description: str
    script: str  # relative to the skill's scripts/ dir
    parameters: dict  # JSON Schema object
    argv: list[str] = field(default_factory=list)  # fixed leading args, e.g. ["add"]
    positional: list[str] = field(default_factory=list)  # params passed without a flag
    confirm: bool = False  # ask the user before running (destructive operations)


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
    timeout: float | None = None  # Per-skill timeout override (seconds); None = use default
    commands: list[SkillCommand] = field(default_factory=list)
    # Optional script (relative to scripts/) run as "<script> --session-reset"
    # with EYETOR_SESSION_ID when an eyetor conversation is reset.
    on_reset: str | None = None
    tools: list[SkillTool] = field(default_factory=list)
    # False hides the generic ``skill_<name>`` (free-text ``args``) tool, for
    # skills fully covered by their typed tools.
    args_tool: bool = True
    # Tool-gating group for this skill's tools and the keywords that enable it
    # on a turn (accent/case-insensitive, whole words; regex allowed). Without
    # a group the skill's tools are sent on every turn.
    group: str | None = None
    triggers: list[str] = field(default_factory=list)


_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _parse_skill_tools(skill_name: str, skill_dir: Path, raw_tools) -> list[SkillTool]:
    tools: list[SkillTool] = []
    for raw in raw_tools or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name", "")
        script = raw.get("script", "")
        params = raw.get("parameters") or {"type": "object", "properties": {}}
        if not _TOOL_NAME_RE.match(name or "") or not raw.get("description"):
            logger.warning("Skill '%s': skipping invalid tool %r", skill_name, name)
            continue
        if not script or not (skill_dir / "scripts" / script).exists():
            logger.warning("Skill '%s': tool %r script %r not found", skill_name, name, script)
            continue
        if not isinstance(params, dict) or params.get("type") != "object":
            logger.warning("Skill '%s': tool %r parameters must be a JSON Schema object", skill_name, name)
            continue
        tools.append(SkillTool(
            name=name,
            description=raw["description"].strip(),
            script=script,
            parameters=params,
            argv=[str(a) for a in raw.get("argv", []) or []],
            positional=[str(a) for a in raw.get("positional", []) or []],
            confirm=bool(raw.get("confirm", False)),
        ))
    return tools


_GROUP_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def _parse_gating(skill_name: str, raw) -> dict:
    """``gating: {group: shopping, triggers: [compra, ticket, ...]}``."""
    if not raw:
        return {}
    group = raw.get("group") if isinstance(raw, dict) else None
    triggers = [str(t) for t in (raw.get("triggers") or [])] if isinstance(raw, dict) else []
    if not group or not _GROUP_NAME_RE.match(str(group)) or not triggers:
        logger.warning("Skill '%s': ignoring invalid gating %r", skill_name, raw)
        return {}
    return {"group": str(group), "triggers": triggers}


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

    # Parse optional commands
    commands: list[SkillCommand] = []
    for raw_cmd in fm.get("commands", []) or []:
        if not isinstance(raw_cmd, dict):
            continue
        cmd_name = raw_cmd.get("name", "")
        cmd_desc = raw_cmd.get("description", "")
        action = raw_cmd.get("action", "")
        if not cmd_name or not cmd_desc or action not in ("script", "prompt"):
            logger.warning("Skill '%s': skipping invalid command %r", name, cmd_name)
            continue
        if not _COMMAND_NAME_RE.match(cmd_name):
            logger.warning("Skill '%s': invalid command name %r", name, cmd_name)
            continue
        if cmd_name in _RESERVED_COMMANDS:
            logger.warning("Skill '%s': command %r is reserved", name, cmd_name)
            continue
        if action == "script":
            script_name = raw_cmd.get("script", "")
            if not script_name:
                logger.warning("Skill '%s': command %r missing 'script' field", name, cmd_name)
                continue
            script_path = skill_dir / "scripts" / script_name
            if not script_path.exists():
                logger.warning("Skill '%s': script %r not found", name, script_name)
                continue
        if action == "prompt" and not raw_cmd.get("prompt", ""):
            logger.warning("Skill '%s': command %r missing 'prompt' field", name, cmd_name)
            continue
        commands.append(SkillCommand(
            name=cmd_name,
            description=cmd_desc,
            action=action,
            script=raw_cmd.get("script", ""),
            args=list(raw_cmd.get("args", []) or []),
            prompt=raw_cmd.get("prompt", ""),
            parse_mode=raw_cmd.get("parse_mode", "HTML"),
            reply_continue=bool(raw_cmd.get("reply_continue", False)),
        ))

    # Optional per-skill timeout (seconds)
    raw_timeout = fm.get("timeout")
    timeout = float(raw_timeout) if raw_timeout is not None else None

    # Optional session-reset hook (script name relative to scripts/).
    on_reset = fm.get("on_reset") or None
    if on_reset:
        if not (skill_dir / "scripts" / on_reset).exists():
            logger.warning("Skill '%s': on_reset script %r not found", name, on_reset)
            on_reset = None

    return SkillMetadata(
        name=name,
        description=description,
        path=skill_dir,
        license=fm.get("license", ""),
        compatibility=fm.get("compatibility", ""),
        author=meta_block.get("author", ""),
        version=str(meta_block.get("version", "")),
        timeout=timeout,
        commands=commands,
        on_reset=on_reset,
        tools=_parse_skill_tools(name, skill_dir, fm.get("tools")),
        args_tool=bool(fm.get("args_tool", True)),
        **_parse_gating(name, fm.get("gating")),
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
        scripts = sorted(
            p for p in scripts_dir.iterdir()
            if p.is_file() and not p.name.startswith(("_", "."))
        )

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
