"""Subagent discovery and loading from ``<name>.md`` files.

Inspired by Anthropic's Agent SDK subagent definitions, adapted to eyetor:
each agent lives in its own Markdown file with YAML frontmatter (metadata)
and a body (the system prompt). Discovered files are exposed through
:class:`AgentRegistry` and can be referenced by name from the orchestrator
workflow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_AGENT_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9_-]{0,62}[a-z0-9])?$")


@dataclass
class AgentDefinition:
    """An agent loaded from disk.

    The Markdown body becomes the agent's ``system_prompt``. Frontmatter
    fields are optional except ``name`` and ``description`` — when
    ``provider``/``model``/``temperature`` are omitted, the orchestrator
    falls back to its own defaults.
    """

    name: str
    description: str
    system_prompt: str
    path: Path
    provider: str = ""
    model: str = ""
    temperature: float | None = None


def load_agent(file_path: Path) -> AgentDefinition | None:
    """Parse a single ``<name>.md`` agent file. Returns ``None`` if invalid."""
    if not file_path.exists() or not file_path.is_file():
        return None
    if file_path.suffix.lower() != ".md":
        return None

    text = file_path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        logger.warning("Agent %s: missing YAML frontmatter", file_path)
        return None

    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Agent %s: invalid YAML frontmatter (%s)", file_path, exc)
        return None

    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()
    if not name or not description:
        logger.warning("Agent %s: missing required 'name' or 'description'", file_path)
        return None

    if not _AGENT_NAME_RE.match(name):
        logger.warning("Agent %s: invalid name %r", file_path, name)
        return None

    if name != file_path.stem:
        logger.warning(
            "Agent %s: name %r must match filename stem %r",
            file_path, name, file_path.stem,
        )
        return None

    body = match.group(2).strip()
    if not body:
        logger.warning("Agent %s: empty body (system prompt)", file_path)
        return None

    raw_temp = fm.get("temperature")
    temperature: float | None
    if raw_temp is None:
        temperature = None
    else:
        try:
            temperature = float(raw_temp)
        except (TypeError, ValueError):
            logger.warning("Agent %s: invalid temperature %r, ignoring", file_path, raw_temp)
            temperature = None

    return AgentDefinition(
        name=name,
        description=description,
        system_prompt=body,
        path=file_path,
        provider=str(fm.get("provider", "")).strip(),
        model=str(fm.get("model", "")).strip(),
        temperature=temperature,
    )


def discover_agents(agents_dirs: list[str | Path]) -> list[AgentDefinition]:
    """Scan agents directories and return valid agent definitions.

    Directories are searched in order; later dirs override earlier ones on
    name conflict (same precedence rule as ``discover_skills``).
    """
    found: dict[str, AgentDefinition] = {}
    for raw_dir in agents_dirs:
        base = Path(raw_dir).expanduser().resolve()
        if not base.exists():
            continue
        for candidate in sorted(base.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() != ".md":
                continue
            agent = load_agent(candidate)
            if agent is not None:
                found[agent.name] = agent
    return list(found.values())
