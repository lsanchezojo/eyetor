"""Tools provided by skills.

Every skill gets a generic ``skill_<name>`` tool (single free-text ``args``
string routed to its scripts), unless its SKILL.md sets ``args_tool: false``.
Skills may also declare typed tools in their frontmatter (``tools:``), which
small models call far more reliably than a hand-built command line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.skills.executor import run_script
from eyetor.skills.loader import SkillMetadata, SkillTool
from eyetor.skills.registry import SkillRegistry
from eyetor.skills.router import RoutingError, ScriptRouter
from eyetor.tools.confirm import guard
from eyetor.tools.host import (
    _is_dangerous,
    _looks_like_package_install_command,
    _skill_execution_timeout,
)


def register_skill_tools(
    tool_registry: ToolRegistry,
    skill_reg: SkillRegistry,
    *,
    use_host_tools: bool,
    host_profile: dict,
) -> None:
    """Expose every discovered skill as tools, with host-safety guards."""
    from eyetor.chat.tool_gating import register_group

    for name in skill_reg.list_names():
        meta = skill_reg.get_metadata(name)
        scripts = skill_reg.list_scripts(name)
        if not scripts:
            continue
        if meta.group and meta.tools:
            register_group(meta.group, meta.triggers)
        if meta.args_tool:
            tool_registry.register(
                _args_tool(
                    name,
                    meta,
                    ScriptRouter(name, scripts),
                    use_host_tools=use_host_tools,
                    host_profile=host_profile,
                )
            )
        for tool in meta.tools:
            tool_registry.register(_typed_tool(name, tool, meta, use_host_tools=use_host_tools))


def _session_env() -> dict[str, str] | None:
    """Env passed to skill scripts so they can scope state to the chat."""
    from eyetor.providers.tracking import current_conversation_id, current_session_id

    sid = current_session_id.get()
    if not sid or sid == "unknown":
        return None
    env = {"EYETOR_SESSION_ID": sid}
    conv = current_conversation_id.get()
    if conv:
        env["EYETOR_CONVERSATION_ID"] = conv
    return env


def _args_tool(
    skill_name: str,
    meta: SkillMetadata,
    router: ScriptRouter,
    *,
    use_host_tools: bool,
    host_profile: dict,
) -> ToolDefinition:
    """``skill_<name>``: one free-text ``args`` param. ScriptRouter maps the
    first token to the right script for multi-script skills; single-script
    skills pass everything through unchanged."""

    async def _handler(args: str = "") -> str:
        try:
            script_path, arg_list = router.route(args)
        except RoutingError as exc:
            return json.dumps({"error": str(exc)})
        routed_args = " ".join(arg_list)
        if (
            use_host_tools
            and skill_name == "shell"
            and host_profile.get("can_install_system_packages")
            and _looks_like_package_install_command(routed_args)
        ):
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Do not install system packages through skill_shell. "
                        "Use the install_package tool with a package name."
                    ),
                    "tool": "install_package",
                    "example": {
                        "package": "megatools",
                        "binary": "megadl",
                    },
                },
                ensure_ascii=False,
            )
        if use_host_tools and _is_dangerous(skill_name, script_path.name, routed_args):
            refused = await guard(f"{skill_name}: {routed_args}")
            if refused:
                return refused
        timeout = _skill_execution_timeout(skill_name, arg_list, meta.timeout)
        return await run_script(script_path, arg_list, timeout=timeout, env=_session_env())

    desc = meta.description[:200]
    public = router.public_scripts
    if len(public) > 1:
        desc += f" Scripts: {', '.join(s.stem for s in public)}."

    return ToolDefinition(
        name=f"skill_{skill_name.replace('-', '_')}",
        description=desc,
        parameters={
            "type": "object",
            "properties": {
                "args": {
                    "type": "string",
                    "description": (
                        "Subcommand and flags only (no script name/interpreter/"
                        "vars). E.g. 'open https://example.com --headed'."
                    ),
                },
            },
            "required": [],
        },
        handler=_handler,
    )


def build_skill_tool_argv(tool: SkillTool, arguments: dict[str, Any]) -> list[str]:
    """Translate typed tool arguments into the script's command line.

    Flags follow the schema's property order and come before the positional
    values; ``None`` values are skipped. ``--flag=value`` keeps values that
    start with ``-`` intact.
    """
    argv = list(tool.argv)
    positional = [
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        for value in (arguments.get(key) for key in tool.positional)
        if value is not None and value != ""
    ]
    props = tool.parameters.get("properties", {})
    for key in list(props) + [k for k in arguments if k not in props]:
        value = arguments.get(key)
        if value is None or key in tool.positional:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, dict)):
            argv.append(f"{flag}={json.dumps(value, ensure_ascii=False)}")
        else:
            argv.append(f"{flag}={value}")
    # Positionals go last; "--" keeps a value like "-50 %" from reading as a flag.
    if any(v.startswith("-") for v in positional):
        argv.append("--")
    return argv + positional


def _describe_args(arguments: dict[str, Any]) -> str:
    """``path=/tmp/x, recursive=True`` — readable in a confirmation prompt."""
    return ", ".join(f"{k}={v}" for k, v in arguments.items() if v is not None)


def _typed_tool(
    skill_name: str, tool: SkillTool, meta: SkillMetadata, *, use_host_tools: bool
) -> ToolDefinition:
    """A typed tool declared in SKILL.md, run as its script's command line."""
    script_path = Path(meta.path) / "scripts" / tool.script

    async def _handler(**arguments: Any) -> str:
        argv = build_skill_tool_argv(tool, arguments)
        if use_host_tools and tool.confirm:
            described = _describe_args(arguments)
            refused = await guard(f"{tool.name}: {described}" if described else tool.name)
            if refused:
                return refused
        return await run_script(
            script_path,
            argv,
            timeout=_skill_execution_timeout(skill_name, argv, meta.timeout),
            env=_session_env(),
        )

    return ToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        handler=_handler,
        skill=skill_name,
        group=meta.group,
    )
