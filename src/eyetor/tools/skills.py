"""One tool per discovered skill (``skill_<name>``, single ``args`` string)."""

from __future__ import annotations

import json

from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.skills.executor import run_script
from eyetor.skills.registry import SkillRegistry
from eyetor.tools.host import (
    _ask_confirm,
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
    """Expose every discovered skill as a tool, with host-safety guards."""
    skill_names = skill_reg.list_names()
    if not skill_names:
        return
    from eyetor.skills.router import RoutingError, ScriptRouter

    # Register one tool per skill — always just an "args" param.
    # ScriptRouter handles mapping the first token to the correct
    # script for multi-script skills; single-script skills pass
    # everything through unchanged.
    for _sn in skill_names:
        _meta = skill_reg.get_metadata(_sn)
        _scripts = skill_reg.list_scripts(_sn)
        if not _scripts:
            continue

        _router = ScriptRouter(_sn, _scripts)

        def _make_handler(skill_name: str, router: ScriptRouter, meta):
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
                    confirmed = await _ask_confirm(skill_name, script_path.name, routed_args)
                    if not confirmed:
                        return json.dumps({"error": "Operation cancelled by user."})
                timeout = _skill_execution_timeout(skill_name, arg_list, meta.timeout)
                from eyetor.providers.tracking import (
                    current_conversation_id,
                    current_session_id,
                )
                _sid = current_session_id.get()
                _env = None
                if _sid and _sid != "unknown":
                    _env = {"EYETOR_SESSION_ID": _sid}
                    _conv = current_conversation_id.get()
                    if _conv:
                        _env["EYETOR_CONVERSATION_ID"] = _conv
                return await run_script(script_path, arg_list, timeout=timeout, env=_env)
            return _handler

        _handler = _make_handler(_sn, _router, _meta)

        _desc = _meta.description[:200]
        _public = _router.public_scripts
        if len(_public) > 1:
            _desc += f" Scripts: {', '.join(s.stem for s in _public)}."

        tool_registry.register(
            ToolDefinition(
                name=f"skill_{_sn.replace('-', '_')}",
                description=_desc,
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
        )

        if _sn == "shopping":
            receipt_script = next(
                (s for s in _scripts if s.name == "receipt.py"), None
            )
            if receipt_script is not None:

                async def shopping_receipt_add_handler(
                    store: str,
                    date: str,
                    items: list[dict],
                    total: float | None = None,
                    image_path: str | None = None,
                    _receipt_script=receipt_script,
                    _meta_timeout=_meta.timeout,
                ) -> str:
                    args = [
                        "add",
                        "--store",
                        store,
                        "--date",
                        date,
                        "--items",
                        json.dumps(items, ensure_ascii=False),
                    ]
                    if total is not None:
                        args.extend(["--total", str(total)])
                    if image_path:
                        args.extend(["--image-path", image_path])
                    return await run_script(
                        _receipt_script,
                        args,
                        timeout=_skill_execution_timeout(
                            "shopping", args, _meta_timeout
                        ),
                    )

                tool_registry.register(
                    ToolDefinition(
                        name="shopping_receipt_add",
                        description=(
                            "Register a shopping receipt from structured data. "
                            "Prefer this over skill_shopping receipt.py add for "
                            "photo receipts or long item lists."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "store": {
                                    "type": "string",
                                    "description": "Store name from the receipt.",
                                },
                                "date": {
                                    "type": "string",
                                    "description": "Purchase date in YYYY-MM-DD format.",
                                },
                                "items": {
                                    "type": "array",
                                    "description": (
                                        "One entry per purchased unit. Do not include "
                                        "items with unknown prices."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "price": {"type": "number"},
                                        },
                                        "required": ["name", "price"],
                                    },
                                },
                                "total": {
                                    "type": "number",
                                    "description": "Receipt total if visible.",
                                },
                                "image_path": {
                                    "type": "string",
                                    "description": (
                                        "Absolute path under ~/.eyetor/images for the "
                                        "saved receipt image."
                                    ),
                                },
                            },
                            "required": ["store", "date", "items"],
                        },
                        handler=shopping_receipt_add_handler,
                    )
                )
