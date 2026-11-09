"""Typed tools declared by skills in SKILL.md frontmatter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from eyetor.access import Access
from eyetor.models.tools import ToolDefinition, ToolRegistry
from eyetor.skills.loader import SkillTool, load_skill_metadata
from eyetor.skills.registry import SkillRegistry
from eyetor.tools.skills import build_skill_tool_argv, register_skill_tools

REPO_SKILLS = Path(__file__).resolve().parents[1] / "skills"


def _write_skill(base: Path, frontmatter: str) -> Path:
    d = base / "demo"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "run.py").write_text("print('{}')\n")
    (d / "SKILL.md").write_text(f"---\nname: demo\ndescription: Demo skill.\n{frontmatter}---\n\nBody\n")
    return d


def test_loader_parses_typed_tools_and_args_tool(tmp_path: Path):
    d = _write_skill(
        tmp_path,
        "args_tool: false\n"
        "tools:\n"
        "  - name: demo_go\n"
        "    description: Go.\n"
        "    script: run.py\n"
        "    argv: [go]\n"
        "    confirm: true\n"
        "    parameters: {type: object, properties: {x: {type: string}}}\n"
        "  - name: Bad-Name\n"
        "    description: Invalid name, skipped.\n"
        "    script: run.py\n"
        "  - name: demo_missing\n"
        "    description: Script missing, skipped.\n"
        "    script: nope.py\n",
    )
    meta = load_skill_metadata(d)
    assert meta.args_tool is False
    assert [t.name for t in meta.tools] == ["demo_go"]
    assert meta.tools[0].argv == ["go"] and meta.tools[0].confirm is True


def test_argv_follows_schema_order_and_value_types():
    tool = SkillTool(
        name="t",
        description="d",
        script="s.py",
        argv=["add"],
        parameters={
            "type": "object",
            "properties": {
                "store": {}, "items": {}, "total": {}, "image_path": {}, "recursive": {}, "dry": {},
            },
        },
    )
    argv = build_skill_tool_argv(
        tool,
        {
            "image_path": None,
            "recursive": True,
            "dry": False,
            "total": 1.5,
            "items": [{"name": "pan", "price": 1}],
            "store": "-Día",
        },
    )
    assert argv == [
        "add",
        "--store=-Día",
        '--items=[{"name": "pan", "price": 1}]',
        "--total=1.5",
        "--recursive",
    ]


def _registry(use_host_tools: bool = True) -> ToolRegistry:
    skill_reg = SkillRegistry()
    skill_reg.discover([REPO_SKILLS])
    reg = ToolRegistry()
    register_skill_tools(reg, skill_reg, use_host_tools=use_host_tools, host_profile={})
    return reg


def test_filesystem_exposes_only_typed_tools():
    names = set(_registry()._tools)
    assert "skill_filesystem" not in names
    assert {"fs_read", "fs_write", "fs_grep", "fs_delete"} <= names
    assert _registry()._tools["fs_read"].skill == "filesystem"
    # Shopping keeps its generic tool and adds the typed one.
    assert {"skill_shopping", "shopping_receipt_add"} <= names


def test_fs_tools_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EYETOR_FS_BASE", str(tmp_path))
    tools = _registry()._tools
    out = json.loads(asyncio.run(tools["fs_write"].handler(path="n.txt", content="-linea\ndos")))
    assert out["ok"] is True
    out = json.loads(asyncio.run(tools["fs_read"].handler(path="n.txt", lines="1")))
    assert out["content"] == "-linea"


def test_access_allows_skill_tool_via_skill_or_name():
    tool = ToolDefinition(
        name="shopping_receipt_add", description="", parameters={}, skill="shopping"
    )
    assert Access(skills={"shopping"}).allows(tool)
    assert Access(tools={"shopping_receipt_add"}).allows(tool)
    assert not Access(skills={"web_search"}).allows(tool)
    core = ToolDefinition(name="kb_search", description="", parameters={})
    assert not Access(skills={"shopping"}).allows(core)
