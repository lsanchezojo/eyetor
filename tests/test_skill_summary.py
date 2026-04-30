"""Skill prompt summary tests."""

from __future__ import annotations

from pathlib import Path

from eyetor.skills.registry import SkillRegistry


def test_summary_context_omits_full_skill_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.py").write_text("", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Short description\n"
        "---\n"
        "SECRET FULL BODY INSTRUCTIONS\n",
        encoding="utf-8",
    )

    reg = SkillRegistry()
    reg.discover([tmp_path])
    context = reg.build_skills_summary_context(["demo-skill"])

    assert "Short description" in context
    assert "run" in context
    assert "SECRET FULL BODY" not in context

