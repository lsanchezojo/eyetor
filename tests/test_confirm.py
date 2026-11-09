"""Confirmation of dangerous operations through the active channel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from eyetor.models.tools import ToolRegistry
from eyetor.skills.registry import SkillRegistry
from eyetor.tools.confirm import CANCELLED, UNAVAILABLE, current_confirmer, guard, is_yes
from eyetor.tools.skills import register_skill_tools
from eyetor.tracking.context import current_channel

REPO_SKILLS = Path(__file__).resolve().parents[1] / "skills"


def test_is_yes():
    assert all(is_yes(t) for t in ["Sí", "sí, hazlo", "Vale.", "adelante", "ok", "De acuerdo"])
    assert not any(
        is_yes(t) for t in ["No", "no, espera", "sí, pero espera", "quizá", "", "si no hay problema"]
    )


def _run(coro_factory, *, channel: str, confirmer=None):
    async def main():
        current_channel.set(channel)
        current_confirmer.set(confirmer)
        return await coro_factory()

    return asyncio.run(main())


def test_guard_uses_the_channel_confirmer():
    asked: list[str] = []

    async def yes(description: str) -> bool:
        asked.append(description)
        return True

    async def no(description: str) -> bool:
        return False

    assert _run(lambda: guard("borrar /tmp/x"), channel="telegram", confirmer=yes) is None
    assert asked == ["borrar /tmp/x"]
    assert _run(lambda: guard("borrar"), channel="telegram", confirmer=no) == CANCELLED


def test_guard_refuses_where_nobody_can_answer():
    # Scheduled tasks (and a service without a TTY) cannot be confirmed.
    assert _run(lambda: guard("borrar"), channel="scheduler") == UNAVAILABLE


def test_confirmer_errors_count_as_no():
    async def broken(description: str) -> bool:
        raise RuntimeError("telegram down")

    assert _run(lambda: guard("borrar"), channel="telegram", confirmer=broken) == CANCELLED


def test_fs_delete_asks_and_only_deletes_on_yes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EYETOR_FS_BASE", str(tmp_path))
    target = tmp_path / "borrame.txt"
    target.write_text("x")
    skill_reg = SkillRegistry()
    skill_reg.discover([REPO_SKILLS])
    reg = ToolRegistry()
    register_skill_tools(reg, skill_reg, use_host_tools=True, host_profile={})
    delete = reg._tools["fs_delete"].handler
    asked: list[str] = []

    async def answer(value: bool):
        async def confirmer(description: str) -> bool:
            asked.append(description)
            return value

        return confirmer

    no = asyncio.run(answer(False))
    out = _run(lambda: delete(path="borrame.txt"), channel="telegram", confirmer=no)
    assert json.loads(out) == json.loads(CANCELLED) and target.exists()
    assert asked == ["fs_delete: path=borrame.txt"]

    yes = asyncio.run(answer(True))
    out = _run(lambda: delete(path="borrame.txt"), channel="telegram", confirmer=yes)
    assert json.loads(out)["ok"] is True and not target.exists()
