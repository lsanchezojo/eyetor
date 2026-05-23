"""filesystem skill — relative paths resolve against the base dir, not CWD."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FS = Path(__file__).resolve().parents[1] / "skills" / "filesystem" / "scripts" / "fs.py"


def _run(args: list[str], base: str | None) -> dict:
    env = dict(os.environ)
    if base is not None:
        env["EYETOR_FS_BASE"] = base
    # cwd intentionally != base to prove resolution is NOT cwd-relative
    proc = subprocess.run(
        [sys.executable, str(FS), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd="/tmp",
    )
    return json.loads(proc.stdout)


def test_dot_resolves_to_base(tmp_path: Path):
    (tmp_path / "marker.txt").write_text("x")
    d = _run(["list", "--path", "."], base=str(tmp_path))
    assert d["ok"] is True
    assert Path(d["path"]) == tmp_path.resolve()
    assert any(e["name"] == "marker.txt" for e in d["entries"])


def test_relative_subdir_resolves_under_base(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    d = _run(["info", "--path", "sub"], base=str(tmp_path))
    assert d["ok"] is True
    assert Path(d["path"]) == (tmp_path / "sub").resolve()


def test_absolute_path_is_kept(tmp_path: Path):
    f = tmp_path / "abs.txt"
    f.write_text("hi")
    d = _run(["info", "--path", str(f)], base="/some/other/base")
    assert d["ok"] is True
    assert Path(d["path"]) == f.resolve()
