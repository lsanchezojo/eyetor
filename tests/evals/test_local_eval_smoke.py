"""Smoke checks for the local eval prompt catalog."""

from __future__ import annotations

from pathlib import Path


def test_eval_script_exists() -> None:
    assert Path("scripts/eval_local.py").exists()

