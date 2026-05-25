"""Tests for skill registry prompt helpers."""

from __future__ import annotations

from pathlib import Path

from eyetor.skills.registry import _script_usage_summary


def test_single_script_usage_summary_strips_script_prefix(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        '''#!/usr/bin/env python3
"""Execute commands.

Usage:
    run.py --cmd "git status"
    run.py --cmd "date"
"""
''',
        encoding="utf-8",
    )

    usage = _script_usage_summary(script, strip_script_prefix=True)

    assert "run.py --cmd" not in usage
    assert '--cmd "git status"' in usage
    assert '--cmd "date"' in usage


def test_multi_script_usage_summary_keeps_script_prefix(tmp_path: Path) -> None:
    script = tmp_path / "gmail.py"
    script.write_text(
        '''#!/usr/bin/env python3
"""Gmail operations.

Usage:
    gmail.py list --max 20
"""
''',
        encoding="utf-8",
    )

    usage = _script_usage_summary(script)

    assert "gmail.py list --max 20" in usage
