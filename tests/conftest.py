from __future__ import annotations

import pytest

from eyetor.chat import tool_gating


@pytest.fixture(autouse=True)
def _isolate_skill_gating_groups():
    """Skill groups are registered process-wide; keep each test's own."""
    saved = dict(tool_gating._SKILL_GROUP_PATTERNS)
    yield
    tool_gating._SKILL_GROUP_PATTERNS.clear()
    tool_gating._SKILL_GROUP_PATTERNS.update(saved)
