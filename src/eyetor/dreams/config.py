from __future__ import annotations

from pydantic import BaseModel


class DreamsThresholds(BaseModel):
    """Configurable thresholds for dream analysis."""

    critical_error: bool = True
    slow_tool_ms: int = 30000
    max_reasoning_tokens: int = 10000


class DreamConfig(BaseModel):
    """Configuration for the dreams system."""

    enabled: bool = True
    schedule: str = "0 3 * * *"
    max_proposals: int = 3
    days_to_analyze: int = 7
    thresholds: DreamsThresholds = DreamsThresholds()
    output_dir: str = "~/.eyetor/dreams"
    tracking_db_path: str = "~/.eyetor/tracking.db"
    memory_db_path: str = "~/.eyetor/memory.db"