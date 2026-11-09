"""Agent workflow building blocks.

- ``router``: classify a message to pick an agent (used by SessionManager).
- ``observer``: progress events for long multi-step turns (Telegram).
"""

from eyetor.workflows.router import Router

__all__ = ["Router"]
