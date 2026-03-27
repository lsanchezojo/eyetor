"""BaseChannel — abstract interface for all communication channels."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Abstract base for communication channels (CLI, Telegram, etc.)."""

    @abstractmethod
    async def start(self) -> None:
        """Start the channel's event loop (blocking until stopped)."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the channel."""
