"""Local speech-to-text with faster-whisper, shared by Telegram and voice.

Models are cached per (model, device, compute_type) so the Telegram channel
(``medium``, accurate for long voice notes) and the voice channel (``small``,
~0.9 s per short command on this CPU) can coexist without reloading.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

_models: dict[tuple[str, str, str], Any] = {}
_lock = threading.Lock()


def get_whisper_model(model: str = "medium", device: str = "cpu", compute_type: str = "int8"):
    """Return a cached ``WhisperModel``; raises ImportError without faster-whisper."""
    from faster_whisper import WhisperModel

    key = (model, device, compute_type)
    with _lock:
        if key not in _models:
            _models[key] = WhisperModel(model, device=device, compute_type=compute_type)
        return _models[key]


def transcribe_sync(
    model,
    audio,
    language: str | None = "es",
    beam_size: int = 5,
    vad_filter: bool = True,
    condition_on_previous_text: bool = True,
) -> str:
    """Transcribe a file path or a float32 16 kHz mono numpy array."""
    segments, _ = model.transcribe(
        audio,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def transcribe(model, audio, **kwargs) -> str:
    """Async wrapper running :func:`transcribe_sync` in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: transcribe_sync(model, audio, **kwargs))
