"""Voice channel: end-of-command detection and speech text cleanup."""

from __future__ import annotations

import asyncio

import numpy as np

from eyetor.audio.tts import clean_for_speech
from eyetor.channels.voice import CHUNK, VoiceChannel, _HALLUCINATION_RE
from eyetor.config import VoiceChannelConfig


def _chunk(level: float) -> np.ndarray:
    return np.full(CHUNK, level, dtype=np.int16)


def _channel(**cfg) -> VoiceChannel:
    ch = VoiceChannel(session_manager=None, cfg=VoiceChannelConfig(**cfg))
    ch._running = True
    return ch


def _record(ch: VoiceChannel, chunks, timeout: float = 4.0):
    async def go():
        for c in chunks:
            ch._queue.put_nowait(c)
        return await ch._record_command(timeout)

    return asyncio.run(go())


def test_records_until_trailing_silence():
    ch = _channel(silence_seconds=0.8)
    # 1 s silence, 1 s speech, 2 s silence (80 ms chunks)
    audio = _record(ch, [_chunk(10)] * 12 + [_chunk(2000)] * 12 + [_chunk(10)] * 25)
    assert audio is not None
    seconds = len(audio) / 16000
    # preroll (0.32 s) + speech (0.96 s) + trailing silence (0.8 s)
    assert 1.9 < seconds < 2.2


def test_no_speech_times_out():
    ch = _channel()
    assert _record(ch, [_chunk(10)] * 60, timeout=2.0) is None


def test_max_command_length():
    ch = _channel(max_command_seconds=2.0)
    audio = _record(ch, [_chunk(2000)] * 100)
    assert abs(len(audio) / 16000 - 2.0) < 0.1


def test_threshold_follows_noise_floor():
    ch = _channel(min_speech_rms=300)
    ch._noise_floor = 200  # noisy room → speech must exceed 800
    assert _record(ch, [_chunk(500)] * 60, timeout=2.0) is None


def test_clean_for_speech():
    text = "- **Salón:** 24.0 °C y 46.4% 🌡️\n- Terraza: 25.5 °C. Ver https://x.y"
    assert clean_for_speech(text) == (
        "Salón: 24 grados y 46 coma 4 por ciento. Terraza: 25 coma 5 grados. Ver"
    )


def test_hallucination_filter():
    assert _HALLUCINATION_RE.match("Gracias.")
    assert _HALLUCINATION_RE.match("Subtítulos realizados por la comunidad de Amara.org")
    assert not _HALLUCINATION_RE.match("Apaga la luz del salón.")


def test_strip_wake_phrase():
    from eyetor.channels.voice import _strip_wake_phrase

    assert _strip_wake_phrase("¡Hey, Javi!") == ""
    assert _strip_wake_phrase("Hey Jarvis, enciende el flexo.") == "enciende el flexo."
    assert _strip_wake_phrase("Oye Aitor, ¿qué hora es?") == "¿qué hora es?"
    assert _strip_wake_phrase("Enciende el flexo.") == "Enciende el flexo."
    assert _strip_wake_phrase("Javier ha llamado") == "Javier ha llamado"
