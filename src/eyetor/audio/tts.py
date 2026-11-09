"""Local text-to-speech with Piper, played through PipeWire (``pacat``).

Synthesis is fast on this CPU (~0.15 s for a 4 s sentence), so the whole
reply is synthesized before playback starts.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

_voices: dict[str, object] = {}
_lock = threading.Lock()


def load_voice(model_path: str):
    """Return a cached ``PiperVoice``; raises ImportError without piper-tts."""
    from piper import PiperVoice

    path = str(Path(model_path).expanduser())
    with _lock:
        if path not in _voices:
            _voices[path] = PiperVoice.load(path)
        return _voices[path]


def synthesize(voice, text: str) -> bytes:
    """Synthesize ``text`` to raw s16le mono PCM at the voice's sample rate."""
    return b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))


_MD_RE = re.compile(r"(\*\*|__|\*|`+|^#+\s*)", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")
_BULLET_RE = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0000fe0f\U0000200d]"
)
_TRAILING_ZERO_RE = re.compile(r"\b(\d+)[.,]0\b")
_DECIMAL_RE = re.compile(r"\b(\d+)[.,](\d+)\b")


def clean_for_speech(text: str) -> str:
    """Strip Markdown, emojis and URLs, and spell units so Piper reads them well."""
    text = _URL_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _MD_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = text.replace("°C", " grados").replace("ºC", " grados").replace("°", " grados")
    text = text.replace("%", " por ciento")
    text = _TRAILING_ZERO_RE.sub(r"\1", text)
    text = _DECIMAL_RE.sub(r"\1 coma \2", text)
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return re.sub(r"\s{2,}", " ", text).strip()


async def play_pcm(pcm: bytes, rate: int, device: str | None = None) -> None:
    """Play raw s16le mono PCM through PipeWire/PulseAudio and wait until done."""
    args = ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={rate}", "--channels=1"]
    if device:
        args.append(f"--device={device}")
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    assert proc.stdin is not None
    proc.stdin.write(pcm)
    await proc.stdin.drain()
    proc.stdin.close()
    await proc.wait()
