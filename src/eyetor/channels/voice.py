"""Voice channel — always-listening local assistant (wake word → STT → agent → TTS).

Pipeline, all on this host:

1. ``parec`` streams 16 kHz mono s16le from the configured PipeWire source in
   80 ms chunks (1280 samples, the frame size openWakeWord expects).
2. openWakeWord (ONNX, ~1 ms per chunk) scores every chunk; above the
   threshold a short chime plays and the command is recorded.
3. End of command is energy-based: speech is anything above
   ``max(min_speech_rms, 4 × idle noise floor)``; recording stops after
   ``silence_seconds`` of silence, ``max_command_seconds`` in total, or
   ``no_speech_timeout`` without any speech.
4. faster-whisper (``small``, greedy) transcribes; the text goes to a normal
   ChatSession whose system prompt asks for short spoken answers.
5. Piper synthesizes the reply and ``pacat`` plays it. The mic is ignored
   while the assistant talks (there is no echo cancellation), then a
   follow-up window listens for another command without the wake word.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from pathlib import Path

import numpy as np

from eyetor.channels.base import BaseChannel
from eyetor.chat.manager import SessionManager
from eyetor.config import VoiceChannelConfig
from eyetor.tools.confirm import current_confirmer, is_yes
from eyetor.tracking.context import current_channel

logger = logging.getLogger(__name__)

RATE = 16000
CHUNK = 1280  # 80 ms at 16 kHz
CHUNK_BYTES = CHUNK * 2
CHUNK_SECONDS = CHUNK / RATE
PREROLL_CHUNKS = 4  # keep ~0.3 s before speech onset so the first syllable survives

# System-prompt suffix for the voice session (scaffolding in English, replies in Spanish).
VOICE_PROMPT_SUFFIX = (
    "# Voice mode\n"
    "This conversation is spoken: the user talks through a microphone and your reply "
    "is read aloud by a text-to-speech voice. Answer in one or two short sentences of "
    "plain Spanish (Spain). No Markdown, lists, emojis, URLs, code or tables. Round "
    "numbers sensibly. The transcription may contain small recognition errors: infer "
    "the most likely intent. For smart-home commands, act first, then confirm briefly. "
    "Never say an action was done unless a tool result confirms it: if no tool fits "
    "or the call failed, say plainly that you could not do it. Don't repeat what the "
    "question already implies (asked which devices are on, name them without adding "
    "'on' to each). Join several items into one natural sentence; if there are many, "
    "summarise (how many, and the main ones) instead of reading them all."
)

# Transcripts that are only whisper's typical silence hallucinations.
_HALLUCINATION_RE = re.compile(
    r"^(gracias( por ver(lo)?)?|subt[ií]tulos .*|amara\.org.*|m[uú]sica|\W*)[.!¡?¿\s]*$",
    re.IGNORECASE,
)


# The wake phrase as whisper tends to transcribe it ("¡Hey, Javi!", "oye Aitor").
# In the follow-up window the user often repeats it before the next command:
# strip it, and if nothing else was said, re-arm instead of asking the agent.
_WAKE_PREFIX_RE = re.compile(
    r"^\W*((h?ey|hei|oye|ok)\W+)?(jarvis|yarvis|harvis|jarbis|javi|aitor|aito|tor)\b[\s,.!:;-]*",
    re.IGNORECASE,
)


# Anything whisper writes for the wake word itself, anywhere in the transcript
# ("Aitor", "¡Ay, tor!", "I tour", "Javi"), used to confirm a detection.
_WAKE_WORD_RE = re.compile(
    r"\b((h?ai|h?ay|ei|i)[\s,.-]*t(o|ó|ou|u)h?rr?|h?aito|jarvis|yarvis|harvis|jarbis|javi)\b",
    re.IGNORECASE,
)
WAKE_VERIFY_CHUNKS = 25  # ~2 s of audio ending at the detection
WAKE_TAIL_CHUNKS = 6  # ~0.5 s before the detection: the model fires late, a command said
# in one breath ("Aitor, enciende…") has already started by then
# Measured on real «Aitor»: logprob ≥ -0.48, no-speech ≤ 0.07; prompted echoes
# on noise have no-speech ≥ 0.58, on look-alike words mostly logprob < -0.55
WAKE_MIN_LOGPROB = -0.55
CANDIDATES_DIR = Path.home() / ".eyetor/voice/candidatos"
WAKE_MAX_NO_SPEECH = 0.3


def _strip_wake_phrase(text: str) -> str:
    return _WAKE_PREFIX_RE.sub("", text, count=1).strip()


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _chime_pcm(rate: int) -> bytes:
    """Two short rising tones with a fade, as raw s16le."""
    tones = []
    for freq in (880.0, 1320.0):
        n = int(rate * 0.09)
        t = np.arange(n) / rate
        env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        tones.append(0.25 * env * np.sin(2 * math.pi * freq * t))
    return (np.concatenate(tones) * 32767).astype(np.int16).tobytes()


class VoiceChannel(BaseChannel):
    """Local wake-word voice assistant bound to a single ChatSession."""

    def __init__(self, session_manager: SessionManager, cfg: VoiceChannelConfig) -> None:
        self._manager = session_manager
        self._cfg = cfg
        self._running = False
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._muted = False
        self._noise_floor = 50.0
        self._oww = None
        self._whisper = None
        self._voice = None
        self._chime = b""
        self._warm_task: asyncio.Task | None = None
        self._wake_tail: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._load_models)
        except Exception as exc:
            logger.error("Voice channel disabled — could not load models: %s", exc)
            return
        self._chime = _chime_pcm(self._voice.config.sample_rate)
        self._task = asyncio.current_task()
        reader = asyncio.create_task(self._read_mic())
        self._start_warm_up()
        logger.info(
            "Voice channel listening (input=%s, wake word=%s, threshold=%.2f)",
            self._cfg.input_device or "default",
            Path(self._cfg.wake_word_model).stem,
            self._cfg.wake_threshold,
        )
        try:
            while self._running:
                await self._wait_for_wake_word()
                await self._conversation()
        except asyncio.CancelledError:
            pass
        finally:
            reader.cancel()
            self._running = False
            await self._kill_mic()

    async def stop(self) -> None:
        self._running = False
        if self._task and self._task is not asyncio.current_task():
            self._task.cancel()  # wakes the loop if it is waiting for audio
        await self._kill_mic()

    async def _kill_mic(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

    def _load_models(self) -> None:
        from openwakeword.model import Model

        from eyetor.audio import stt, tts

        oww_dir = Path(self._cfg.oww_models_dir).expanduser()
        self._oww = Model(
            wakeword_models=[str(Path(self._cfg.wake_word_model).expanduser())],
            inference_framework="onnx",
            melspec_model_path=str(oww_dir / "melspectrogram.onnx"),
            embedding_model_path=str(oww_dir / "embedding_model.onnx"),
        )
        self._whisper = stt.get_whisper_model(self._cfg.stt_model)
        self._voice = tts.load_voice(self._cfg.piper_voice)

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    async def _spawn_mic(self) -> None:
        args = ["parec", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1",
                "--latency-msec=80"]
        if self._cfg.input_device:
            args.append(f"--device={self._cfg.input_device}")
        self._proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )

    async def _read_mic(self) -> None:
        """Feed 80 ms chunks into the queue; respawn parec if the stream dies."""
        while self._running:
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn_mic()
            assert self._proc and self._proc.stdout
            try:
                data = await self._proc.stdout.readexactly(CHUNK_BYTES)
            except asyncio.IncompleteReadError:
                logger.warning("Voice channel: microphone stream ended, reopening in 2 s")
                await self._kill_mic()
                await asyncio.sleep(2)
                continue
            if self._muted:
                continue
            chunk = np.frombuffer(data, dtype=np.int16)
            if self._queue.full():
                self._queue.get_nowait()  # drop the oldest rather than lag behind
            self._queue.put_nowait(chunk)

    def _flush(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def _speech_pending(self) -> bool:
        """True if the queue already holds speech (at least ~0.25 s of it)."""
        chunks = []
        while not self._queue.empty():
            chunks.append(self._queue.get_nowait())
        for chunk in chunks:
            self._queue.put_nowait(chunk)
        threshold = self._speech_threshold()
        return sum(_rms(c) >= threshold for c in chunks) >= 3

    def _speech_threshold(self) -> float:
        return max(self._cfg.min_speech_rms, 4 * self._noise_floor)

    async def _wait_for_wake_word(self) -> None:
        self._oww.reset()
        name = Path(self._cfg.wake_word_model).stem
        # Diagnostics: after each burst of sound, log its peak level and best
        # wake score, so misses can be told apart from a silent/quiet mic.
        burst_rms = burst_score = 0.0
        quiet_chunks = 0
        recent: list[np.ndarray] = []
        while self._running:
            chunk = await self._queue.get()
            recent = (recent + [chunk])[-WAKE_VERIFY_CHUNKS:]
            level = _rms(chunk)
            if level < 3 * self._noise_floor:  # track the floor, ignore loud bursts
                self._noise_floor = 0.98 * self._noise_floor + 0.02 * max(level, 1.0)
            if self._cfg.wake_dither:
                noisy = chunk + np.random.normal(0.0, self._cfg.wake_dither, chunk.size)
                chunk = np.clip(noisy, -32768, 32767).astype(np.int16)
            score = float(self._oww.predict(chunk).get(name, 0.0))
            if score >= self._cfg.wake_threshold:
                if self._cfg.wake_verify and not await self._confirm_wake(recent, score):
                    self._oww.reset()  # don't re-fire on the same audio
                    recent = []
                    continue
                logger.info("Voice: wake word detected (score %.2f, level %.0f)", score, level)
                self._wake_tail = recent[-WAKE_TAIL_CHUNKS:]
                return
            if level >= max(60.0, 3 * self._noise_floor):  # lower than speech on purpose
                burst_rms = max(burst_rms, level)
                burst_score = max(burst_score, score)
                quiet_chunks = 0
            elif burst_rms:
                burst_score = max(burst_score, score)  # the score peaks just after speech
                quiet_chunks += 1
                if quiet_chunks >= 12:  # ~1 s of quiet closes the burst
                    logger.info(
                        "Voice: sound burst without wake word (peak level %.0f, "
                        "speech threshold %.0f, best score %.2f)",
                        burst_rms, self._speech_threshold(), burst_score,
                    )
                    burst_rms = burst_score = 0.0
                    quiet_chunks = 0

    async def _confirm_wake(self, chunks: list[np.ndarray], score: float) -> bool:
        """Second opinion from whisper on the audio that triggered the model.

        Whisper's VAD clips such a short word and, unprompted, it mishears it
        ("Adiós", "¡Tad!"), so VAD is off and the wake word goes in as the
        prompt. The prompt makes it echo "Aitor" on noise or look-alikes too,
        but then with low confidence or a high no-speech probability.
        """
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        t0 = time.monotonic()

        def run() -> tuple[str, float, float]:
            segments = list(self._whisper.transcribe(
                audio, language=self._cfg.language, beam_size=1, vad_filter=False,
                condition_on_previous_text=False, initial_prompt=self._cfg.wake_prompt,
            )[0])
            if not segments:
                return "", -9.0, 1.0
            text = " ".join(s.text for s in segments).strip()
            return (text, min(s.avg_logprob for s in segments),
                    max(s.no_speech_prob for s in segments))

        text, logprob, no_speech = await asyncio.get_running_loop().run_in_executor(None, run)
        ok = (bool(_WAKE_WORD_RE.search(text)) and logprob >= WAKE_MIN_LOGPROB
              and no_speech <= WAKE_MAX_NO_SPEECH)
        logger.info("Voice: wake candidate (score %.2f) heard as %r (logprob %.2f, "
                    "no-speech %.2f) → %s (%.2fs)", score, text, logprob, no_speech,
                    "ok" if ok else "rejected", time.monotonic() - t0)
        self._keep_candidate(chunks, ok)
        return ok

    @staticmethod
    def _keep_candidate(chunks: list[np.ndarray], ok: bool) -> None:
        """Save the audio of each wake candidate: real data to retrain the model with."""
        import wave

        out = CANDIDATES_DIR / ("ok" if ok else "rejected")
        try:
            out.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(np.concatenate(chunks).astype(np.int16).tobytes())
        except OSError as exc:
            logger.debug("Voice: could not save wake candidate: %s", exc)

    async def _record_command(
        self, start_timeout: float, lead: list[np.ndarray] | None = None
    ) -> np.ndarray | None:
        """Record until trailing silence; None if nobody spoke in ``start_timeout``.

        ``lead`` is audio known to precede the recording (prepended once speech starts).
        """
        threshold = self._speech_threshold()
        preroll: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        waited = silence = 0.0
        while self._running:
            chunk = await self._queue.get()
            loud = _rms(chunk) >= threshold
            if not frames:
                if loud:
                    frames = (lead or []) + preroll + [chunk]
                    continue
                preroll = (preroll + [chunk])[-PREROLL_CHUNKS:]
                waited += CHUNK_SECONDS
                if waited >= start_timeout:
                    return None
                continue
            frames.append(chunk)
            silence = 0.0 if loud else silence + CHUNK_SECONDS
            if silence >= self._cfg.silence_seconds:
                break
            if len(frames) * CHUNK_SECONDS >= self._cfg.max_command_seconds:
                break
        if not frames:
            return None
        return np.concatenate(frames).astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def _start_warm_up(self) -> None:
        """Refill the LLM prompt cache in the background.

        The local server has a single slot shared with every other channel,
        so any Telegram or scheduler turn evicts the voice prefix. Warming up
        on the wake word overlaps that prefill with the user speaking.
        """
        if self._warm_task and not self._warm_task.done():
            return
        self._warm_task = asyncio.create_task(self._warm_up())

    async def _warm_up(self) -> None:
        t0 = time.monotonic()
        try:
            await self._manager.get_or_create(self._cfg.session_id).warm_up()
        except Exception as exc:
            logger.warning("Voice: prompt cache warm-up failed: %s", exc)
            return
        logger.info("Voice: prompt cache warmed in %.1fs", time.monotonic() - t0)

    async def _conversation(self) -> None:
        self._start_warm_up()
        # "Aitor, enciende la luz" in one breath: the command is already queued
        # (captured while whisper verified the wake word), and the chime would
        # mute and flush it.
        lead = self._wake_tail if self._speech_pending() else None
        if self._cfg.chime and lead is None:
            await self._play(self._chime)
        timeout = self._cfg.no_speech_timeout
        while self._running:
            audio = await self._record_command(timeout, lead)
            lead = None
            if audio is None:
                logger.info("Voice: no speech, back to wake word")
                return
            from eyetor.audio import stt

            t0 = time.monotonic()
            text = await stt.transcribe(
                self._whisper,
                audio,
                language=self._cfg.language,
                beam_size=self._cfg.stt_beam_size,
                condition_on_previous_text=False,
            )
            t_stt = time.monotonic() - t0
            if not text or _HALLUCINATION_RE.match(text):
                logger.info("Voice: empty transcript (%.1fs audio)", len(audio) / RATE)
                return
            logger.info("Voice: heard %r (%.1fs audio, stt %.2fs)", text, len(audio) / RATE, t_stt)
            command = _strip_wake_phrase(text)
            if not command or _HALLUCINATION_RE.match(command):
                logger.info("Voice: only the wake phrase, listening for the command")
                if self._cfg.chime:
                    await self._play(self._chime)
                timeout = self._cfg.no_speech_timeout
                continue
            text = command

            t1 = time.monotonic()
            reply = await self._ask(text)
            t_llm = time.monotonic() - t1
            t2 = time.monotonic()
            await self._say(reply)
            logger.info(
                "Voice: replied in %.2fs (stt %.2fs, agent %.2fs, tts+play %.2fs)",
                time.monotonic() - t0, t_stt, t_llm, time.monotonic() - t2,
            )
            timeout = self._cfg.followup_seconds

    async def _ask(self, text: str) -> str:
        current_channel.set("voice")
        current_confirmer.set(self._confirm_by_voice)
        session = self._manager.get_or_create(self._cfg.session_id)
        try:
            reply = await session.send_sync(text)
        except Exception as exc:
            logger.error("Voice: agent error: %s", exc)
            return "Lo siento, ha habido un error al procesar la orden."
        return reply.strip() or "No he podido generar una respuesta."

    async def _confirm_by_voice(self, description: str) -> bool:
        """Ask out loud and listen for a yes. Silence or anything unclear is a no."""
        from eyetor.audio import stt

        await self._say(f"Antes necesito tu confirmación para esto: {description}. ¿Lo hago?")
        audio = await self._record_command(self._cfg.followup_seconds)
        if audio is None:
            logger.info("Voice: no answer to the confirmation, cancelling")
            await self._say("Sin respuesta, lo cancelo.")
            return False
        answer = await stt.transcribe(
            self._whisper,
            audio,
            language=self._cfg.language,
            beam_size=self._cfg.stt_beam_size,
            condition_on_previous_text=False,
        )
        confirmed = is_yes(_strip_wake_phrase(answer or ""))
        logger.info("Voice: confirmation answer %r → %s", answer, confirmed)
        return confirmed

    async def _say(self, text: str) -> None:
        from eyetor.audio import tts

        speech = tts.clean_for_speech(text)
        if not speech:
            return
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, tts.synthesize, self._voice, speech)
        await self._play(pcm)

    async def _play(self, pcm: bytes) -> None:
        """Play audio with the mic muted, then drop anything captured meanwhile."""
        from eyetor.audio import tts

        self._muted = True
        try:
            await tts.play_pcm(pcm, self._voice.config.sample_rate, self._cfg.output_device)
            await asyncio.sleep(0.15)  # let the room echo die out
        finally:
            self._flush()
            self._muted = False
