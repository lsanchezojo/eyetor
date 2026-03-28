"""Telegram channel — bot with streaming progressive messages.

Requires the 'telegram' optional dependency:
    pip install eyetor[telegram]

Voice message transcription requires faster-whisper:
    pip install faster-whisper
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import os
from pathlib import Path

from eyetor.channels.base import BaseChannel
from eyetor.chat.manager import SessionManager
from eyetor.config import TelegramChannelConfig

logger = logging.getLogger(__name__)

_CHUNK_TOKENS = 20  # Edit message every N characters


class TelegramChannel(BaseChannel):
    """Telegram bot channel using aiogram.

    Features:
    - Streaming-progressive responses (edits message as tokens arrive)
    - Per-chat session management
    - Optional user authentication by chat_id or username
    """

    def __init__(
        self,
        session_manager: SessionManager,
        config: TelegramChannelConfig,
        skill_reg=None,
        scheduler=None,
    ) -> None:
        self._manager = session_manager
        self._config = config
        self._skill_reg = skill_reg
        self._scheduler = scheduler
        self._dp = None
        self._bot = None

    async def start(self) -> None:
        try:
            from aiogram import Bot, Dispatcher, F
            from aiogram.filters import Command
            from aiogram.types import Message, BotCommand
        except ImportError:
            raise ImportError(
                "aiogram is required for the Telegram channel. "
                "Install it with: pip install eyetor[telegram]"
            )

        bot_token = self._config.bot_token
        if not bot_token:
            raise ValueError("Telegram bot_token is not configured. Set TELEGRAM_BOT_TOKEN env var.")

        if not self._config.ssl_verify:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

        bot = Bot(token=bot_token)
        dp = Dispatcher()
        self._bot = bot
        self._dp = dp

        auth_config = self._config.auth
        allowed_users = set(str(u) for u in (auth_config.allowed_users or []))

        def _is_authorized(msg: Message) -> bool:
            if not auth_config.enabled:
                return True
            user = msg.from_user
            if not user:
                return False
            return (
                str(user.id) in allowed_users
                or (user.username and user.username in allowed_users)
            )

        @dp.message(Command("start"))
        async def cmd_start(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return
            session_id = f"telegram-{msg.chat.id}"
            self._manager.get_or_create(session_id)
            await msg.answer(
                "Hello! I'm Eyetor, a multi-agent AI assistant.\n"
                "Commands: /reset (new conversation), /help"
            )

        @dp.message(Command("reset"))
        async def cmd_reset(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            session_id = f"telegram-{msg.chat.id}"
            self._manager.reset(session_id)
            await msg.answer("Conversation reset. How can I help you?")

        @dp.message(Command("skills"))
        async def cmd_skills(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            await msg.answer(_format_skills_text(self._skill_reg), parse_mode="HTML")

        @dp.message(Command("tasks"))
        async def cmd_tasks(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            await msg.answer(_format_tasks_text(self._scheduler), parse_mode="HTML")

        @dp.message(Command("help"))
        async def cmd_help(msg: Message) -> None:
            await msg.answer(
                "Eyetor commands:\n"
                "/reset — start a new conversation\n"
                "/skills — list available skills\n"
                "/tasks — list scheduled tasks\n"
                "/help — show this help\n\n"
                "Send a message to chat, a voice note to transcribe, "
                "or a receipt photo (with store name as caption) to ingest prices."
            )

        @dp.message(F.text)
        async def on_message(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return

            session_id = f"telegram-{msg.chat.id}"
            session = self._manager.get_or_create(session_id)

            # Send placeholder and stream response progressively
            placeholder = await msg.answer("...")
            buffer = ""
            last_edit = ""
            try:
                async for chunk in session.send(msg.text or ""):
                    buffer += chunk
                    if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                        try:
                            await placeholder.edit_text(buffer or "...")
                            last_edit = buffer
                        except Exception:
                            pass  # Ignore edit conflicts

                # Final edit always applies HTML formatting
                if buffer:
                    try:
                        await placeholder.edit_text(_md_to_html(buffer), parse_mode="HTML")
                    except Exception:
                        await msg.answer(_md_to_html(buffer), parse_mode="HTML")
            except Exception as exc:
                logger.error("Telegram message handler error: %s", exc)
                await placeholder.edit_text(f"Error: {exc}")

        @dp.message(F.photo)
        async def on_photo(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return

            photo = msg.photo[-1]  # last = largest
            if photo.file_size and photo.file_size > 25 * 1024 * 1024:
                await msg.answer("Photo too large (max 25 MB).")
                return

            caption = msg.caption or ""
            try:
                # Save to a persistent path so it survives follow-up messages
                receipts_dir = Path.home() / ".eyetor" / "receipts"
                receipts_dir.mkdir(parents=True, exist_ok=True)
                import time as _time
                img_path = receipts_dir / f"{msg.chat.id}_{int(_time.time())}.jpg"

                tg_file = await bot.get_file(photo.file_id)
                await bot.download_file(tg_file.file_path, destination=str(img_path))

                store_hint = (
                    f' La tienda es "{caption.strip()}".' if caption.strip() else ""
                )
                prompt = (
                    f"Te mando una foto de un ticket de supermercado.{store_hint} "
                    f"Ingéstalo con la skill grocery-intel (usa receipt.py ingest). "
                    f"La imagen está en: {img_path}"
                )
                session_id = f"telegram-{msg.chat.id}"
                session = self._manager.get_or_create(session_id)

                placeholder = await msg.answer("🧾 Procesando ticket...")
                buffer = ""
                last_edit = ""
                try:
                    async for chunk in session.send(prompt):
                        buffer += chunk
                        if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                            try:
                                await placeholder.edit_text(buffer or "...")
                                last_edit = buffer
                            except Exception:
                                pass
                    if buffer:
                        try:
                            await placeholder.edit_text(
                                _md_to_html(buffer), parse_mode="HTML"
                            )
                        except Exception:
                            await msg.answer(_md_to_html(buffer), parse_mode="HTML")
                except Exception as exc:
                    logger.error("Telegram photo handler error: %s", exc)
                    await placeholder.edit_text(f"Error: {exc}")
            except Exception as exc:
                logger.error("Photo download error: %s", exc)
                await msg.answer(f"No se pudo procesar la foto: {exc}")

        @dp.message(F.voice | F.audio)
        async def on_voice(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return

            # Transcribe audio with faster-whisper
            transcription = await _transcribe_voice(bot, msg)
            if transcription is None:
                return  # error already sent to user

            session_id = f"telegram-{msg.chat.id}"
            session = self._manager.get_or_create(session_id)

            placeholder = await msg.answer(f"🎤 <i>{_escape_html(transcription)}</i>\n\n...", parse_mode="HTML")
            buffer = ""
            last_edit = ""
            try:
                async for chunk in session.send(transcription):
                    buffer += chunk
                    if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                        try:
                            await placeholder.edit_text(
                                f"🎤 <i>{_escape_html(transcription)}</i>\n\n{buffer}",
                            )
                            last_edit = buffer
                        except Exception:
                            pass

                if buffer:
                    try:
                        await placeholder.edit_text(
                            f"🎤 <i>{_escape_html(transcription)}</i>\n\n{_md_to_html(buffer)}",
                            parse_mode="HTML",
                        )
                    except Exception:
                        await msg.answer(_md_to_html(buffer), parse_mode="HTML")
            except Exception as exc:
                logger.error("Telegram voice handler error: %s", exc)
                await placeholder.edit_text(f"Error: {exc}")

        await bot.set_my_commands([
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="reset", description="Start a new conversation"),
            BotCommand(command="skills", description="List available skills"),
            BotCommand(command="tasks", description="List scheduled tasks"),
            BotCommand(command="help", description="Show help"),
        ])

        logger.info("Starting Telegram bot...")
        await dp.start_polling(bot)

    async def stop(self) -> None:
        if self._dp:
            await self._dp.stop_polling()
        if self._bot:
            await self._bot.session.close()


_whisper_model = None  # Module-level cache — loaded once on first use


async def _transcribe_voice(bot, msg) -> str | None:
    """Download voice/audio and transcribe it.

    Priority:
    1. OpenAI-compatible /v1/audio/transcriptions API (WHISPER_BASE_URL or OPENAI_API_KEY)
    2. Local faster-whisper (if installed)

    Returns the transcription string, or None if an error occurred
    (error message already sent to the user).
    """
    file_obj = msg.voice or msg.audio
    if file_obj is None:
        await msg.answer("Could not read the audio file.")
        return None

    # Size guard: 25 MB max
    if getattr(file_obj, "file_size", None) and file_obj.file_size > 25 * 1024 * 1024:
        await msg.answer("Audio file too large (max 25 MB).")
        return None

    tmp_path = None
    try:
        tg_file = await bot.get_file(file_obj.file_id)
        suffix = ".ogg" if msg.voice else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        await bot.download_file(tg_file.file_path, destination=tmp_path)

        whisper_url = os.environ.get("WHISPER_BASE_URL", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if whisper_url or openai_key:
            return await _transcribe_via_api(tmp_path, whisper_url or None, openai_key or None, suffix)

        return await _transcribe_local(msg, tmp_path)

    except Exception as exc:
        logger.error("Voice transcription error: %s", exc)
        await msg.answer(f"Error transcribing audio: {exc}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


async def _transcribe_via_api(path: str, base_url: str | None, api_key: str | None, suffix: str) -> str:
    """Transcribe using an OpenAI-compatible /v1/audio/transcriptions endpoint."""
    import httpx
    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions" if base_url else "https://api.openai.com/v1/audio/transcriptions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    mime = "audio/ogg" if suffix == ".ogg" else "audio/mpeg"
    async with httpx.AsyncClient(timeout=60) as client:
        with open(path, "rb") as f:
            r = await client.post(
                url,
                headers=headers,
                files={"file": (os.path.basename(path), f, mime)},
                data={"model": "whisper-1"},
            )
        r.raise_for_status()
        return r.json()["text"].strip()


async def _transcribe_local(msg, path: str) -> str | None:
    """Transcribe using local faster-whisper, with module-level model cache."""
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        await msg.answer(
            "Voice transcription is not configured. Options:\n"
            "• Set <code>WHISPER_BASE_URL</code> to a local Whisper server\n"
            "• Set <code>OPENAI_API_KEY</code> to use OpenAI Whisper API\n"
            "• Install faster-whisper: <code>pip install faster-whisper</code>",
            parse_mode="HTML",
        )
        return None

    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_whisper_model, _whisper_model, path)


def _run_whisper_model(model, audio_path: str) -> str:
    """Run faster-whisper transcription synchronously (called in thread pool)."""
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_to_html(text: str) -> str:
    """Convert Markdown to Telegram HTML (supported subset).

    Handles code blocks first (protected from inline processing), then
    applies inline transforms on the remaining text segments.
    """
    # Split on fenced code blocks to protect their content
    code_block_re = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    segments = []
    last = 0
    for m in code_block_re.finditer(text):
        segments.append(("text", text[last:m.start()]))
        lang = m.group(1).strip()
        code = _escape_html(m.group(2).strip())
        tag = f'<code class="language-{lang}">' if lang else "<code>"
        segments.append(("code", f"<pre>{tag}{code}</code></pre>"))
        last = m.end()
    segments.append(("text", text[last:]))

    parts = []
    for kind, content in segments:
        if kind == "code":
            parts.append(content)
        else:
            parts.append(_inline_md_to_html(content))
    return "".join(parts)


def _inline_md_to_html(text: str) -> str:
    """Apply inline Markdown → HTML transforms on a plain-text segment."""
    # Escape HTML entities before adding any tags
    text = _escape_html(text)

    # Inline code `...`
    text = re.sub(r"`([^`\n]+)`", lambda m: f"<code>{m.group(1)}</code>", text)

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text, flags=re.DOTALL)

    # Italic: *text* or _text_ (single, not preceded/followed by same char)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text, flags=re.DOTALL)

    # Headers: # ## ### → bold on its own line
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "─────", text, flags=re.MULTILINE)

    # Unordered list items: - / * / + at line start → bullet
    text = re.sub(r"^[ \t]*[-*+] ", "• ", text, flags=re.MULTILINE)

    return text


def _format_tasks_text(scheduler) -> str:
    """Return an HTML-formatted list of scheduled tasks."""
    if scheduler is None:
        return "Scheduler is not configured."
    tasks = scheduler.list_tasks()
    if not tasks:
        return "No scheduled tasks."
    lines = ["<b>Scheduled tasks:</b>"]
    for t in tasks:
        status = "✅" if t["enabled"] else "⏸"
        next_run = t["next_run"]
        if next_run:
            # Trim to readable format: 2026-03-28T09:00:00+01:00 → 2026-03-28 09:00
            next_run = next_run[:16].replace("T", " ")
        next_str = f" — next: {next_run}" if next_run else ""
        notify_icon = {"telegram": "💬", "log": "📄", "none": "🔇"}.get(t["notify"], "")
        lines.append(
            f"{status} {notify_icon} <b>{_escape_html(t['name'])}</b>\n"
            f"    <code>{_escape_html(t['schedule'])}</code>{next_str}\n"
            f"    ID: <code>{t['id']}</code>"
        )
    return "\n\n".join(lines)


def _format_skills_text(skill_reg) -> str:
    """Return a plain-text skills list with descriptions for Telegram (HTML)."""
    if skill_reg is None:
        return "No skills configured."
    metadata = skill_reg.all_metadata()
    if not metadata:
        return "No skills configured."
    lines = ["<b>Available skills:</b>"]
    for m in metadata:
        lines.append(f"  <code>{m.name}</code> — {m.description}")
    return "\n".join(lines)
