"""Telegram channel — bot with streaming progressive messages.

Requires the 'telegram' optional dependency:
    pip install eyetor[telegram]

Voice message transcription requires faster-whisper:
    pip install faster-whisper
"""

from __future__ import annotations

import asyncio
from datetime import date
import json
import logging
import re
import tempfile
import os
from pathlib import Path

from eyetor.channels.base import BaseChannel
from eyetor.chat.manager import SessionManager
from eyetor.config import TelegramChannelConfig
from eyetor.tracking.context import current_channel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.tracking.usage import UsageTracker

logger = logging.getLogger(__name__)

_CHUNK_TOKENS = 20  # Edit message every N characters
_TG_MAX_LEN = 4096  # Telegram message character limit
_IMAGE_MARKER_RE = re.compile(r"\[IMAGE:(.*?)\]")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:\s+de)?\s+"
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|setiembre|octubre|noviembre|diciembre)"
    r"(?:\s+de)?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _valid_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _expand_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def _extract_caption_date(caption: str) -> str | None:
    """Return the first complete date found in a Telegram image caption."""
    if not caption.strip():
        return None

    if match := _ISO_DATE_RE.search(caption):
        year, month, day = (int(part) for part in match.groups())
        if parsed := _valid_iso_date(year, month, day):
            return parsed

    if match := _NUMERIC_DATE_RE.search(caption):
        day, month, year = (int(part) for part in match.groups())
        if parsed := _valid_iso_date(_expand_year(year), month, day):
            return parsed

    if match := _TEXT_DATE_RE.search(caption):
        day_raw, month_raw, year_raw = match.groups()
        month = _MONTHS_ES[month_raw.lower()]
        if parsed := _valid_iso_date(int(year_raw), month, int(day_raw)):
            return parsed

    return None


def _build_image_prompt(
    *,
    user_text: str,
    description: str,
    img_path: Path,
    caption_date: str | None,
) -> str:
    """Build the channel-generic prompt for forwarding an image to the agent."""
    caption_meta = (
        f"Fecha completa detectada en el caption: {caption_date}.\n\n"
        if caption_date
        else "No hay fecha completa detectada en el caption.\n\n"
    )
    if user_text:
        intro = f"El usuario ha enviado una imagen con este mensaje: «{user_text}»"
        closing = (
            "Responde a lo que pide el usuario. Usa el análisis de la imagen "
            "como contexto, pero céntrate en su petición. Si una herramienta "
            "disponible encaja con la petición, puedes usarla."
        )
    else:
        intro = "El usuario ha enviado una imagen sin mensaje adicional."
        closing = (
            "Responde al usuario basándote en el contenido descrito. Si una "
            "herramienta disponible encaja con el contenido o la tarea inferida, "
            "puedes usarla."
        )
    return (
        f"{intro}\n\n"
        f"{caption_meta}"
        f"Análisis de la imagen (modelo de visión):\n{description}\n\n"
        f"Imagen guardada en: {img_path}\n\n"
        f"{closing}"
    )


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
        tracker: "UsageTracker | None" = None,
        full_config: "VectorConfig | None" = None,
        agent_reg=None,
    ) -> None:
        self._manager = session_manager
        self._config = config
        self._skill_reg = skill_reg
        self._agent_reg = agent_reg
        self._scheduler = scheduler
        self._tracker = tracker
        self._dp = None
        self._bot = None

        self._show_thinking: dict[int, bool] = {}  # per chat_id toggle for /thinking

        # Resolve vision provider chain from config (primary + fallbacks).
        # Empty list → _describe_image falls back to VISION_* env vars.
        self._vision_specs: list[dict] = []
        if full_config and full_config.vision_provider:
            primary = full_config.vision_provider
            names = [primary] + [
                n for n in (full_config.vision_fallback or []) if n != primary
            ]
            for name in names:
                prov_cfg = full_config.providers.get(name)
                if not prov_cfg:
                    logger.warning("Vision provider '%s' not found in providers", name)
                    continue
                # vision_model override only applies to the primary provider
                model = (
                    full_config.vision_model or prov_cfg.model
                    if name == primary
                    else prov_cfg.model
                )
                spec = {
                    "name": name,
                    "base_url": prov_cfg.base_url,
                    "api_key": prov_cfg.api_key or "",
                    "model": model,
                }
                # Local llama.cpp (Gemma-4) razona por defecto; para describir
                # imágenes no hace falta y desperdicia el presupuesto de tokens.
                if prov_cfg.type == "llamacpp":
                    spec["extra"] = {"chat_template_kwargs": {"enable_thinking": False}}
                self._vision_specs.append(spec)
            if self._vision_specs:
                logger.info(
                    "Vision providers: %s",
                    ", ".join(
                        f"{s['name']}(model={s['model']})" for s in self._vision_specs
                    ),
                )

    async def start(self) -> None:
        try:
            from aiogram import Bot, Dispatcher, F
            from aiogram.filters import Command
            from aiogram.types import (
                Message,
                BotCommand,
                BotCommandScopeAllChatAdministrators,
                BotCommandScopeAllGroupChats,
                BotCommandScopeAllPrivateChats,
                BotCommandScopeChat,
                BotCommandScopeDefault,
            )
        except ImportError:
            raise ImportError(
                "aiogram is required for the Telegram channel. "
                "Install it with: pip install eyetor[telegram]"
            )

        bot_token = self._config.bot_token
        if not bot_token:
            raise ValueError(
                "Telegram bot_token is not configured. Set TELEGRAM_BOT_TOKEN env var."
            )

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
            return str(user.id) in allowed_users or (
                user.username and user.username in allowed_users
            )

        @dp.message(Command("start"))
        async def cmd_start(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return
            session_id = f"telegram-{msg.chat.id}"
            self._manager.get_or_create(session_id)
            await msg.answer(
                "¡Hola! Soy Eyetor, un asistente de IA multi-agente.\n"
                "Commands: /reset (new conversation), /help"
            )

        @dp.message(Command("reset"))
        async def cmd_reset(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            session_id = f"telegram-{msg.chat.id}"
            self._manager.reset(session_id)
            await msg.answer("Conversación reiniciada. Corto y cambio :)")

        @dp.message(Command("skills"))
        async def cmd_skills(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            await msg.answer(_format_skills_text(self._skill_reg), parse_mode="HTML")

        @dp.message(Command("agents"))
        async def cmd_agents(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            await msg.answer(_format_agents_text(self._agent_reg), parse_mode="HTML")

        @dp.message(Command("tasks"))
        async def cmd_tasks(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            await msg.answer(_format_tasks_text(self._scheduler), parse_mode="HTML")

        @dp.message(Command("usage"))
        async def cmd_usage(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            session_id = f"telegram-{msg.chat.id}"
            text = _format_usage_text(self._tracker, session_id=session_id)
            await _send_long(msg, text, parse_mode="HTML")

        @dp.message(Command("tools"))
        async def cmd_tools(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            session_id = f"telegram-{msg.chat.id}"
            session = self._manager.get_or_create(session_id)
            text = _format_tools_text(session.tool_registry)
            await _send_long(msg, text, parse_mode="HTML")

        @dp.message(Command("model"))
        async def cmd_model(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            session_id = f"telegram-{msg.chat.id}"
            session = self._manager.get_or_create(session_id)
            parts = (msg.text or "").split()
            if len(parts) == 1:
                # List available providers
                providers = self._manager.list_providers()
                current = session.provider
                current_model = getattr(current, "model", "?")
                prov_name = getattr(current, "_provider_name", None) or "?"
                lines = [
                    f"<b>Proveedor actual:</b> <code>{prov_name}</code> (modelo: {current_model})\n"
                ]
                lines.append("<b>Proveedores disponibles:</b>")
                for name, model in providers.items():
                    lines.append(f"  <code>{name}</code> — {model}")
                lines.append("\nUso: /model &lt;provider&gt; [model]")
                await msg.answer("\n".join(lines), parse_mode="HTML")
            else:
                provider_name = parts[1]
                model_override = parts[2] if len(parts) > 2 else None
                try:
                    result = session.change_provider(provider_name, model_override)
                    await msg.answer(result)
                except Exception as exc:
                    await msg.answer(f"Error: {exc}")

        # --- Dynamic skill commands ---
        _skill_commands = []
        if self._skill_reg:
            from eyetor.skills.executor import run_script as _run_skill_script

            for _meta, _cmd in self._skill_reg.get_all_commands():
                _skill_commands.append(_cmd)

                if _cmd.action == "script":
                    _script_path = _meta.path / "scripts" / _cmd.script
                    _default_args = list(_cmd.args)
                    _parse = _cmd.parse_mode or None

                    @dp.message(Command(_cmd.name))
                    async def _skill_script_handler(
                        msg: Message,
                        _path=_script_path,
                        _args=_default_args,
                        _pm=_parse,
                    ) -> None:
                        if not _is_authorized(msg):
                            return
                        user_args = (msg.text or "").split()[1:]
                        raw = await _run_skill_script(_path, _args + user_args)
                        await _send_skill_script_result(msg, raw, parse_mode=_pm)

                elif _cmd.action == "prompt":
                    _template = _cmd.prompt

                    @dp.message(Command(_cmd.name))
                    async def _skill_prompt_handler(
                        msg: Message,
                        _tmpl=_template,
                    ) -> None:
                        if not _is_authorized(msg):
                            return
                        user_parts = (msg.text or "").split(maxsplit=1)
                        args_text = user_parts[1] if len(user_parts) > 1 else ""
                        prompt_text = _tmpl.replace("{args}", args_text)

                        session_id = f"telegram-{msg.chat.id}"
                        session = self._manager.get_or_create(session_id)
                        placeholder = await msg.answer("...")
                        buffer = ""
                        last_edit = ""
                        try:
                            current_channel.set("telegram")
                            async for chunk in session.send(prompt_text):
                                buffer += chunk
                                if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                                    try:
                                        await placeholder.edit_text(buffer or "...")
                                        last_edit = buffer
                                    except Exception:
                                        pass
                            if buffer:
                                html = _md_to_html(buffer)
                                await _finalize_as_new(msg, placeholder, html, buffer)
                        except Exception as exc:
                            logger.exception("Skill prompt command error")
                            await placeholder.edit_text(f"Error: {_format_exc(exc)}")

        @dp.message(Command("thinking"))
        async def cmd_thinking(msg: Message) -> None:
            chat_id = msg.chat.id
            current = self._show_thinking.get(chat_id, False)
            self._show_thinking[chat_id] = not current
            state = "activado ✅" if not current else "desactivado ❌"
            await msg.answer(f"Modo thinking {state}")

        @dp.message(Command("help"))
        async def cmd_help(msg: Message) -> None:
            extra = ""
            for _sc in _skill_commands:
                extra += f"/{_sc.name} — {_sc.description}\n"
            await msg.answer(
                "Eyetor commands:\n"
                "/reset — start a new conversation\n"
                "/skills — list available skills\n"
                "/agents — list loaded subagent definitions\n"
                "/tools — list registered tools\n"
                "/model — list or change LLM provider\n"
                "/tasks — list scheduled tasks\n"
                "/usage — show token usage and costs\n"
                "/thinking — toggle thinking display\n"
                f"{extra}"
                "/help — show this help\n\n"
                "Send a message to chat, a voice note to transcribe, "
                "or a photo to analyze."
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
                current_channel.set("telegram")
                async for chunk in session.send(msg.text or ""):
                    buffer += chunk
                    if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                        try:
                            await placeholder.edit_text(buffer or "...")
                            last_edit = buffer
                        except Exception:
                            pass  # Ignore edit conflicts

                # Send reasoning/thinking block as a separate message (if enabled)
                if session.last_reasoning and self._show_thinking.get(msg.chat.id, False):
                    reasoning_html = f"💭 <blockquote>{_escape_html(session.last_reasoning.strip())}</blockquote>"
                    try:
                        await msg.answer(reasoning_html, parse_mode="HTML")
                    except Exception:
                        await msg.answer(f"💭 {session.last_reasoning.strip()}")

                # Strip [IMAGE:...] markers from text (images sent separately)
                clean_buffer = _IMAGE_MARKER_RE.sub("", buffer).strip()
                image_paths = _collect_image_paths(buffer, session)
                if clean_buffer:
                    await _finalize_as_new(
                        msg, placeholder, _md_to_html(clean_buffer), clean_buffer
                    )
                else:
                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    # No text and no image → never leave the user with silence.
                    if not image_paths:
                        await msg.answer(
                            "(no he podido generar una respuesta; reintenta)"
                        )

                # Send generated images: from [IMAGE:] markers + tool results
                await _send_images(msg, image_paths)

            except Exception as exc:
                logger.exception("Telegram message handler error")
                await placeholder.edit_text(f"Error: {_format_exc(exc)}")

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
            placeholder = None
            try:
                import base64 as _b64
                import io as _io

                # Download photo bytes into memory
                tg_file = await bot.get_file(photo.file_id)
                buf = _io.BytesIO()
                await bot.download_file(tg_file.file_path, destination=buf)
                img_bytes = buf.getvalue()
                img_b64 = _b64.b64encode(img_bytes).decode()

                # Persist to disk (skills that need the file path still work)
                images_dir = Path.home() / ".eyetor" / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                import time as _time

                img_path = images_dir / f"{msg.chat.id}_{int(_time.time())}.jpg"
                img_path.write_bytes(img_bytes)

                placeholder = await msg.answer("📷 Procesando imagen...")

                # Step 1: Send image to the vision provider to get a description
                description = await _describe_image(
                    img_b64,
                    caption,
                    specs=self._vision_specs or None,
                )
                logger.debug("Vision description: %s", description[:300])

                # Step 2: Send the description (+ metadata) to the main LLM session
                user_text = caption.strip() if caption.strip() else ""
                caption_date = _extract_caption_date(user_text)
                prompt = _build_image_prompt(
                    user_text=user_text,
                    description=description,
                    img_path=img_path,
                    caption_date=caption_date,
                )

                session_id = f"telegram-{msg.chat.id}"
                session = self._manager.get_or_create(session_id)

                buffer = ""
                last_edit = ""
                current_channel.set("telegram")
                async for chunk in session.send(prompt):
                    buffer += chunk
                    if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                        try:
                            await placeholder.edit_text(buffer or "...")
                            last_edit = buffer
                        except Exception:
                            pass

                clean_buffer = _IMAGE_MARKER_RE.sub("", buffer).strip()
                image_paths = _collect_image_paths(buffer, session)
                if clean_buffer:
                    html = _md_to_html(clean_buffer)
                    await _finalize_as_new(msg, placeholder, html, clean_buffer)
                else:
                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    if not image_paths:
                        await msg.answer(
                            "(no he podido generar una respuesta; reintenta)"
                        )

                await _send_images(msg, image_paths)
            except Exception as exc:
                logger.exception("Photo handler error")
                detail = _format_exc(exc)
                if placeholder is not None:
                    try:
                        await placeholder.edit_text(f"Error procesando la imagen: {detail}")
                    except Exception:
                        await msg.answer(f"Error procesando la imagen: {detail}")
                else:
                    await msg.answer(f"No se pudo procesar la foto: {detail}")

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

            placeholder = await msg.answer(
                f"🎤 <i>{_escape_html(transcription)}</i>\n\n...", parse_mode="HTML"
            )
            buffer = ""
            last_edit = ""
            try:
                current_channel.set("telegram")
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

                clean_buffer = _IMAGE_MARKER_RE.sub("", buffer).strip()
                image_paths = _collect_image_paths(buffer, session)
                if clean_buffer:
                    html = f"🎤 <i>{_escape_html(transcription)}</i>\n\n{_md_to_html(clean_buffer)}"
                    plain = f"🎤 {transcription}\n\n{clean_buffer}"
                    await _finalize_as_new(msg, placeholder, html, plain)
                else:
                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    if not image_paths:
                        await msg.answer(
                            "(no he podido generar una respuesta; reintenta)"
                        )

                await _send_images(msg, image_paths)
            except Exception as exc:
                logger.exception("Telegram voice handler error")
                await placeholder.edit_text(f"Error: {_format_exc(exc)}")

        commands = [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="reset", description="Start a new conversation"),
            BotCommand(command="skills", description="List available skills"),
            BotCommand(command="agents", description="List loaded subagent definitions"),
            BotCommand(command="tasks", description="List scheduled tasks"),
            BotCommand(command="usage", description="Show token usage and costs"),
            BotCommand(command="tools", description="List registered tools"),
            BotCommand(command="model", description="List or change LLM provider"),
        ]
        for _sc in _skill_commands:
            commands.append(BotCommand(command=_sc.name, description=_sc.description))
        commands.append(BotCommand(command="help", description="Show help"))
        await _sync_bot_commands(
            bot,
            commands,
            allowed_users=allowed_users,
            scopes={
                "default": BotCommandScopeDefault,
                "all_private": BotCommandScopeAllPrivateChats,
                "all_group": BotCommandScopeAllGroupChats,
                "all_admins": BotCommandScopeAllChatAdministrators,
                "chat": BotCommandScopeChat,
            },
        )

        logger.info("Starting Telegram bot...")
        await dp.start_polling(bot)

    async def stop(self) -> None:
        if self._dp:
            await self._dp.stop_polling()
        if self._bot:
            await self._bot.session.close()


def _format_exc(exc: BaseException) -> str:
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


async def _sync_bot_commands(
    bot,
    commands: list,
    *,
    allowed_users: set[str],
    scopes: dict[str, type],
) -> None:
    """Replace stale Telegram command menus across common scopes.

    Telegram stores bot command menus by scope and language. A previous agent
    using the same bot token may have registered commands in a more specific
    scope than ``default`` (for example ``all_private_chats`` or a concrete
    chat), which overrides a later plain ``set_my_commands`` call. We clear the
    scopes Eyetor can reasonably know, then set the current command list.
    """
    languages = (None, "es", "en")
    base_scopes = [
        scopes["default"](),
        scopes["all_private"](),
        scopes["all_group"](),
        scopes["all_admins"](),
    ]
    chat_scopes = []
    for raw in allowed_users:
        try:
            chat_scopes.append(scopes["chat"](chat_id=int(raw)))
        except (TypeError, ValueError):
            # Usernames cannot be used as BotCommandScopeChat ids.
            continue

    for scope in [*base_scopes, *chat_scopes]:
        for language_code in languages:
            try:
                await bot.delete_my_commands(
                    scope=scope,
                    language_code=language_code,
                )
            except Exception as exc:
                logger.warning(
                    "Could not delete Telegram commands for scope=%s lang=%s: %s",
                    type(scope).__name__,
                    language_code or "default",
                    exc,
                )

    # Set both default and all-private menus. If we know the numeric allowed
    # chat id, set it explicitly too so a stale chat-specific menu is replaced.
    target_scopes = [scopes["default"](), scopes["all_private"](), *chat_scopes]
    for scope in target_scopes:
        try:
            await bot.set_my_commands(commands, scope=scope)
        except Exception as exc:
            logger.warning(
                "Could not set Telegram commands for scope=%s: %s",
                type(scope).__name__,
                exc,
            )
            if type(scope).__name__ == "BotCommandScopeDefault":
                raise


async def _describe_image(
    img_b64: str,
    caption: str = "",
    *,
    specs: list[dict] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Send an image to a vision LLM and return a text description.

    Tries each provider spec in ``specs`` (``{"name", "base_url", "api_key",
    "model"}``) in order, moving on to the next when one fails (connection error,
    HTTP error, or a 200 body without ``choices`` — common with OpenRouter ``:free``
    models that return ``{"error": {...}}`` with status 200). Raises only when every
    provider fails.

    When ``specs`` is None, a single spec is built from the ``base_url``/``api_key``/
    ``model`` arguments, falling back to VISION_BASE_URL / VISION_API_KEY /
    VISION_MODEL environment variables.

    The vision prompt stays channel-generic: it asks for visible text,
    dates, numbers and layout without mentioning any downstream tool.
    """
    import httpx

    if not specs:
        specs = [
            {
                "name": "vision",
                "base_url": base_url
                or os.environ.get("VISION_BASE_URL", "http://localhost:8080/v1"),
                "api_key": api_key
                if api_key is not None
                else os.environ.get("VISION_API_KEY", ""),
                "model": model or os.environ.get("VISION_MODEL", "default"),
            }
        ]

    prompt = (
        "Analiza esta imagen de forma precisa y neutral. Indica primero qué "
        "tipo de imagen es (documento, foto, captura de pantalla, objeto, "
        "lista, tabla, etc.).\n\n"
        "Transcribe el texto visible importante. Si aparecen fechas, importes, "
        "cantidades, tablas, listas o pares nombre-valor, extráelos de forma "
        "ordenada y completa. Si algún dato no se lee con seguridad, márcalo "
        "como dudoso en vez de inventarlo.\n\n"
        "Si no hay texto relevante, describe los elementos visibles y su contexto."
    )
    if caption.strip():
        prompt += (
            "\n\nCaption del usuario, solo como contexto adicional:\n"
            f"{caption.strip()}\n\n"
            "Si el caption contiene datos concretos que no se ven en la imagen "
            "(por ejemplo fechas, lugares o cantidades), menciónalos aparte."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]

    async def _call_one(spec: dict) -> str:
        base = (spec.get("base_url") or "").rstrip("/")
        key = (spec.get("api_key") or "").strip()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lsanchezojo/eyetor",
            "X-Title": "Eyetor",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": spec.get("model") or "default",
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
            **(spec.get("extra") or {}),
        }
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            resp = await client.post(
                f"{base}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            # OpenRouter :free models often return HTTP 200 with an error body.
            err = ""
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                err = data["error"].get("message", "")
            logger.warning(
                "Vision provider %s returned no choices; body: %s",
                spec.get("name", "?"),
                str(data)[:300],
            )
            raise RuntimeError(err or "respuesta sin 'choices'")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("respuesta sin contenido")
        return content

    errors: list[str] = []
    for spec in specs:
        name = spec.get("name", "?")
        try:
            return await _call_one(spec)
        except Exception as exc:  # noqa: BLE001 — try next provider on any failure
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            logger.warning(
                "Vision provider %s failed: %s; trying next", name, exc
            )

    raise RuntimeError(
        "Todos los proveedores de visión fallaron — " + "; ".join(errors)
    )


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
            return await _transcribe_via_api(
                tmp_path, whisper_url or None, openai_key or None, suffix
            )

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


async def _transcribe_via_api(
    path: str, base_url: str | None, api_key: str | None, suffix: str
) -> str:
    """Transcribe using an OpenAI-compatible /v1/audio/transcriptions endpoint."""
    import httpx

    url = (
        f"{base_url.rstrip('/')}/v1/audio/transcriptions"
        if base_url
        else "https://api.openai.com/v1/audio/transcriptions"
    )
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


def _split_message(text: str, limit: int = _TG_MAX_LEN) -> list[str]:
    """Split text into chunks of at most *limit* characters.

    Tries to break at newlines first, then at spaces, to keep messages
    readable.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to break at last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            # Try space
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit  # hard cut
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


async def _finalize_as_new(msg, placeholder, html: str, plain: str) -> None:
    """Delete the streaming placeholder and send the final answer as a NEW
    message.

    Editing the placeholder keeps its original send time (when "..." was
    posted, right after the user's question), so the visible timestamp would
    not reflect when the answer was actually ready. Sending a fresh message
    fixes that. On HTML parse failure (e.g. crossed <b>/<i> tags) it retries
    as plain text. ``_send_long`` splits messages over Telegram's limit.
    """
    try:
        from aiogram.exceptions import TelegramBadRequest
    except ImportError:  # pragma: no cover
        TelegramBadRequest = Exception  # type: ignore

    try:
        await placeholder.delete()
    except Exception:
        pass

    try:
        await _send_long(msg, html, parse_mode="HTML")
        return
    except TelegramBadRequest as exc:
        logger.warning(
            "Telegram HTML parse failed on finalize, retrying as plain text: %s", exc
        )
    except Exception as exc:
        logger.warning("Finalize HTML send failed, retrying as plain text: %s", exc)

    try:
        await _send_long(msg, plain or "...")
    except Exception as exc:
        logger.error("Telegram plain-text finalize fallback failed: %s", exc)


async def _send_long(msg, text: str, parse_mode: str | None = None) -> None:
    """Send a potentially long message, splitting if it exceeds Telegram's limit."""
    for part in _split_message(text):
        await msg.answer(part, parse_mode=parse_mode)


async def _send_skill_script_result(msg, raw: str, parse_mode: str | None = None) -> None:
    """Render a skill script's stdout for a channel command.

    Scripts conventionally print a JSON object. When that object carries an
    ``image_path`` (e.g. a screenshot), the file is sent as a photo; otherwise
    a human-friendly ``message``/``error`` is shown. Non-JSON output (or any
    other shape) falls back to sending the raw text verbatim.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = None

    if not isinstance(data, dict):
        await _send_long(msg, raw, parse_mode=parse_mode)
        return

    image_path = data.get("image_path")
    message = data.get("message") or ""
    if image_path and Path(image_path).exists():
        from aiogram.types import FSInputFile

        try:
            await msg.answer_photo(FSInputFile(image_path), caption=message or None)
            return
        except Exception as exc:
            logger.error("Failed to send skill image %s: %s", image_path, exc)
            await _send_long(msg, message or f"[Image: {image_path}]", parse_mode=parse_mode)
            return

    if data.get("error"):
        await _send_long(msg, f"⚠️ {data['error']}", parse_mode=parse_mode)
        return

    if message:
        await _send_long(msg, message, parse_mode=parse_mode)
        return

    await _send_long(msg, raw, parse_mode=parse_mode)


def _collect_image_paths(buffer: str, session) -> list[str]:
    """Collect image paths from [IMAGE:] markers and generate_image tool results.

    Only scans tool results from the latest turn (after the last user message).
    """
    paths: set[str] = set()

    # From markers in LLM text
    for p in _IMAGE_MARKER_RE.findall(buffer):
        paths.add(p.strip())

    # From tool results in session history (only current turn)
    history = session.get_history()
    for msg in reversed(history):
        if msg.role == "user":
            break
        if msg.role == "tool" and msg.content:
            try:
                data = json.loads(msg.content)
                if (
                    isinstance(data, dict)
                    and data.get("status") == "ok"
                    and "image_path" in data
                ):
                    paths.add(data["image_path"])
            except (json.JSONDecodeError, TypeError):
                pass

    return list(paths)


async def _send_images(msg, image_paths: list[str]) -> None:
    """Send image files as Telegram photos."""
    if not image_paths:
        return
    from aiogram.types import FSInputFile

    for img_path in image_paths:
        p = Path(img_path)
        if p.exists():
            try:
                await msg.answer_photo(FSInputFile(p))
            except Exception as exc:
                logger.error("Failed to send image %s: %s", img_path, exc)
                await msg.answer(f"[Image: {img_path}]")
        else:
            logger.warning("Image file not found: %s", img_path)
            await msg.answer(f"[Image not found: {img_path}]")


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
        segments.append(("text", text[last : m.start()]))
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
    result = "".join(parts)

    # Guard: if the conversion produced crossed/unbalanced tags (which
    # Telegram rejects), fall back to escaped plain text.
    if not _is_balanced_html(result):
        return _escape_html(text)
    return result


_TAG_RE = re.compile(r"</?(?:b|i|s|u|code|pre)(?:\s[^>]*)?>")


def _is_balanced_html(s: str) -> bool:
    """Check that b/i/s/u/code/pre tags are strictly nested (no crossing)."""
    stack: list[str] = []
    for m in _TAG_RE.finditer(s):
        tag = m.group(0)
        name_m = re.match(r"</?([a-z]+)", tag)
        if not name_m:
            continue
        name = name_m.group(1)
        if tag.startswith("</"):
            if not stack or stack[-1] != name:
                return False
            stack.pop()
        else:
            stack.append(name)
    return not stack


def _inline_md_to_html(text: str) -> str:
    """Apply inline Markdown → HTML transforms on a plain-text segment.

    Inline code spans (`...`) are tokenised first so that their contents
    are never touched by later bold/italic transforms — otherwise an
    underscore inside a code span would be matched as italic, producing
    crossed tags that Telegram rejects.
    """
    # Step 1: extract inline code spans as opaque tokens.
    inline_code_re = re.compile(r"`([^`\n]+)`")
    code_spans: list[str] = []

    def _stash_code(m: re.Match) -> str:
        idx = len(code_spans)
        code_spans.append(f"<code>{_escape_html(m.group(1))}</code>")
        return f"\x00CODE{idx}\x00"

    text = inline_code_re.sub(_stash_code, text)

    # Step 2: escape HTML entities on the rest.
    text = _escape_html(text)

    # Step 3: neutralise backslash-escaped markdown specials (\_ \* \` \~)
    # so they are not interpreted as emphasis markers.
    escape_map = {
        "_": "\x00U\x00",
        "*": "\x00A\x00",
        "`": "\x00B\x00",
        "~": "\x00T\x00",
    }
    text = re.sub(r"\\([_*`~])", lambda m: escape_map[m.group(1)], text)

    # Step 4: emphasis / headers / lists.
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

    # Step 5: restore escaped markdown specials as literal characters.
    for ch, token in escape_map.items():
        text = text.replace(token, ch)

    # Step 6: restore inline code spans.
    def _restore_code(m: re.Match) -> str:
        return code_spans[int(m.group(1))]

    text = re.sub(r"\x00CODE(\d+)\x00", _restore_code, text)

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


_LOCAL_PROVIDERS = {"ollama", "llamacpp", "llamacpp-mtp", "local"}
_CLOUD_PROVIDERS = {"openrouter", "anthropic", "google", "openai", "azure"}


def _fmt_num(n: int) -> str:
    """Format integer with Spanish thousands separator (dot)."""
    return f"{n:,}".replace(",", ".")


def _provider_emoji(provider: str) -> str:
    return "🖥" if provider.lower() in _LOCAL_PROVIDERS else "🌐"


def _model_short(model: str, max_len: int = 32) -> str:
    name = model.split("/")[-1] if "/" in model else model
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _format_usage_text(tracker, session_id: str | None = None) -> str:
    """Return an HTML-formatted usage report for Telegram."""
    if tracker is None:
        return "Tracking de uso no configurado."

    from collections import defaultdict
    from datetime import datetime

    now = datetime.now()
    utc_offset = now - datetime.utcnow()  # for converting stored UTC → local
    day_names = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    month_names = [
        "ene",
        "feb",
        "mar",
        "abr",
        "may",
        "jun",
        "jul",
        "ago",
        "sep",
        "oct",
        "nov",
        "dic",
    ]
    day_label = f"{day_names[now.weekday()]} {now.day} {month_names[now.month - 1]}"

    lines: list[str] = [f"<b>📊 Uso · {day_label}</b>", "━━━━━━━━━━━━━━━━"]

    # --- Session totals (if session_id provided) ---
    day_records = tracker.get_records(period="day")

    if session_id and day_records:
        sess_records = [r for r in day_records if r.session_id == session_id]
        if sess_records:
            s_prompt = sum(r.prompt_tokens for r in sess_records)
            s_comp = sum(r.completion_tokens for r in sess_records)
            s_cost = sum(r.estimated_cost for r in sess_records)
            cost_str = f"${s_cost:.4f}" if s_cost > 0 else "$0"
            lines.append(
                f"\n💬 <b>Sesión actual</b>"
                f"\n   {len(sess_records)} ll · "
                f"{_fmt_num(s_prompt)}→{_fmt_num(s_comp)} tok · {cost_str}"
            )

    # --- Footer: day vs week totals ---
    week_records = tracker.get_records(period="week")

    def _totals_from_records(recs):
        prompt = sum(r.prompt_tokens for r in recs)
        completion = sum(r.completion_tokens for r in recs)
        cost = sum(r.estimated_cost for r in recs)
        tool = sum(1 for r in recs if r.finish_reason == "tool_calls")
        return prompt, completion, len(recs), tool, cost

    def _footer_line(
        label: str, prompt: int, comp: int, calls: int, tool: int, cost: float
    ) -> str:
        tok_part = f"{_fmt_num(prompt)} → {_fmt_num(comp)} tokens"
        calls_part = f"{calls} llamadas ({tool} tool_call)"
        cost_part = f"${cost:.4f}" if cost > 0 else "$0"
        return f"{label}   {tok_part} · {calls_part} · {cost_part}"

    def _append_totals_footer() -> None:
        day_prompt, day_comp, day_calls, day_tool, day_cost = _totals_from_records(
            day_records
        )
        week_prompt, week_comp, week_calls, week_tool, week_cost = _totals_from_records(
            week_records
        )
        lines.append("\n──────────────")
        lines.append(
            _footer_line("📅 Hoy   ", day_prompt, day_comp, day_calls, day_tool, day_cost)
        )
        lines.append(
            _footer_line(
                "📆 Semana", week_prompt, week_comp, week_calls, week_tool, week_cost
            )
        )

    # --- Individual calls for today, grouped by (provider, model) ---
    _MAX_CALLS_SHOWN = 5

    if not day_records:
        if week_records:
            lines.append("\nSin actividad hoy.")
            _append_totals_footer()
        else:
            lines.append("\nSin actividad registrada.")
        return "\n".join(lines)

    # Group preserving insertion order; display calls oldest→newest
    groups: dict[tuple, list] = defaultdict(list)
    for r in reversed(day_records):
        groups[(r.provider, r.model)].append(r)

    for (provider, model), calls in groups.items():
        emoji = _provider_emoji(provider)
        provider_label = _escape_html(provider)
        model_label = _escape_html(_model_short(model))
        shown = list(reversed(calls[-_MAX_CALLS_SHOWN:]))
        hidden = len(calls) - _MAX_CALLS_SHOWN
        label_extra = f" (últimas {_MAX_CALLS_SHOWN})" if hidden > 0 else ""
        lines.append(f"\n{emoji} <b>{provider_label}</b> · {model_label}{label_extra}")

        total_cost = sum(r.estimated_cost for r in calls)
        total_tok = sum(r.prompt_tokens + r.completion_tokens for r in calls)
        cost_sum = f"${total_cost:.4f}" if total_cost > 0 else "$0"
        lines.append(
            f"   {len(calls)} ll · {_fmt_num(total_tok)} tok · {cost_sum}"
        )

        for i, r in enumerate(shown):
            branch = "└" if i == len(shown) - 1 else "├"
            ts_local = datetime.fromisoformat(r.timestamp) + utc_offset
            hhmm = ts_local.strftime("%H:%M")
            tok = f"{_fmt_num(r.prompt_tokens)}→{_fmt_num(r.completion_tokens)}"
            speed = f"{r.speed_tps:.1f} tps".replace(".", ",") if r.speed_tps else "—"
            finish = r.finish_reason or "—"
            lines.append(
                f"   {branch} <code>{hhmm}</code>  {tok}  {speed}  {finish}"
            )

    # --- Breakdown by phase / agent (today) ---
    def _agg(key_fn, recs):
        agg: dict[str, list] = defaultdict(list)
        for r in recs:
            k = key_fn(r)
            if k:
                agg[k].append(r)
        return agg

    def _emit_breakdown(title: str, agg: dict) -> None:
        if not agg:
            return
        lines.append(f"\n{title}")
        ordered = sorted(
            agg.items(),
            key=lambda kv: -sum(
                x.prompt_tokens + x.completion_tokens for x in kv[1]
            ),
        )
        for key, recs in ordered:
            tok = sum(r.prompt_tokens + r.completion_tokens for r in recs)
            cost = sum(r.estimated_cost for r in recs)
            cost_str = f"${cost:.4f}" if cost > 0 else "$0"
            lines.append(
                f"  {_escape_html(key)}: {len(recs)} ll · "
                f"{_fmt_num(tok)} tok · {cost_str}"
            )

    _emit_breakdown("🧩 <b>Por fase</b>", _agg(lambda r: r.phase, day_records))
    _emit_breakdown("🤖 <b>Por agente</b>", _agg(lambda r: r.agent, day_records))

    _append_totals_footer()

    return "\n".join(lines)


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


def _format_agents_text(agent_reg) -> str:
    """Return a plain-text subagent list with descriptions for Telegram (HTML)."""
    if agent_reg is None:
        return "No agents configured."
    agents = agent_reg.all()
    if not agents:
        return "No agents configured."
    lines = ["<b>Available agents:</b>"]
    for a in agents:
        lines.append(f"  <code>{a.name}</code> — {a.description}")
    return "\n".join(lines)


def _format_tools_text(tool_registry) -> str:
    """Return a formatted list of registered tools for Telegram (HTML)."""
    if tool_registry is None:
        return "No tools registered."
    tools = tool_registry._tools
    if not tools:
        return "No tools registered."
    lines = [f"<b>Registered tools ({len(tools)}):</b>"]
    for name, defn in tools.items():
        lines.append(f"  <code>{name}</code> — {defn.description}")
    return "\n".join(lines)
