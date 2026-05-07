"""Telegram channel — bot with streaming progressive messages.

Requires the 'telegram' optional dependency:
    pip install eyetor[telegram]

Voice message transcription requires faster-whisper:
    pip install faster-whisper
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import os
from pathlib import Path

from eyetor.channels.base import BaseChannel
from eyetor.channels.errors import format_user_error
from eyetor.chat.manager import SessionManager
from eyetor.config import TelegramChannelConfig
from eyetor.utils.tool_calls import strip_textual_tool_calls

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eyetor.config import VectorConfig
    from eyetor.tracking.usage import UsageTracker

logger = logging.getLogger(__name__)

_CHUNK_TOKENS = 20  # Edit message every N characters
_TG_MAX_LEN = 4096  # Telegram message character limit
_IMAGE_MARKER_RE = re.compile(r"\[IMAGE:(.*?)\]")


def _sanitize_model_text(text: str) -> str:
    cleaned, _ = strip_textual_tool_calls(text)
    return cleaned


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
        dreams_scheduler=None,
    ) -> None:
        self._manager = session_manager
        self._config = config
        self._skill_reg = skill_reg
        self._scheduler = scheduler
        self._tracker = tracker
        self._dreams_scheduler = dreams_scheduler
        self._dp = None
        self._bot = None

        self._show_thinking: dict[int, bool] = {}  # per chat_id toggle for /thinking

        # Resolve vision provider settings from config (fallback to env vars in _describe_image)
        self._vision_base_url: str | None = None
        self._vision_api_key: str | None = None
        self._vision_model: str | None = None
        if full_config and full_config.vision_provider:
            prov_cfg = full_config.providers.get(full_config.vision_provider)
            if prov_cfg:
                self._vision_base_url = prov_cfg.base_url
                self._vision_api_key = prov_cfg.api_key or ""
                self._vision_model = full_config.vision_model or prov_cfg.model
                logger.info(
                    "Vision provider: %s model=%s url=%s",
                    full_config.vision_provider,
                    self._vision_model,
                    self._vision_base_url,
                )

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

        @dp.message(Command("dreams"))
        async def cmd_dreams(msg: Message) -> None:
            if not _is_authorized(msg):
                return
            parts = (msg.text or "").split()
            command = parts[1].lower() if len(parts) > 1 else "list"

            if self._dreams_scheduler is None:
                await msg.answer("Sistema de sueños no configurado.")
                return

            try:
                if command == "list" or command == "":
                    result = await self._dreams_scheduler.handle_list()
                    await _send_long(msg, result, parse_mode="HTML")
                elif command == "run":
                    result = await self._dreams_scheduler.handle_run()
                    await msg.answer(result)
                elif command.startswith("apply"):
                    if len(parts) < 3:
                        await msg.answer("Uso: /dreams apply <id>")
                        return
                    proposal_id = int(parts[2])
                    result = await self._dreams_scheduler.handle_apply(proposal_id)
                    await msg.answer(result)
                elif command.startswith("dismiss"):
                    if len(parts) < 3:
                        await msg.answer("Uso: /dreams dismiss <id>")
                        return
                    proposal_id = int(parts[2])
                    result = await self._dreams_scheduler.handle_dismiss(proposal_id)
                    await msg.answer(result)
                else:
                    await msg.answer(
                        "Uso: /dreams [list|run|apply <id>|dismiss <id>]"
                    )
            except ValueError:
                await msg.answer("ID de propuesta inválido.")
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
                        await _send_long(msg, raw, parse_mode=_pm)

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
                            async for chunk in self._manager.route_and_send(
                                session_id, prompt_text
                            ):
                                buffer += chunk
                                if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                                    try:
                                        visible = _sanitize_model_text(buffer)
                                        await placeholder.edit_text(visible or "...")
                                        last_edit = buffer
                                    except Exception:
                                        pass
                            if buffer:
                                visible = _sanitize_model_text(buffer)
                                html = _md_to_html(visible)
                                await _safe_edit_or_send(msg, placeholder, html, visible)
                        except Exception as exc:
                            logger.exception("Skill prompt command error")
                            await placeholder.edit_text(format_user_error(exc))

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
                async for chunk in self._manager.route_and_send(
                    session_id, msg.text or ""
                ):
                    buffer += chunk
                    if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                        try:
                            visible = _sanitize_model_text(buffer)
                            await placeholder.edit_text(visible or "...")
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

                # Final edit always applies HTML formatting
                if buffer:
                    # Strip [IMAGE:...] markers from text (images sent separately)
                    clean_buffer = _IMAGE_MARKER_RE.sub(
                        "", _sanitize_model_text(buffer)
                    ).strip()
                    html = _md_to_html(clean_buffer) if clean_buffer else ""
                    if html:
                        await _safe_edit_or_send(msg, placeholder, html, clean_buffer)
                    else:
                        await placeholder.delete()

                # Send generated images: from [IMAGE:] markers + tool results
                image_paths = _collect_image_paths(buffer, session)
                await _send_images(msg, image_paths)

            except Exception as exc:
                logger.exception("Telegram message handler error")
                await placeholder.edit_text(format_user_error(exc))

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
            placeholder = await msg.answer("📷 Procesando imagen...")
            try:
                import io as _io

                tg_file = await bot.get_file(photo.file_id)
                buf = _io.BytesIO()
                await bot.download_file(tg_file.file_path, destination=buf)
                await self._handle_image_bytes(
                    msg, placeholder, buf.getvalue(), caption, suffix=".jpg"
                )
            except Exception as exc:
                logger.exception("Photo handler error")
                await _replace_with_friendly(placeholder, msg, exc)

        @dp.message(F.document)
        async def on_document(msg: Message) -> None:
            if not _is_authorized(msg):
                await msg.answer("Unauthorized. Contact the administrator.")
                return

            doc = msg.document
            if doc.file_size and doc.file_size > 25 * 1024 * 1024:
                await msg.answer("Fichero demasiado grande (máx 25 MB).")
                return

            caption = msg.caption or ""
            placeholder = await msg.answer("📎 Procesando fichero...")
            try:
                import io as _io

                tg_file = await bot.get_file(doc.file_id)
                buf = _io.BytesIO()
                await bot.download_file(tg_file.file_path, destination=buf)
                file_bytes = buf.getvalue()

                file_name = doc.file_name or "document"
                suffix = Path(file_name).suffix.lower() or ""
                mime = (doc.mime_type or "").lower()

                # Route 1: image disguised as document → vision pipeline
                if mime.startswith("image/") or suffix in {
                    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
                }:
                    await self._handle_image_bytes(
                        msg, placeholder, file_bytes, caption,
                        suffix=suffix or ".jpg",
                    )
                    return

                # Route 2: text-extractable document → KB extractors
                from eyetor.knowledge.extractors import get_extractor

                extractor = get_extractor(suffix)
                if extractor is not None:
                    await self._handle_document_text(
                        msg, placeholder, file_bytes, caption,
                        file_name=file_name, suffix=suffix, extractor=extractor,
                    )
                    return

                # Route 3: unsupported format → answer caption only
                await self._handle_unsupported_document(
                    msg, placeholder, caption, file_name=file_name, mime=mime,
                )
            except Exception as exc:
                logger.exception("Document handler error")
                await _replace_with_friendly(placeholder, msg, exc)

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
                async for chunk in self._manager.route_and_send(
                    session_id, transcription
                ):
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
                    html = f"🎤 <i>{_escape_html(transcription)}</i>\n\n{_md_to_html(buffer)}"
                    plain = f"🎤 {transcription}\n\n{buffer}"
                    await _safe_edit_or_send(msg, placeholder, html, plain)
            except Exception as exc:
                logger.exception("Telegram voice handler error")
                await placeholder.edit_text(format_user_error(exc))

        commands = [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="reset", description="Start a new conversation"),
            BotCommand(command="skills", description="List available skills"),
            BotCommand(command="tasks", description="List scheduled tasks"),
            BotCommand(command="usage", description="Show token usage and costs"),
            BotCommand(command="tools", description="List registered tools"),
            BotCommand(command="model", description="List or change LLM provider"),
        ]
        for _sc in _skill_commands:
            commands.append(BotCommand(command=_sc.name, description=_sc.description))
        commands.append(BotCommand(command="help", description="Show help"))
        await bot.set_my_commands(commands)

        logger.info("Starting Telegram bot...")
        await dp.start_polling(bot)

    async def stop(self) -> None:
        if self._dp:
            await self._dp.stop_polling()
        if self._bot:
            await self._bot.session.close()

    # ------------------------------------------------------------------
    # Photo / document shared pipelines
    # ------------------------------------------------------------------

    async def _handle_image_bytes(
        self,
        msg,
        placeholder,
        img_bytes: bytes,
        caption: str,
        *,
        suffix: str = ".jpg",
    ) -> None:
        """Vision pipeline: describe the image, then send to the main LLM session.

        Used by both the photo handler and the document handler when the
        attachment is an image disguised as a file.
        """
        import base64 as _b64
        import time as _time

        img_b64 = _b64.b64encode(img_bytes).decode()

        images_dir = Path.home() / ".eyetor" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        img_path = images_dir / f"{msg.chat.id}_{int(_time.time())}{suffix}"
        img_path.write_bytes(img_bytes)

        description = await _describe_image(
            img_b64,
            caption,
            base_url=self._vision_base_url,
            api_key=self._vision_api_key,
            model=self._vision_model,
        )
        logger.debug("Vision description: %s", description[:300])

        prompt = _build_image_prompt(description, caption, img_path)

        await self._stream_session_to_placeholder(msg, placeholder, prompt)

    async def _handle_document_text(
        self,
        msg,
        placeholder,
        file_bytes: bytes,
        caption: str,
        *,
        file_name: str,
        suffix: str,
        extractor,
    ) -> None:
        """Extract text from a supported document and send it to the LLM session.

        Writes bytes to a temp file (extractors operate on paths), runs the
        extractor in a thread (some are blocking I/O), then injects the text as
        context together with the user's caption.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            doc = await asyncio.to_thread(extractor, tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        if doc is None or not (doc.text or "").strip():
            await placeholder.edit_text(
                f"No he podido extraer texto de «{file_name}». "
                f"Si quieres que responda a tu mensaje igualmente, reenvíalo como texto."
            )
            return

        # Cap extracted text to avoid blowing the context. The tool-result cap
        # (P0a, default 8000 chars) protects tools, but here we're feeding the
        # text directly into the prompt — apply our own cap.
        cap = 12000
        body = doc.text.strip()
        truncated_note = ""
        if len(body) > cap:
            truncated_note = f"\n\n[…truncado, {len(body) - cap} chars adicionales omitidos]"
            body = body[:cap]

        title_line = f" (título: «{doc.title}»)" if doc.title else ""
        user_text = caption.strip()
        intro = (
            f"El usuario ha adjuntado un fichero «{file_name}»{title_line}"
        )
        if user_text:
            prompt = (
                f"{intro} con este mensaje: «{user_text}»\n\n"
                f"Contenido extraído del fichero:\n---\n{body}{truncated_note}\n---\n\n"
                f"Responde a la petición del usuario usando el contenido del fichero "
                f"como contexto principal."
            )
        else:
            prompt = (
                f"{intro}, sin mensaje adicional.\n\n"
                f"Contenido extraído:\n---\n{body}{truncated_note}\n---\n\n"
                f"Resume o comenta lo que consideres relevante para el usuario."
            )

        await self._stream_session_to_placeholder(msg, placeholder, prompt)

    async def _handle_unsupported_document(
        self,
        msg,
        placeholder,
        caption: str,
        *,
        file_name: str,
        mime: str,
    ) -> None:
        """Fallback for documents whose format we cannot parse.

        Answers the caption (if any) so the user still gets a response, and
        tells them which formats are supported.
        """
        from eyetor.knowledge.extractors import supported_extensions

        supported = ", ".join(supported_extensions())
        notice = (
            f"He recibido «{file_name}» (mime: {mime or 'desconocido'}) pero no "
            f"puedo extraer su contenido. Formatos soportados: {supported}."
        )
        user_text = caption.strip()
        if not user_text:
            await placeholder.edit_text(notice)
            return

        prompt = (
            f"El usuario ha adjuntado un fichero «{file_name}» en un formato "
            f"que no puedo leer (mime: {mime or 'desconocido'}). "
            f"Su mensaje es: «{user_text}». "
            f"Responde a su mensaje sin asumir nada del contenido del fichero."
        )
        await self._stream_session_to_placeholder(
            msg, placeholder, prompt, prefix_notice=notice + "\n\n"
        )

    async def _stream_session_to_placeholder(
        self,
        msg,
        placeholder,
        prompt: str,
        *,
        prefix_notice: str = "",
    ) -> None:
        """Run a session.send() and stream tokens into the placeholder message."""
        session_id = f"telegram-{msg.chat.id}"
        self._manager.get_or_create(session_id)

        buffer = ""
        last_edit = ""
        async for chunk in self._manager.route_and_send(
            session_id, prompt, allow_chain=False
        ):
            buffer += chunk
            if len(buffer) - len(last_edit) >= _CHUNK_TOKENS:
                try:
                    visible = prefix_notice + _sanitize_model_text(buffer)
                    await placeholder.edit_text(visible or "...")
                    last_edit = buffer
                except Exception:
                    pass
        if buffer:
            clean_buffer = _sanitize_model_text(buffer)
            html = (
                _escape_html(prefix_notice) + _md_to_html(clean_buffer)
                if prefix_notice
                else _md_to_html(clean_buffer)
            )
            await _safe_edit_or_send(
                msg, placeholder, html, prefix_notice + clean_buffer
            )


async def _replace_with_friendly(placeholder, msg, exc: BaseException) -> None:
    """Replace a placeholder with a user-friendly error message."""
    friendly = format_user_error(exc)
    if placeholder is not None:
        try:
            await placeholder.edit_text(friendly)
            return
        except Exception:
            pass
    await msg.answer(friendly)


def _format_exc(exc: BaseException) -> str:
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _build_image_prompt(description: str, caption: str, img_path: Path) -> str:
    """Build the main-agent prompt for an already-described image attachment."""
    user_text = caption.strip()
    attachment_context = (
        "## Metadatos internos del adjunto\n"
        f"- local_attachment_path: {img_path}\n"
        "- Usa este path solo como argumento de una herramienta registrada que "
        "procese adjuntos o imágenes."
    )
    vision_guard = (
        "El análisis de la imagen de arriba es la fuente autoritativa de su "
        "contenido visible. No intentes inspeccionar el fichero con herramientas "
        "genéricas de lectura de archivos para entender la imagen; usa la ruta solo como "
        "entrada de una herramienta registrada que procese adjuntos o imágenes."
    )
    response_guard = (
        "Tu respuesta al usuario debe ser humana y útil. No respondas solo con "
        "la ruta del archivo local y no la menciones salvo que sea imprescindible "
        "para explicar un error técnico. Si no usas ninguna herramienta, responde "
        "igualmente basándote en el análisis de visión."
    )
    if user_text:
        return (
            f"El usuario ha enviado una imagen con este mensaje: «{user_text}»\n\n"
            f"Análisis de la imagen (modelo de visión):\n{description}\n\n"
            f"{attachment_context}\n\n"
            f"{vision_guard}\n\n"
            f"{response_guard}\n\n"
            f"Responde a lo que pide el usuario usando ese análisis como contexto. "
            f"Si una herramienta registrada necesita el archivo original del "
            f"adjunto, usa `local_attachment_path`."
        )
    return (
        f"El usuario ha enviado una imagen sin mensaje adicional.\n\n"
        f"Análisis de la imagen (modelo de visión):\n{description}\n\n"
        f"{attachment_context}\n\n"
        f"{vision_guard}\n\n"
        f"{response_guard}\n\n"
        f"Responde al usuario basándote en el contenido descrito. Si una "
        f"herramienta registrada necesita el archivo original del adjunto, usa "
        f"`local_attachment_path`."
    )


async def _describe_image(
    img_b64: str,
    caption: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Send an image to the configured vision LLM and return a text description.

    Connection settings are resolved from config (vision_provider / vision_model in
    default.yaml). Falls back to VISION_BASE_URL / VISION_API_KEY / VISION_MODEL
    environment variables when not provided.

    The vision prompt asks the model to classify the image type and, if it is
    a receipt/ticket, extract structured data (store, items, prices).
    """
    import httpx

    base_url = (
        base_url or os.environ.get("VISION_BASE_URL", "http://localhost:8080/v1")
    ).rstrip("/")
    api_key = (
        api_key if api_key is not None else os.environ.get("VISION_API_KEY", "")
    ).strip()
    model = model or os.environ.get("VISION_MODEL", "default")

    if caption.strip():
        prompt = caption.strip()
    else:
        prompt = (
            "Analiza esta imagen. Primero indica qué tipo de imagen es "
            "(ticket de compra, factura, documento, foto, captura de pantalla, etc.).\n\n"
            "Si es un ticket o recibo de compra, extrae TODOS los productos y precios "
            "que puedas leer, incluyendo el nombre de la tienda y la fecha si aparecen. "
            "Usa este formato:\n"
            "- Tipo: ticket de compra\n"
            "- Tienda: [nombre]\n"
            "- Fecha: [fecha si visible]\n"
            "- Productos:\n"
            "  - [nombre producto]: [precio]€\n"
            "  - ...\n"
            "- Total: [total]€\n\n"
            "Si NO es un ticket, describe la imagen de forma detallada."
        )

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lsanchezojo/eyetor",
        "X-Title": "Eyetor",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
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
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        resp = await client.post(
            f"{base_url}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Vision API: respuesta sin choices/message: {data!r}"
            ) from exc

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            logger.warning(
                "Vision content vacío; usando reasoning como descripción "
                "(finish_reason=%s)",
                data["choices"][0].get("finish_reason"),
            )
            return reasoning

        logger.error("Vision API devolvió content y reasoning vacíos: %r", data)
        raise RuntimeError(
            "El modelo de visión no devolvió descripción "
            f"(finish_reason={data['choices'][0].get('finish_reason')!r}). "
            "Reintenta el envío."
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


async def _safe_edit_or_send(msg, placeholder, html: str, plain: str) -> None:
    """Edit *placeholder* with HTML; on parse failure retry as plain text.

    Protects against malformed HTML that Telegram rejects (e.g. crossed
    <b>/<i> tags). "Message is not modified" errors are treated as
    success — the placeholder already shows identical content from the
    streaming phase, so re-sending would duplicate the message.
    """
    try:
        from aiogram.exceptions import TelegramBadRequest
    except ImportError:  # pragma: no cover
        TelegramBadRequest = Exception  # type: ignore

    def _not_modified(exc: BaseException) -> bool:
        return "not modified" in str(exc).lower()

    # Happy path: HTML.
    if len(html) <= _TG_MAX_LEN:
        try:
            await placeholder.edit_text(html, parse_mode="HTML")
            return
        except TelegramBadRequest as exc:
            if _not_modified(exc):
                return  # Same content already on screen — nothing to do.
            logger.warning(
                "Telegram HTML parse failed, retrying as plain text: %s", exc
            )
        except Exception as exc:
            logger.warning("edit_text failed, retrying as plain text: %s", exc)
    else:
        try:
            await placeholder.delete()
            await _send_long(msg, html, parse_mode="HTML")
            return
        except TelegramBadRequest as exc:
            if _not_modified(exc):
                return
            logger.warning("Telegram HTML parse failed on long send: %s", exc)
        except Exception as exc:
            logger.warning("Long HTML send failed, retrying as plain text: %s", exc)

    # Fallback: plain text, no parse_mode.
    plain_body = plain or "..."
    if len(plain_body) <= _TG_MAX_LEN:
        try:
            await placeholder.edit_text(plain_body)
            return
        except TelegramBadRequest as exc:
            if _not_modified(exc):
                return  # Same content already on screen.
        except Exception:
            pass
        try:
            await msg.answer(plain_body)
        except Exception as exc:
            logger.error("Telegram plain-text fallback also failed: %s", exc)
    else:
        try:
            await placeholder.delete()
        except Exception:
            pass
        try:
            await _send_long(msg, plain_body)
        except Exception as exc:
            logger.error("Telegram long plain-text fallback failed: %s", exc)


async def _send_long(msg, text: str, parse_mode: str | None = None) -> None:
    """Send a potentially long message, splitting if it exceeds Telegram's limit."""
    for part in _split_message(text):
        await msg.answer(part, parse_mode=parse_mode)


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


_LOCAL_PROVIDERS = {"ollama", "llamacpp", "local"}
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

    lines: list[str] = [f"<b>📊 Uso — {day_label}</b>"]

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
                f"\n💬 <b>Esta sesión</b>"
                f"\n  Tokens: {_fmt_num(s_prompt)} prompt · {_fmt_num(s_comp)} completion"
                f"\n  Coste: {cost_str}"
                f"\n  Llamadas: {len(sess_records)}"
            )

    # --- Individual calls for today, grouped by (provider, model) ---
    _MAX_CALLS_SHOWN = 5

    if not day_records:
        lines.append("\nSin actividad registrada.")
        return "\n".join(lines)

    # Group preserving insertion order; display calls oldest→newest
    groups: dict[tuple, list] = defaultdict(list)
    for r in reversed(day_records):
        groups[(r.provider, r.model)].append(r)

    for (provider, model), calls in groups.items():
        emoji = _provider_emoji(provider)
        model_label = _escape_html(_model_short(model))
        shown = list(reversed(calls[-_MAX_CALLS_SHOWN:]))
        hidden = len(calls) - _MAX_CALLS_SHOWN
        label_extra = f" (últimas {_MAX_CALLS_SHOWN})" if hidden > 0 else ""
        lines.append(f"\n{emoji} <b>{model_label}</b>{label_extra}")

        for r in shown:
            ts_local = datetime.fromisoformat(r.timestamp) + utc_offset
            hhmm = ts_local.strftime("%H:%M")
            tok = (
                f"{_fmt_num(r.prompt_tokens)} → {_fmt_num(r.completion_tokens)} tokens"
            )
            cost = f"${r.estimated_cost:.4f}" if r.estimated_cost > 0 else "$0"
            speed = f"{r.speed_tps:.1f} tps".replace(".", ",") if r.speed_tps else "—"
            finish = r.finish_reason or "—"
            lines.append(f"  <code>{hhmm}</code>  {tok} | {cost} | {speed} | {finish}")

        total_tok = sum(r.prompt_tokens + r.completion_tokens for r in calls)
        lines.append(f"  ─ {len(calls)} llamadas · {_fmt_num(total_tok)} tok")

    # --- Footer: day vs week totals ---
    week_records = tracker.get_records(period="week")

    def _totals_from_records(recs):
        prompt = sum(r.prompt_tokens for r in recs)
        completion = sum(r.completion_tokens for r in recs)
        cost = sum(r.estimated_cost for r in recs)
        tool = sum(1 for r in recs if r.finish_reason == "tool_calls")
        return prompt, completion, len(recs), tool, cost

    day_prompt, day_comp, day_calls, day_tool, day_cost = _totals_from_records(
        day_records
    )
    week_prompt, week_comp, week_calls, week_tool, week_cost = _totals_from_records(
        week_records
    )

    def _footer_line(
        label: str, prompt: int, comp: int, calls: int, tool: int, cost: float
    ) -> str:
        tok_part = f"{_fmt_num(prompt)} → {_fmt_num(comp)} tokens"
        calls_part = f"{calls} llamadas ({tool} tool_call)"
        cost_part = f"${cost:.4f}" if cost > 0 else "$0"
        return f"{label}   {tok_part} · {calls_part} · {cost_part}"

    lines.append("\n──────────────")
    lines.append(
        _footer_line("Hoy   ", day_prompt, day_comp, day_calls, day_tool, day_cost)
    )
    lines.append(
        _footer_line("Semana", week_prompt, week_comp, week_calls, week_tool, week_cost)
    )

    # Media/día (semana)
    week_days = 7
    if week_calls > 0:
        avg_prompt = week_prompt // week_days
        avg_comp = week_comp // week_days
        avg_calls = week_calls // week_days
        avg_tool = week_tool // week_days
        avg_cost = week_cost / week_days
        lines.append(
            _footer_line(
                "Promedio diario ", avg_prompt, avg_comp, avg_calls, avg_tool, avg_cost
            )
        )

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
