"""Tests for the Telegram document-handler helpers (no aiogram needed).

The routing branch (image-as-file → vision; extractable → text; inline vs KB)
lives inside the ``on_document`` closure, but its decision inputs are the pure
helpers and constants exercised here.
"""

from __future__ import annotations

from eyetor.channels.telegram import (
    _IMAGE_SUFFIXES,
    _build_document_prompt,
    _build_kb_document_prompt,
    _format_caducidad,
)
from eyetor.knowledge.extractors import get_extractor


class TestImageRouting:
    def test_image_suffixes_route_to_vision(self) -> None:
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".heic"):
            assert suffix in _IMAGE_SUFFIXES

    def test_image_mime_decision(self) -> None:
        # Mirrors the handler's `mime.startswith("image/") or suffix in _IMAGE_SUFFIXES`.
        assert "image/png".startswith("image/")
        assert not "application/pdf".startswith("image/")

    def test_extractable_vs_unsupported(self) -> None:
        assert get_extractor(".pdf") is not None
        assert get_extractor(".txt") is not None
        assert get_extractor(".docx") is not None
        assert get_extractor(".zip") is None


class TestDocumentPrompt:
    def test_inline_with_caption_includes_text_and_request(self) -> None:
        p = _build_document_prompt(
            user_text="[Ana]: resúmelo",
            file_name="informe.pdf",
            text="CONTENIDO DEL INFORME",
            truncated=False,
        )
        assert "informe.pdf" in p
        assert "resúmelo" in p
        assert "CONTENIDO DEL INFORME" in p
        assert "truncado" not in p

    def test_inline_without_caption_asks_summary(self) -> None:
        p = _build_document_prompt(
            user_text="",
            file_name="notas.txt",
            text="texto",
            truncated=False,
        )
        assert "sin mensaje adicional" in p
        assert "Resume" in p

    def test_truncated_flag_adds_note(self) -> None:
        p = _build_document_prompt(
            user_text="",
            file_name="grande.txt",
            text="x" * 100,
            truncated=True,
        )
        assert "truncado" in p


class TestKbDocumentPrompt:
    def test_mentions_workspace_and_kb_search(self) -> None:
        p = _build_kb_document_prompt(
            user_text="¿cuál es el total?",
            file_name="cuentas.xlsx",
            workspace="tg-upload-99",
        )
        assert "tg-upload-99" in p
        assert "kb_search" in p
        assert "cuentas.xlsx" in p

    def test_without_caption_asks_summary(self) -> None:
        p = _build_kb_document_prompt(
            user_text="",
            file_name="doc.pdf",
            workspace="tg-upload-1",
        )
        assert "resumen" in p


class TestFormatCaducidad:
    def test_none_is_empty(self) -> None:
        assert _format_caducidad(None) == ""
        assert _format_caducidad("") == ""

    def test_iso_date_is_shortened(self) -> None:
        assert _format_caducidad("2026-07-08T12:34:56.789") == " (caduca el 2026-07-08)"
