"""Telegram image prompt helpers."""

from __future__ import annotations

from pathlib import Path

from eyetor.channels.telegram import _build_image_prompt, _extract_caption_date


def test_extracts_iso_caption_date() -> None:
    assert _extract_caption_date("documento alcampo 2026-05-21") == "2026-05-21"


def test_extracts_spanish_numeric_caption_date() -> None:
    assert _extract_caption_date("fecha 21/05/2026") == "2026-05-21"


def test_extracts_textual_spanish_caption_date() -> None:
    assert _extract_caption_date("compra del 21 de mayo de 2026") == "2026-05-21"


def test_ignores_incomplete_caption_date() -> None:
    assert _extract_caption_date("documento del 21/05") is None


def test_image_prompt_is_tool_agnostic_but_keeps_context() -> None:
    text = _build_image_prompt(
        user_text="registralo con fecha 2026-05-21",
        description=(
            "Tipo: documento. Texto visible: Supermercado Ejemplo. "
            "Lineas: Pan 1,00 EUR; Leche 1,25 EUR. Total 2,25 EUR."
        ),
        img_path=Path("/home/haziel/.eyetor/images/123_1.jpg"),
        caption_date="2026-05-21",
    )

    assert "shopping_receipt_add" not in text
    assert "receipt.py" not in text
    assert "Imagen guardada en: /home/haziel/.eyetor/images/123_1.jpg" in text
    assert "Fecha completa detectada en el caption: 2026-05-21" in text
    assert "Pan 1,00 EUR" in text
    assert "Si una herramienta disponible encaja" in text


def test_telegram_channel_does_not_reference_shopping_tools() -> None:
    source = Path("src/eyetor/channels/telegram.py").read_text(encoding="utf-8")

    assert "shopping_receipt_add" not in source
    assert "receipt.py" not in source
