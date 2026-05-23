"""Telegram image caption date extraction."""

from __future__ import annotations

from eyetor.channels.telegram import _extract_caption_date


def test_extracts_iso_caption_date() -> None:
    assert _extract_caption_date("ticket alcampo 2026-05-21") == "2026-05-21"


def test_extracts_spanish_numeric_caption_date() -> None:
    assert _extract_caption_date("fecha 21/05/2026") == "2026-05-21"


def test_extracts_textual_spanish_caption_date() -> None:
    assert _extract_caption_date("compra del 21 de mayo de 2026") == "2026-05-21"


def test_ignores_incomplete_caption_date() -> None:
    assert _extract_caption_date("ticket del 21/05") is None
