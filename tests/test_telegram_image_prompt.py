"""Telegram image prompt construction tests."""

from __future__ import annotations

from pathlib import Path

from eyetor.channels.telegram import _build_image_prompt


def test_image_prompt_exposes_attachment_path_for_registered_tools() -> None:
    img_path = Path("/tmp/eyetor/attachment.jpg")
    prompt = _build_image_prompt(
        "Tipo: ticket de compra\nTotal: 12,30",
        "",
        img_path,
    )

    assert str(img_path) in prompt
    assert "herramienta registrada" in prompt
    assert "local_attachment_path" in prompt
    assert "No respondas solo con" in prompt


def test_image_prompt_preserves_caption_and_vision_analysis() -> None:
    img_path = Path("/tmp/eyetor/doc.png")
    prompt = _build_image_prompt(
        "Tipo: documento",
        "solo analiza esto",
        img_path,
    )

    assert "solo analiza esto" in prompt
    assert "Tipo: documento" in prompt
    assert str(img_path) in prompt
    assert "respuesta al usuario debe ser humana" in prompt


def test_image_prompt_does_not_reference_specific_skills_or_domains() -> None:
    prompt = _build_image_prompt(
        "Tipo: ticket de compra",
        "",
        Path("/tmp/eyetor/ticket.jpg"),
    ).lower()

    assert "skill_" not in prompt
    assert "grocery" not in prompt
    assert "filesystem" not in prompt
