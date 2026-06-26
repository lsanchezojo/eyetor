"""Tests for voice-note transcription config defaults and YAML loading."""

from __future__ import annotations

from eyetor.config import TranscriptionConfig, VectorConfig


def test_transcription_defaults_local_spanish_medium():
    cfg = TranscriptionConfig()
    assert cfg.enabled is True
    assert cfg.backend == "local"
    assert cfg.model == "medium"
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.language == "es"
    assert cfg.beam_size == 5
    assert cfg.base_url is None
    assert cfg.api_key is None


def test_vector_config_includes_transcription_by_default():
    cfg = VectorConfig()
    assert isinstance(cfg.transcription, TranscriptionConfig)
    assert cfg.transcription.model == "medium"


def test_transcription_overrides_from_mapping():
    cfg = VectorConfig(
        transcription={"backend": "api", "model": "large-v3", "language": None}
    )
    assert cfg.transcription.backend == "api"
    assert cfg.transcription.model == "large-v3"
    assert cfg.transcription.language is None
    # untouched fields keep their defaults
    assert cfg.transcription.compute_type == "int8"
