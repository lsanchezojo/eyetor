from pathlib import Path

import pytest

import eyetor.config as config_module
from eyetor.config import _deep_merge, load_config


def test_deep_merge_dicts_recursive_lists_and_scalars_replace():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1, "c": {"keep": True}}
    override = {"a": {"y": [3], "z": 2}, "b": None}
    assert _deep_merge(base, override) == {
        "a": {"x": 1, "y": [3], "z": 2},
        "b": None,
        "c": {"keep": True},
    }


@pytest.fixture
def layers(tmp_path: Path, monkeypatch):
    default = tmp_path / "default.yaml"
    local = tmp_path / "local.yaml"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", default)
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)
    return default, local


def test_local_overlays_default(layers):
    default, local = layers
    default.write_text(
        "fallback:\n  fallback_chain: [a, b]\nlog_level: INFO\n"
        "providers:\n  a: {type: llamacpp, base_url: 'http://x', model: m}\n"
    )
    local.write_text(
        "fallback:\n  fallback_chain: [c]\n"
        "providers:\n  c: {type: llamacpp, base_url: 'http://y', model: m}\n"
    )
    cfg = load_config(local_path=local)
    assert cfg.fallback.fallback_chain == ["c"]
    assert set(cfg.providers) == {"a", "c"}
    assert cfg.log_level == "INFO"


def test_missing_local_uses_default_only(layers):
    default, local = layers
    default.write_text("log_level: DEBUG\n")
    assert load_config(local_path=local).log_level == "DEBUG"


def test_explicit_path_is_used_alone(layers, tmp_path):
    default, local = layers
    default.write_text("log_level: DEBUG\n")
    local.write_text("log_level: WARNING\n")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("chatlog_enabled: false\n")
    cfg = load_config(explicit, local_path=local)
    assert cfg.log_level == "INFO"  # model default, not DEBUG/WARNING
    assert cfg.chatlog_enabled is False
