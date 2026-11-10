"""System prompt line naming the LLM (primary + fallbacks)."""

from __future__ import annotations

import httpx

from eyetor.app import describe_models, model_display_name
from eyetor.config import ProviderConfig, VectorConfig


def _cfg() -> VectorConfig:
    return VectorConfig(
        providers={
            "bolt": ProviderConfig(type="llamacpp", base_url="http://x:8086/v1", model="default"),
            "openrouter": ProviderConfig(type="openrouter", base_url="https://o", model="vendor/big-model"),
            "named": ProviderConfig(type="gemini", base_url="https://g", model="g-1", display_name="Gemini Uno"),
        }
    )


def test_llamacpp_default_model_asks_the_server(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://x:8086/v1/models"
        return httpx.Response(200, json={"data": [{"id": "/m/Nemotron-Bolt-Q4_K_M.gguf"}]},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert model_display_name(_cfg().providers["bolt"], "bolt") == "Nemotron-Bolt-Q4_K_M"


def test_server_down_falls_back_to_provider_key(monkeypatch):
    def down(url, timeout):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", down)
    assert describe_models(_cfg(), ["missing", "bolt", "openrouter"]) == "Eyetor is powered by bolt"


def test_display_name_and_model_id():
    assert describe_models(_cfg(), ["named"]) == "Eyetor is powered by Gemini Uno"
    assert describe_models(_cfg(), ["openrouter"]) == "Eyetor is powered by vendor/big-model"


def test_no_chain_no_line():
    assert describe_models(_cfg(), []) == ""
