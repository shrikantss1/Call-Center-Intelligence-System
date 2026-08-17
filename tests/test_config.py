import pytest
from dataclasses import FrozenInstanceError

from src.utils.config import load_config, Config


def test_load_config_requires_llm_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError):
        load_config()


def test_load_config_parses_types(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.85")
    monkeypatch.setenv("LOW_CONFIDENCE_HALT_RATIO", "0.25")
    monkeypatch.setenv("MAX_RETRIES_PER_NODE", "5")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")

    cfg = load_config()

    assert isinstance(cfg, Config)
    assert cfg.llm_provider == "openai"
    assert cfg.openai_api_key == "sk-test"
    assert abs(cfg.confidence_threshold - 0.85) < 1e-9
    assert abs(cfg.low_confidence_halt_ratio - 0.25) < 1e-9
    assert cfg.max_retries_per_node == 5
    assert abs(cfg.llm_timeout_seconds - 120.0) < 1e-9


def test_optional_defaults_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("MAX_RETRIES_PER_NODE", raising=False)

    cfg = load_config()

    assert cfg.openai_api_key is None
    assert cfg.confidence_threshold is None
    assert cfg.max_retries_per_node is None


def test_invalid_int_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MAX_RETRIES_PER_NODE", "notint")
    with pytest.raises(ValueError):
        load_config()


def test_config_is_frozen(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    cfg = load_config()
    with pytest.raises(FrozenInstanceError):
        cfg.llm_provider = "other"
