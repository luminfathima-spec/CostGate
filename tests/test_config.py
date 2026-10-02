"""
Unit tests for configuration loading and API key protection.
"""

import os
import pytest
from app.config import Settings


def test_default_settings():
    """Verify that default settings load with expected placeholders and values."""
    settings = Settings(gemini_api_key="")
    assert settings.model_small == "MODEL_SMALL"
    assert settings.model_medium == "MODEL_MEDIUM"
    assert settings.model_large == "MODEL_LARGE"
    assert settings.price_input_small >= 0.0
    assert settings.price_output_small >= 0.0
    assert 0.0 <= settings.cache_similarity_threshold <= 1.0
    assert 0.0 <= settings.quality_threshold <= 1.0
    assert settings.llm_timeout_seconds > 0
    assert settings.llm_max_retries >= 0


def test_api_key_never_leaks_in_repr():
    """Verify that GEMINI_API_KEY is masked in repr and str representations."""
    secret = "secret-gemini-key-123456789"
    settings = Settings(gemini_api_key=secret)

    repr_str = repr(settings)
    str_val = str(settings)

    # Must NOT contain raw secret
    assert secret not in repr_str
    assert secret not in str_val
    assert "***MASKED***" in repr_str
    assert "***MASKED***" in str_val

    # safe_dict must also mask
    safe_data = settings.safe_dict()
    assert safe_data["gemini_api_key"] == "***MASKED***"


def test_threshold_validation():
    """Verify invalid threshold values trigger validation errors."""
    with pytest.raises(ValueError):
        Settings(cache_similarity_threshold=1.5)

    with pytest.raises(ValueError):
        Settings(quality_threshold=-0.1)


def test_env_override(monkeypatch):
    """Verify that environment variables properly override settings."""
    monkeypatch.setenv("MODEL_SMALL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("PRICE_INPUT_SMALL", "0.05")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.90")

    settings = Settings()
    assert settings.model_small == "gemini-2.5-flash-lite"
    assert settings.price_input_small == 0.05
    assert settings.cache_similarity_threshold == 0.90
