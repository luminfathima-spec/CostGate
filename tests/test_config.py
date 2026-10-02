"""
Unit tests for configuration loading, security protection, and validation rules.
"""

import pytest
from pydantic import ValidationError
from app.config import Settings
from app.main import log_configured_models


@pytest.fixture
def valid_config_dict():
    """Fixture providing all required settings with valid values."""
    return {
        "gemini_api_key": "ai-secret-key-12345",
        "model_small": "gemini-2.5-flash-lite",
        "model_medium": "gemini-2.5-flash",
        "model_large": "gemini-2.5-pro",
        "embedding_model": "text-embedding-004",
        "price_input_small": 0.075,
        "price_output_small": 0.30,
        "price_input_medium": 0.15,
        "price_output_medium": 0.60,
        "price_input_large": 1.25,
        "price_output_large": 5.00,
    }


def test_valid_settings_and_new_defaults(valid_config_dict):
    """Verify that settings load with new defaults for cache, routing, quality, and host."""
    settings = Settings(**valid_config_dict)

    # Required fields loaded correctly
    assert settings.model_small == "gemini-2.5-flash-lite"
    assert settings.model_medium == "gemini-2.5-flash"
    assert settings.model_large == "gemini-2.5-pro"
    assert settings.embedding_model == "text-embedding-004"
    assert settings.price_input_small == 0.075
    assert settings.price_output_large == 5.00

    # New defaults
    assert settings.cache_similarity_threshold == 0.92
    assert settings.cache_ttl_hours == 72
    assert settings.route_high_conf == 0.75
    assert settings.route_low_conf == 0.40
    assert settings.quality_threshold == 0.8
    assert settings.host == "127.0.0.1"


def test_missing_pricing_raises_error(valid_config_dict):
    """Verify that missing any price field fails without silent default fallback."""
    for price_key in [
        "price_input_small",
        "price_output_small",
        "price_input_medium",
        "price_output_medium",
        "price_input_large",
        "price_output_large",
    ]:
        invalid_data = valid_config_dict.copy()
        del invalid_data[price_key]
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_data)
        assert price_key in str(exc_info.value)


def test_missing_model_names_raises_error(valid_config_dict):
    """Verify that missing model identifiers fails validation."""
    for model_key in ["model_small", "model_medium", "model_large", "embedding_model"]:
        invalid_data = valid_config_dict.copy()
        del invalid_data[model_key]
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_data)
        assert model_key in str(exc_info.value)


def test_placeholder_model_name_fails(valid_config_dict):
    """Verify that using placeholder model names fails with a clear error message."""
    for model_key, placeholder in [
        ("model_small", "MODEL_SMALL"),
        ("model_medium", "MODEL_MEDIUM"),
        ("model_large", "MODEL_LARGE"),
    ]:
        invalid_data = valid_config_dict.copy()
        invalid_data[model_key] = placeholder
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_data)
        assert "is still set to placeholder" in str(exc_info.value)


def test_routing_confidence_order_validation(valid_config_dict):
    """Verify that route_low_conf must be strictly less than route_high_conf."""
    # When route_low_conf >= route_high_conf
    invalid_data = valid_config_dict.copy()
    invalid_data["route_low_conf"] = 0.80
    invalid_data["route_high_conf"] = 0.75

    with pytest.raises(ValidationError) as exc_info:
        Settings(**invalid_data)
    assert "route_low_conf" in str(exc_info.value)
    assert "must be strictly less than" in str(exc_info.value)

    # Equal values should also fail
    invalid_data["route_low_conf"] = 0.50
    invalid_data["route_high_conf"] = 0.50
    with pytest.raises(ValidationError):
        Settings(**invalid_data)


def test_quality_and_cache_threshold_range(valid_config_dict):
    """Verify that thresholds outside [0.0, 1.0] are rejected."""
    # Quality threshold > 1.0
    data = valid_config_dict.copy()
    data["quality_threshold"] = 1.2
    with pytest.raises(ValidationError):
        Settings(**data)

    # Cache similarity < 0.0
    data = valid_config_dict.copy()
    data["cache_similarity_threshold"] = -0.05
    with pytest.raises(ValidationError):
        Settings(**data)


def test_api_key_masked_in_repr(valid_config_dict):
    """Verify that sensitive API key never leaks in str, repr, or safe_dict."""
    settings = Settings(**valid_config_dict)
    secret = valid_config_dict["gemini_api_key"]

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert "***MASKED***" in repr(settings)
    assert settings.safe_dict()["gemini_api_key"] == "***MASKED***"


def test_startup_log_line(valid_config_dict):
    """Verify that startup log line prints model names and never the API key."""
    settings = Settings(**valid_config_dict)
    log_output = log_configured_models(settings)

    assert "gemini-2.5-flash-lite" in log_output
    assert "gemini-2.5-flash" in log_output
    assert "gemini-2.5-pro" in log_output
    assert "text-embedding-004" in log_output
    assert valid_config_dict["gemini_api_key"] not in log_output
