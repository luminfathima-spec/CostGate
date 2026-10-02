"""
Unit tests for configuration loading, security protection, validation rules, and model retirement handling.
"""

import pytest
from pydantic import ValidationError
from app.config import Settings, get_settings
from app.main import log_configured_models
from app.llm_client import ModelRetiredError, check_for_model_retirement


@pytest.fixture
def valid_config_dict():
    """Fixture providing all required settings with valid values."""
    return {
        "gemini_api_key": "ai-secret-key-12345",
        "model_small": "gemini-3.5-flash-lite",
        "model_medium": "gemini-3.7-flash",
        "model_large": "gemini-3.8-flash",
        "embedding_model": "gemini-embedding-2",
        "enable_medium": False,
        "price_date": "2026-10-02",
        "price_input_small": 0.30,
        "price_output_small": 2.50,
        "price_input_medium": 0.75,
        "price_output_medium": 3.75,
        "price_input_large": 0.75,
        "price_output_large": 3.75,
        "price_embedding": 0.20,
    }


def test_valid_settings_and_defaults(valid_config_dict):
    """Verify that settings load with new defaults for cache, routing, quality, and host."""
    settings = Settings(_env_file=None, **valid_config_dict)

    # Required fields loaded correctly
    assert settings.model_small == "gemini-3.5-flash-lite"
    assert settings.model_medium == "gemini-3.7-flash"
    assert settings.model_large == "gemini-3.8-flash"
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.enable_medium is False
    assert settings.price_date == "2026-10-02"
    assert settings.price_embedding == 0.20

    # Defaults
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
        "price_embedding",
    ]:
        invalid_data = valid_config_dict.copy()
        del invalid_data[price_key]
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **invalid_data)
        assert price_key in str(exc_info.value)


def test_missing_model_names_raises_error(valid_config_dict):
    """Verify that missing model identifiers fails validation."""
    for model_key in ["model_small", "model_medium", "model_large", "embedding_model"]:
        invalid_data = valid_config_dict.copy()
        del invalid_data[model_key]
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **invalid_data)
        assert model_key in str(exc_info.value)


def test_placeholder_model_name_fails(valid_config_dict):
    """Verify that using placeholder model names fails with a clear error message."""
    for model_key, placeholder in [
        ("model_small", "MODEL_SMALL"),
        ("model_medium", "MODEL_MEDIUM"),
        ("model_large", "MODEL_LARGE"),
        ("embedding_model", "EMBEDDING_MODEL"),
    ]:
        invalid_data = valid_config_dict.copy()
        invalid_data[model_key] = placeholder
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **invalid_data)
        assert "is still set to placeholder" in str(exc_info.value)


def test_routing_confidence_order_validation(valid_config_dict):
    """Verify that route_low_conf must be strictly less than route_high_conf."""
    invalid_data = valid_config_dict.copy()
    invalid_data["route_low_conf"] = 0.80
    invalid_data["route_high_conf"] = 0.75

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **invalid_data)
    assert "route_low_conf" in str(exc_info.value)
    assert "must be strictly less than" in str(exc_info.value)


def test_api_key_masked_in_repr(valid_config_dict):
    """Verify that sensitive API key never leaks in str, repr, or safe_dict."""
    settings = Settings(_env_file=None, **valid_config_dict)
    secret = valid_config_dict["gemini_api_key"]

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert "***MASKED***" in repr(settings)
    assert settings.safe_dict()["gemini_api_key"] == "***MASKED***"


def test_startup_log_line(valid_config_dict):
    """Verify that startup log line prints model names and never the API key."""
    settings = Settings(_env_file=None, **valid_config_dict)
    log_output = log_configured_models(settings)

    assert "gemini-3.5-flash-lite" in log_output
    assert "gemini-3.7-flash" in log_output
    assert "gemini-3.8-flash" in log_output
    assert "gemini-embedding-2" in log_output
    assert valid_config_dict["gemini_api_key"] not in log_output


def test_model_retirement_handler():
    """Verify that 404 'no longer available' triggers ModelRetiredError without retrying."""
    retired_err = Exception("404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available to new users.")
    with pytest.raises(ModelRetiredError) as exc_info:
        check_for_model_retirement(retired_err, "gemini-2.5-flash")
    
    assert "is retired or no longer available (404 NOT_FOUND)" in str(exc_info.value)
    assert "Please update your model configuration in .env" in str(exc_info.value)

    # Standard non-retirement errors should not raise ModelRetiredError
    rate_limit_err = Exception("429 RESOURCE_EXHAUSTED. Rate limit exceeded.")
    check_for_model_retirement(rate_limit_err, "gemini-3.5-flash-lite")


def test_load_from_configured_env():
    """Verify that live Settings loads correctly from the project's actual .env file."""
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.model_small == "gemini-3.5-flash-lite"
    assert settings.model_medium == "gemini-3.7-flash"
    assert settings.model_large == "gemini-3.8-flash"
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.enable_medium is False
    assert settings.price_embedding == 0.20
    assert settings.price_date == "2026-10-02"
