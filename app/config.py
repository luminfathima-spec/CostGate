"""
Configuration loader for CostGate.

Loads environment variables from .env using pydantic-settings.
Guarantees that sensitive credentials like GEMINI_API_KEY are never leaked or printed.
"""

from functools import lru_cache
from typing import Any, Dict
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable bindings and security guards."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Keys & Secrets
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key. Never logged or exposed.",
    )

    # Model Tier Placeholders
    model_small: str = Field(
        default="MODEL_SMALL",
        description="Identifier for the small model tier (e.g., gemini-2.5-flash-lite)",
    )
    model_medium: str = Field(
        default="MODEL_MEDIUM",
        description="Identifier for the medium model tier (e.g., gemini-2.5-flash)",
    )
    model_large: str = Field(
        default="MODEL_LARGE",
        description="Identifier for the large model tier (e.g., gemini-2.5-pro)",
    )
    model_embedding: str = Field(
        default="text-embedding-004",
        description="Identifier for the embedding model used for semantic caching",
    )

    # Pricing per 1,000,000 tokens (USD)
    price_input_small: float = Field(default=0.075, ge=0.0)
    price_output_small: float = Field(default=0.30, ge=0.0)

    price_input_medium: float = Field(default=0.15, ge=0.0)
    price_output_medium: float = Field(default=0.60, ge=0.0)

    price_input_large: float = Field(default=1.25, ge=0.0)
    price_output_large: float = Field(default=5.00, ge=0.0)

    # Thresholds
    cache_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cache hits (0.0 to 1.0)",
    )
    quality_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Quality score threshold below which a response is escalated",
    )

    # Resiliency: Timeouts & Retries
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_base_delay: float = Field(default=1.0, ge=0.0)

    # Storage & Persistence
    db_path: str = Field(default="costgate.db", description="Path to SQLite database")

    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    environment: str = Field(default="development")

    @field_validator("cache_similarity_threshold", "quality_threshold")
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v

    def safe_dict(self) -> Dict[str, Any]:
        """Return dictionary representation with sensitive fields masked."""
        data = self.model_dump()
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = "***MASKED***"
        else:
            data["gemini_api_key"] = "***NOT_SET***"
        return data

    def __repr__(self) -> str:
        """Prevent API key from leaking in string/repr output."""
        masked_fields = ", ".join(f"{k}={v!r}" for k, v in self.safe_dict().items())
        return f"Settings({masked_fields})"

    def __str__(self) -> str:
        return self.__repr__()


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
