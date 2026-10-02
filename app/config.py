"""
Configuration loader for CostGate.

Loads environment variables from .env using pydantic-settings.
Guarantees that sensitive credentials like GEMINI_API_KEY are never leaked or printed.
"""

from functools import lru_cache
from typing import Any, Dict
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_NAMES = {"MODEL_SMALL", "MODEL_MEDIUM", "MODEL_LARGE", "EMBEDDING_MODEL"}


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

    # Required Model Tier Identifiers (No defaults, fail if set to placeholder)
    model_small: str = Field(
        ...,
        description="Identifier for the small model tier (e.g., gemini-3.5-flash-lite)",
    )
    model_medium: str = Field(
        ...,
        description="Identifier for the medium model tier (e.g., gemini-3.7-flash)",
    )
    model_large: str = Field(
        ...,
        description="Identifier for the large model tier (e.g., gemini-3.8-flash)",
    )
    embedding_model: str = Field(
        ...,
        description="Identifier for the embedding model (e.g., gemini-embedding-2)",
    )

    # Gateway Routing Flags
    enable_medium: bool = Field(
        default=False,
        description="When false, routes small -> large directly since medium and large share identical pricing",
    )

    # Price Snapshot Date
    price_date: str = Field(
        default="2026-10-02",
        description="Date of pricing reference snapshot",
    )

    # Required Pricing per 1,000,000 tokens (USD) - No silent defaults
    price_input_small: float = Field(
        ..., ge=0.0, description="USD per 1M input tokens for small tier"
    )
    price_output_small: float = Field(
        ..., ge=0.0, description="USD per 1M output tokens for small tier"
    )

    price_input_medium: float = Field(
        ..., ge=0.0, description="USD per 1M input tokens for medium tier"
    )
    price_output_medium: float = Field(
        ..., ge=0.0, description="USD per 1M output tokens for medium tier"
    )

    price_input_large: float = Field(
        ..., ge=0.0, description="USD per 1M input tokens for large tier"
    )
    price_output_large: float = Field(
        ..., ge=0.0, description="USD per 1M output tokens for large tier"
    )

    price_embedding: float = Field(
        ..., ge=0.0, description="USD per 1M tokens for embedding model lookup"
    )

    # Routing Confidence Thresholds
    route_high_conf: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="High confidence threshold for direct routing",
    )
    route_low_conf: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Low confidence threshold for routing escalation",
    )

    # Semantic Cache Configuration
    cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cache hits (0.0 to 1.0)",
    )
    cache_ttl_hours: int = Field(
        default=72,
        gt=0,
        description="Cache entry time-to-live in hours",
    )

    # Quality Gating (0.0 to 1.0 scale)
    # Note: Any 1-5 confidence rating must be divided by 5 before comparing against quality_threshold.
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Quality score threshold (0.0 to 1.0). Any 1-5 confidence rating must be divided by 5 before comparing.",
    )

    # Resiliency: Timeouts & Retries
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_base_delay: float = Field(default=1.0, ge=0.0)

    # Free-Tier Guardrails
    daily_call_limit: int = Field(
        default=1200,
        gt=0,
        description="Daily request limit to protect free tier quota",
    )
    rpm_limit: int = Field(
        default=12,
        gt=0,
        description="Requests per minute safety limit for free tier",
    )

    # Storage & Persistence
    db_path: str = Field(default="costgate.db", description="Path to SQLite database")

    # Server Configuration
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    environment: str = Field(default="development")

    @field_validator("model_small", "model_medium", "model_large", "embedding_model")
    @classmethod
    def validate_not_placeholder(cls, v: str, info) -> str:
        """Fail with a clear error message if a model name still equals placeholder text."""
        if v.strip().upper() in PLACEHOLDER_NAMES:
            raise ValueError(
                f"Config error: '{info.field_name}' is still set to placeholder '{v}'. "
                "Please configure an actual Gemini model identifier in .env before starting the gateway."
            )
        return v.strip()

    @field_validator("cache_similarity_threshold", "quality_threshold", "route_high_conf", "route_low_conf")
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_routing_threshold_order(self) -> "Settings":
        """Validate that route_low_conf is strictly less than route_high_conf."""
        if self.route_low_conf >= self.route_high_conf:
            raise ValueError(
                f"Config error: route_low_conf ({self.route_low_conf}) must be strictly less than "
                f"route_high_conf ({self.route_high_conf})."
            )
        return self

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
