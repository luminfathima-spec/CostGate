"""
Pydantic schemas for request/response payloads and internal data structures.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for POST /ask query endpoint."""
    question: str = Field(..., min_length=1, description="User question or prompt")
    force_model: Optional[str] = Field(
        default=None,
        description="Optional model override (e.g. gemini-3.5-flash-lite)",
    )


class AskResponse(BaseModel):
    """Response payload returned by POST /ask."""
    answer: str = Field(..., description="Model response text")
    model: str = Field(..., description="Model identifier used")
    cost: float = Field(..., description="Calculated reference cost in USD")
    latency_ms: float = Field(..., description="Round-trip latency in milliseconds")
    input_tokens: int = Field(default=0, description="Prompt tokens")
    output_tokens: int = Field(default=0, description="Candidate tokens")
    thinking_tokens: int = Field(default=0, description="Thinking/reasoning tokens")


class StatsResponse(BaseModel):
    """Aggregate metrics returned by GET /stats."""
    total_requests: int
    total_reference_cost: float
    average_latency_ms: float
    calls_used_today: int


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = "ok"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Structured error payload."""
    error: str
    detail: Optional[str] = None
