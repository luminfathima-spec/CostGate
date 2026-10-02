"""
Pydantic schemas for request/response payloads and internal data structures.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user query payload."""
    question: str = Field(..., min_length=1, description="The user query or prompt")


class ChatResponse(BaseModel):
    """Outgoing response payload from the gateway."""
    answer: str = Field(..., description="The generated or cached answer")
    route_taken: str = Field(..., description="Route taken: 'cache', 'small', 'medium', or 'large'")
    model_used: str = Field(..., description="Identifier of the model used, or 'cache'")
    cache_hit: bool = Field(default=False, description="Whether the answer was retrieved from cache")
    input_tokens: int = Field(default=0, description="Number of prompt/input tokens")
    output_tokens: int = Field(default=0, description="Number of candidate/output tokens")
    cost: float = Field(default=0.0, description="Total estimated cost in USD")
    latency_ms: float = Field(default=0.0, description="Latency in milliseconds")
    escalated: bool = Field(default=False, description="Whether model tier escalation was triggered")
    quality_score: Optional[float] = Field(default=None, description="Quality assessment score")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
