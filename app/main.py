"""
FastAPI entrypoint for CostGate service.
"""

from fastapi import FastAPI
from app.config import get_settings
from app.schemas import HealthResponse

settings = get_settings()

app = FastAPI(
    title="CostGate API",
    description="Cost-aware LLM Gateway with semantic caching and quality-based escalation.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")
