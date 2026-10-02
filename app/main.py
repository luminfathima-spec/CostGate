"""
FastAPI entrypoint for CostGate service.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from app.config import Settings, get_settings
from app.schemas import HealthResponse

# Configure structured application logger
logger = logging.getLogger("costgate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def log_configured_models(settings: Settings) -> str:
    """Format and log configured model identifiers without sensitive credentials."""
    log_msg = (
        f"CostGate models configured: "
        f"small='{settings.model_small}', "
        f"medium='{settings.model_medium}', "
        f"large='{settings.model_large}', "
        f"embedding='{settings.embedding_model}'"
    )
    logger.info(log_msg)
    return log_msg


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager for FastAPI."""
    settings = get_settings()
    log_configured_models(settings)
    yield


app = FastAPI(
    title="CostGate API",
    description="Cost-aware LLM Gateway with semantic caching and quality-based escalation.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")
