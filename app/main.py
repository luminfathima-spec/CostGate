"""
FastAPI entrypoint for CostGate service.
Implements Phase 1 Baseline Gateway: /ask, /health, /stats, with free-tier guardrails.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.schemas import AskRequest, AskResponse, StatsResponse, HealthResponse, ErrorResponse
from app.llm_client import GeminiClient, ModelRetiredError
from app.cost import compute_cost
from app.limits import RateLimiter, DailyCallCounter, RateLimitError, DailyLimitExceededError
from app.logger import RequestLogger

logger = logging.getLogger("costgate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Global service instances
_rate_limiter = RateLimiter()
_daily_counter = DailyCallCounter()
_request_logger = RequestLogger()
_llm_client = GeminiClient()


def log_configured_models(settings: Settings) -> str:
    """Format and log configured model identifiers without sensitive credentials."""
    log_msg = (
        f"CostGate models configured: "
        f"small='{settings.model_small}', "
        f"medium='{settings.model_medium}', "
        f"large='{settings.model_large}', "
        f"embedding='{settings.embedding_model}', "
        f"enable_medium={settings.enable_medium}"
    )
    logger.info(log_msg)
    return log_msg


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup lifecycle manager: verifies settings and prints model identifiers."""
    settings = get_settings()
    log_configured_models(settings)
    _request_logger.init_db()
    yield


app = FastAPI(
    title="CostGate API",
    description="Cost-aware LLM Gateway with semantic caching and free-tier protections.",
    version="0.1.0",
    lifespan=lifespan,
)


# Exception handlers for clear JSON error responses
@app.exception_handler(ModelRetiredError)
async def handle_model_retired(request: Request, exc: ModelRetiredError):
    return JSONResponse(
        status_code=status.HTTP_410_GONE,
        content={"error": "ModelRetiredError", "detail": str(exc)},
    )


@app.exception_handler(RateLimitError)
async def handle_rate_limit(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "RateLimitError", "detail": str(exc)},
    )


@app.exception_handler(DailyLimitExceededError)
async def handle_daily_limit(request: Request, exc: DailyLimitExceededError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "DailyLimitExceededError", "detail": str(exc)},
    )


@app.exception_handler(TimeoutError)
async def handle_timeout(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "TimeoutError", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Retrieve aggregate performance and reference cost statistics."""
    stats = _request_logger.get_stats()
    return StatsResponse(
        total_requests=stats["total_requests"],
        total_reference_cost=stats["total_reference_cost"],
        average_latency_ms=stats["average_latency_ms"],
        calls_used_today=stats["calls_used_today"],
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """
    Query endpoint for baseline gateway.
    Phase 1 Rule: Always routes to MODEL_LARGE (gemini-3.8-flash) unless force_model is given.
    Guarded by RPM rate limiter and persistent daily call counter.
    """
    settings = get_settings()

    # Model selection for Phase 1: MODEL_LARGE unless overridden
    target_model = payload.force_model.strip() if payload.force_model else settings.model_large

    # 1. Free-tier Guard: Check RPM limit
    try:
        _rate_limiter.check_and_acquire()
    except RateLimitError as re:
        logger.warning("Rate limit check failed: %s", re)
        raise re

    # 2. Free-tier Guard: Check persistent daily quota
    try:
        _daily_counter.check_and_increment(target_model, limit=settings.daily_call_limit)
    except DailyLimitExceededError as de:
        logger.warning("Daily call limit exceeded for %s: %s", target_model, de)
        raise de

    # 3. Call LLM Client (includes timeout, retries on 429/5xx, retirement guard)
    try:
        gen_result = _llm_client.generate(model=target_model, prompt=payload.question)
    except (ModelRetiredError, RateLimitError, TimeoutError):
        raise
    except Exception as err:
        logger.error("Inference failure for model '%s': %s", target_model, err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream provider error: {str(err)}",
        )

    answer = gen_result["text"]
    in_tokens = gen_result["input_tokens"]
    out_tokens = gen_result["output_tokens"]
    think_tokens = gen_result["thinking_tokens"]
    latency_ms = gen_result["latency_ms"]

    # 4. Calculate reference cost (including thinking tokens as output)
    cost = compute_cost(
        model=target_model,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        thinking_tokens=think_tokens,
        settings=settings,
    )

    # 5. Log request to SQLite audit trail
    _request_logger.log_request(
        question=payload.question,
        route="baseline_large" if not payload.force_model else "forced_model",
        model=target_model,
        cache_hit=False,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        thinking_tokens=think_tokens,
        cost=cost,
        latency_ms=latency_ms,
        escalated=False,
        quality_score=None,
        price_date=settings.price_date,
    )

    return AskResponse(
        answer=answer,
        model=target_model,
        cost=cost,
        latency_ms=latency_ms,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        thinking_tokens=think_tokens,
    )
