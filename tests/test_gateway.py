"""
Unit tests for the CostGate API endpoints (/ask, /health, /stats) with mocked Gemini calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app, _llm_client, _rate_limiter, _daily_counter, _request_logger
from app.llm_client import ModelRetiredError, GeminiClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient with a fresh isolated SQLite database."""
    test_db = str(tmp_path / "gateway_test.db")
    monkeypatch.setattr(_request_logger, "db_path", test_db)
    monkeypatch.setattr(_daily_counter, "db_path", test_db)
    _request_logger.init_db()
    _daily_counter._init_table()
    _rate_limiter.reset()
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health returns status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "0.1.0"}


def test_ask_routes_to_large_by_default(client, monkeypatch):
    """Verify /ask routes to MODEL_LARGE by default in Phase 1."""
    mock_gen = MagicMock(return_value={
        "text": "Paris is the capital of France.",
        "input_tokens": 10,
        "output_tokens": 8,
        "thinking_tokens": 20,
        "latency_ms": 150.0,
    })
    monkeypatch.setattr(_llm_client, "generate", mock_gen)

    resp = client.post("/ask", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"] == "Paris is the capital of France."
    assert data["model"] == "gemini-3.8-flash"
    assert data["input_tokens"] == 10
    assert data["output_tokens"] == 8
    assert data["thinking_tokens"] == 20
    assert data["cost"] > 0
    assert data["latency_ms"] == 150.0

    mock_gen.assert_called_once_with(model="gemini-3.8-flash", prompt="What is the capital of France?")


def test_ask_force_model(client, monkeypatch):
    """Verify force_model parameter overrides the default model selection."""
    mock_gen = MagicMock(return_value={
        "text": "Answer from small model.",
        "input_tokens": 5,
        "output_tokens": 4,
        "thinking_tokens": 0,
        "latency_ms": 80.0,
    })
    monkeypatch.setattr(_llm_client, "generate", mock_gen)

    resp = client.post("/ask", json={
        "question": "Quick question",
        "force_model": "gemini-3.5-flash-lite",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "gemini-3.5-flash-lite"
    mock_gen.assert_called_once_with(model="gemini-3.5-flash-lite", prompt="Quick question")


def test_404_model_retired_no_retry(client, monkeypatch):
    """Verify that a 404 'no longer available' error immediately aborts and returns 410."""
    call_count = 0

    def mock_retire_call(model, prompt):
        nonlocal call_count
        call_count += 1
        raise ModelRetiredError("Configured model 'gemini-2.5-pro' is retired or no longer available (404 NOT_FOUND).")

    monkeypatch.setattr(_llm_client, "generate", mock_retire_call)

    resp = client.post("/ask", json={"question": "Test retirement"})
    assert resp.status_code == 410
    data = resp.json()
    assert data["error"] == "ModelRetiredError"
    assert "retired or no longer available" in data["detail"]
    assert call_count == 1  # Strictly 1 call, zero retries


def test_rate_limit_exceeded_returns_429(client, monkeypatch):
    """Verify that reaching RPM limit returns HTTP 429."""
    monkeypatch.setattr(_rate_limiter, "rpm_limit", 1)

    mock_gen = MagicMock(return_value={
        "text": "First answer",
        "input_tokens": 5,
        "output_tokens": 5,
        "thinking_tokens": 0,
        "latency_ms": 100.0,
    })
    monkeypatch.setattr(_llm_client, "generate", mock_gen)

    # First call succeeds
    resp1 = client.post("/ask", json={"question": "First call"})
    assert resp1.status_code == 200

    # Immediate second call must fail with 429
    resp2 = client.post("/ask", json={"question": "Second call"})
    assert resp2.status_code == 429
    assert "RateLimitError" in resp2.json()["error"]


def test_daily_call_limit_returns_429(client, monkeypatch):
    """Verify that exceeding daily quota returns HTTP 429."""
    mock_gen = MagicMock(return_value={
        "text": "Answer",
        "input_tokens": 5,
        "output_tokens": 5,
        "thinking_tokens": 0,
        "latency_ms": 100.0,
    })
    monkeypatch.setattr(_llm_client, "generate", mock_gen)

    # Set daily limit to 2
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "daily_call_limit", 2)

    # 1st call
    assert client.post("/ask", json={"question": "q1"}).status_code == 200
    # 2nd call
    assert client.post("/ask", json={"question": "q2"}).status_code == 200
    # 3rd call should hit daily limit
    resp3 = client.post("/ask", json={"question": "q3"})
    assert resp3.status_code == 429
    assert "DailyLimitExceededError" in resp3.json()["error"]


def test_stats_endpoint(client, monkeypatch):
    """Verify /stats computes total requests, cost, and latency."""
    mock_gen = MagicMock(return_value={
        "text": "Answer",
        "input_tokens": 1000,
        "output_tokens": 1000,
        "thinking_tokens": 0,
        "latency_ms": 200.0,
    })
    monkeypatch.setattr(_llm_client, "generate", mock_gen)

    client.post("/ask", json={"question": "Test query 1"})
    client.post("/ask", json={"question": "Test query 2"})

    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()

    assert stats["total_requests"] == 2
    assert stats["total_reference_cost"] > 0
    assert stats["average_latency_ms"] == 200.0
    assert stats["calls_used_today"] == 2
