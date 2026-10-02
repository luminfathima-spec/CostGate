"""
Unit tests for free-tier rate limits and persistent daily call counter with date reset.
"""

import os
import pytest
from app.limits import (
    RateLimiter,
    DailyCallCounter,
    RateLimitError,
    DailyLimitExceededError,
)


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    db_file = str(tmp_path / "test_limits.db")
    return db_file


def test_daily_call_counter_increment_and_persistence(temp_db):
    """Verify call counter increments and persists in SQLite."""
    counter = DailyCallCounter(db_path=temp_db)
    model = "gemini-3.8-flash"
    date_today = "2026-10-02"

    assert counter.get_calls_today(model, date_str=date_today) == 0

    # Increment
    new_count = counter.increment(model, date_str=date_today)
    assert new_count == 1
    assert counter.get_calls_today(model, date_str=date_today) == 1

    # Second increment
    new_count = counter.increment(model, date_str=date_today)
    assert new_count == 2
    assert counter.get_calls_today(model, date_str=date_today) == 2


def test_daily_call_counter_date_reset(temp_db):
    """Verify call count resets when the Pacific Time calendar date changes."""
    counter = DailyCallCounter(db_path=temp_db)
    model = "gemini-3.8-flash"

    # Calls on day 1
    counter.increment(model, date_str="2026-10-01")
    counter.increment(model, date_str="2026-10-01")
    assert counter.get_calls_today(model, date_str="2026-10-01") == 2

    # New date (day 2) must be clean 0
    assert counter.get_calls_today(model, date_str="2026-10-02") == 0

    # Incrementing day 2 does not alter day 1
    counter.increment(model, date_str="2026-10-02")
    assert counter.get_calls_today(model, date_str="2026-10-02") == 1
    assert counter.get_calls_today(model, date_str="2026-10-01") == 2


def test_daily_limit_exceeded(temp_db):
    """Verify DailyLimitExceededError is raised when limit is reached."""
    counter = DailyCallCounter(db_path=temp_db)
    model = "gemini-3.8-flash"
    limit = 3

    counter.check_and_increment(model, limit=limit)
    counter.check_and_increment(model, limit=limit)
    counter.check_and_increment(model, limit=limit)

    with pytest.raises(DailyLimitExceededError) as exc_info:
        counter.check_and_increment(model, limit=limit)
    assert "Daily free-tier call quota reached" in str(exc_info.value)


def test_rate_limiter_rpm_guard():
    """Verify RateLimiter blocks requests exceeding the configured RPM."""
    limiter = RateLimiter(rpm_limit=3)

    # First 3 requests must succeed
    limiter.check_and_acquire()
    limiter.check_and_acquire()
    limiter.check_and_acquire()

    # 4th request within same minute must raise RateLimitError
    with pytest.raises(RateLimitError) as exc_info:
        limiter.check_and_acquire()
    assert "Free-tier safety limit reached: 3 requests/min" in str(exc_info.value)

    # After reset, acquiring works again
    limiter.reset()
    limiter.check_and_acquire()
