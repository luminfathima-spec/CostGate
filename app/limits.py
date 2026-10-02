"""
Free-tier guardrails: Rate Limiter (RPM) and Persistent Daily Call Counter (RPD).
Protects free tier quotas and resets counts based on Pacific Time.
"""

import time
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
from app.config import get_settings

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Fallback will be used


class RateLimitError(RuntimeError):
    """Raised when request rate exceeds configured RPM guard."""
    pass


class DailyLimitExceededError(RuntimeError):
    """Raised when daily call quota is exhausted."""
    pass


def get_pacific_date() -> str:
    """
    Get current date string (YYYY-MM-DD) in US Pacific Time (PT).
    Google resets free tier daily quotas at midnight Pacific Time.
    """
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Robust fallback: UTC-7 during US Daylight Saving (March to Nov), else UTC-8
    now_utc = datetime.now(timezone.utc)
    # Simple DST check for month in [4..10]
    offset_hours = -7 if 4 <= now_utc.month <= 10 else -8
    pacific_time = now_utc + timedelta(hours=offset_hours)
    return pacific_time.strftime("%Y-%m-%d")


class RateLimiter:
    """
    In-memory sliding window rate limiter ensuring requests stay
    under the free-tier RPM limit with a safety margin.
    """

    def __init__(self, rpm_limit: Optional[int] = None):
        self.rpm_limit = rpm_limit or get_settings().rpm_limit
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def check_and_acquire(self) -> None:
        """
        Record a request timestamp or raise RateLimitError if limit exceeded.
        """
        with self._lock:
            now = time.time()
            cutoff = now - 60.0

            # Discard timestamps older than 1 minute
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.rpm_limit:
                wait_sec = max(0.1, 60.0 - (now - self._timestamps[0]))
                raise RateLimitError(
                    f"Free-tier safety limit reached: {self.rpm_limit} requests/min. "
                    f"Please wait {wait_sec:.1f}s before trying again."
                )

            self._timestamps.append(now)

    def reset(self) -> None:
        """Clear recorded timestamps (primarily for test isolation)."""
        with self._lock:
            self._timestamps.clear()


class DailyCallCounter:
    """
    Persistent SQLite call counter tracking daily usage per model.
    Resets automatically when the Pacific Time calendar date changes.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_settings().db_path
        self._init_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        """Create daily_usage persistence table if not present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_usage (
                    model TEXT NOT NULL,
                    date TEXT NOT NULL,
                    call_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (model, date)
                )
                """
            )
            conn.commit()

    def get_calls_today(self, model: str, date_str: Optional[str] = None) -> int:
        """Retrieve total calls made today (Pacific Time) for the given model."""
        target_date = date_str or get_pacific_date()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT call_count FROM daily_usage WHERE model = ? AND date = ?",
                (model, target_date),
            )
            row = cursor.fetchone()
            return row["call_count"] if row else 0

    def increment(self, model: str, date_str: Optional[str] = None) -> int:
        """
        Atomically increment the call count for today (Pacific Time).
        Returns the new call count.
        """
        target_date = date_str or get_pacific_date()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO daily_usage (model, date, call_count)
                VALUES (?, ?, 1)
                ON CONFLICT(model, date) DO UPDATE SET call_count = call_count + 1
                RETURNING call_count
                """,
                (model, target_date),
            )
            row = cursor.fetchone()
            conn.commit()
            return row[0] if row else 1

    def check_and_increment(self, model: str, limit: Optional[int] = None) -> int:
        """
        Verify that daily limit is not exceeded, then increment counter.
        Raises DailyLimitExceededError if limit is reached.
        """
        daily_limit = limit or get_settings().daily_call_limit
        current = self.get_calls_today(model)
        if current >= daily_limit:
            raise DailyLimitExceededError(
                f"Daily free-tier call quota reached for model '{model}' "
                f"({current}/{daily_limit} calls today PT). Quota resets at midnight Pacific Time."
            )
        return self.increment(model)

    def get_all_calls_today(self, date_str: Optional[str] = None) -> Dict[str, int]:
        """Return usage dictionary {model: count} for all models today."""
        target_date = date_str or get_pacific_date()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model, call_count FROM daily_usage WHERE date = ?",
                (target_date,),
            )
            return {row["model"]: row["call_count"] for row in cursor.fetchall()}
