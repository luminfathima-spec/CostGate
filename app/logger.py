"""
SQLite logging module for tracking all request metrics, latency, token usage, and audit history.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from app.config import get_settings
from app.limits import get_pacific_date


class RequestLogger:
    """Manages SQLite persistence for incoming queries and gateway performance records."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_settings().db_path
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create requests audit table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    route TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    thinking_tokens INTEGER NOT NULL DEFAULT 0,
                    cost REAL NOT NULL DEFAULT 0.0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    escalated INTEGER NOT NULL DEFAULT 0,
                    quality_score REAL,
                    price_date TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_request(
        self,
        question: str,
        route: str,
        model: str,
        cache_hit: bool,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        cost: float,
        latency_ms: float,
        escalated: bool = False,
        quality_score: Optional[float] = None,
        price_date: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """
        Record a processed request to the SQLite database.
        Returns the inserted row ID.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        p_date = price_date or get_settings().price_date

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO requests (
                    timestamp, question, route, model, cache_hit,
                    input_tokens, output_tokens, thinking_tokens,
                    cost, latency_ms, escalated, quality_score, price_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    question,
                    route,
                    model,
                    1 if cache_hit else 0,
                    input_tokens,
                    output_tokens,
                    thinking_tokens,
                    cost,
                    latency_ms,
                    1 if escalated else 0,
                    quality_score,
                    p_date,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics for the gateway service:
        total_requests, total_reference_cost, average_latency_ms, calls_used_today.
        """
        today_pt = get_pacific_date()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Global aggregates
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(cost), 0.0) as total_reference_cost,
                    COALESCE(AVG(latency_ms), 0.0) as avg_latency
                FROM requests
                """
            )
            row = cursor.fetchone()
            total_requests = row["total_requests"] if row else 0
            total_reference_cost = round(row["total_reference_cost"], 6) if row else 0.0
            avg_latency = round(row["avg_latency"], 2) if row else 0.0

            # Calls today (Pacific Time date matching timestamp prefix or daily_usage table)
            cursor.execute(
                """
                SELECT COALESCE(SUM(call_count), 0) as today_calls
                FROM daily_usage
                WHERE date = ?
                """,
                (today_pt,),
            )
            usage_row = cursor.fetchone()
            calls_used_today = usage_row["today_calls"] if usage_row else 0

            return {
                "total_requests": total_requests,
                "total_reference_cost": total_reference_cost,
                "average_latency_ms": avg_latency,
                "calls_used_today": calls_used_today,
            }

    def get_recent_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch the most recent logged requests."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM requests ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]
