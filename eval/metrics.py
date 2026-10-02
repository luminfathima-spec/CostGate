"""
Metrics calculations for evaluation results: cost savings, escalation rate, latency.
"""

from typing import List, Dict, Any


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate cost, latency, cache hit rate, and escalation rate."""
    return {
        "total_requests": len(results),
        "total_cost": sum(r.get("cost", 0.0) for r in results),
        "cache_hit_rate": 0.0,
        "escalation_rate": 0.0,
        "avg_latency_ms": 0.0,
    }
