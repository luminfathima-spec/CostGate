"""
Evaluation summary reporting script for CostGate.
Analyzes results CSVs and prints overall and per-category metrics:
judge score, % >= 4, total reference cost, cost/100, latency, and thinking tokens.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

root_dir = Path(__file__).resolve().parent.parent


def parse_results_file(file_path: Path) -> List[Dict[str, Any]]:
    """Read evaluation CSV and cast numeric columns."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                records.append({
                    "id": int(r["id"]),
                    "category": r["category"],
                    "answer": r["answer"],
                    "input_tokens": int(r.get("input_tokens", 0)),
                    "output_tokens": int(r.get("output_tokens", 0)),
                    "thinking_tokens": int(r.get("thinking_tokens", 0)),
                    "cost": float(r.get("cost", 0.0)),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "judge_score": int(r.get("judge_score", 1)),
                    "judge_reason": r.get("judge_reason", ""),
                    "model": r.get("model", "unknown"),
                    "price_input": float(r.get("price_input", 0.0)),
                    "price_output": float(r.get("price_output", 0.0)),
                    "price_date": r.get("price_date", "unknown"),
                })
            except (ValueError, KeyError):
                continue
    return records


def compute_metrics_slice(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics for a list of evaluated query records."""
    if not items:
        return {
            "count": 0,
            "avg_score": 0.0,
            "pct_ge_4": 0.0,
            "total_cost": 0.0,
            "cost_per_100": 0.0,
            "avg_latency": 0.0,
            "avg_thinking_tokens": 0.0,
        }

    n = len(items)
    avg_score = sum(x["judge_score"] for x in items) / n
    pct_ge_4 = (sum(1 for x in items if x["judge_score"] >= 4) / n) * 100.0
    total_cost = sum(x["cost"] for x in items)
    cost_per_100 = (total_cost / n) * 100.0
    avg_latency = sum(x["latency_ms"] for x in items) / n
    avg_thinking = sum(x["thinking_tokens"] for x in items) / n

    return {
        "count": n,
        "avg_score": round(avg_score, 2),
        "pct_ge_4": round(pct_ge_4, 1),
        "total_cost": round(total_cost, 6),
        "cost_per_100": round(cost_per_100, 4),
        "avg_latency": round(avg_latency, 1),
        "avg_thinking_tokens": round(avg_thinking, 1),
    }


def print_strategy_report(strategy_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Print full metrics breakdown for a strategy."""
    overall = compute_metrics_slice(records)
    sample = records[0] if records else {}

    model_id = sample.get("model", "n/a")
    p_in = sample.get("price_input", 0.0)
    p_out = sample.get("price_output", 0.0)
    p_date = sample.get("price_date", "n/a")

    print("=" * 80)
    print(f"STRATEGY: {strategy_name.upper()}")
    print("=" * 80)
    print(f"Model ID            : {model_id}")
    print(f"Pricing per 1M tok  : ${p_in:.2f} input / ${p_out:.2f} output (PRICE_DATE: {p_date})")
    print(f"Evaluated Questions : {overall['count']}")
    print("-" * 80)
    print(f"Average Judge Score : {overall['avg_score']} / 5.0")
    print(f"High Quality (>= 4) : {overall['pct_ge_4']}%")
    print(f"Total Ref Cost      : ${overall['total_cost']:.6f} (answers only, excludes judge)")
    print(f"Cost per 100 Queries: ${overall['cost_per_100']:.4f}")
    print(f"Average Latency     : {overall['avg_latency']} ms")
    print(f"Average Thinking Tok: {overall['avg_thinking_tokens']} tokens/query")
    print("-" * 80)

    # Per-category breakdown
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    print(f"{'Category':<15} {'Count':<6} {'Avg Score':<10} {'% >= 4':<8} {'Cost/100':<12} {'Avg Lat (ms)':<14} {'Avg Think Tok'}")
    print("-" * 80)
    categories = ["simple_fact", "explanation", "reasoning", "technical"]
    for cat in categories:
        cat_records = by_cat.get(cat, [])
        cm = compute_metrics_slice(cat_records)
        print(
            f"{cat:<15} {cm['count']:<6} {cm['avg_score']:<10.2f} {cm['pct_ge_4']:<8.1f}% "
            f"${cm['cost_per_100']:<11.4f} {cm['avg_latency']:<14.1f} {cm['avg_thinking_tokens']:.1f}"
        )
    print("=" * 80 + "\n")
    return overall


def main():
    eval_dir = root_dir / "eval"
    small_file = eval_dir / "results_always_small.csv"
    large_file = eval_dir / "results_always_large.csv"

    small_records = parse_results_file(small_file)
    large_records = parse_results_file(large_file)

    if not small_records and not large_records:
        print("No evaluation results files found in eval/.")
        print("Run `python eval/run_eval.py --strategy always_small` first.")
        return

    small_summary = None
    large_summary = None

    if small_records:
        small_summary = print_strategy_report("always_small", small_records)
    if large_records:
        large_summary = print_strategy_report("always_large", large_records)

    # If both are present, print comparison summary
    if small_summary and large_summary and small_summary["count"] > 0 and large_summary["count"] > 0:
        print("=" * 80)
        print("COMPARISON: ALWAYS_SMALL vs ALWAYS_LARGE")
        print("=" * 80)
        c_small = small_summary["cost_per_100"]
        c_large = large_summary["cost_per_100"]
        savings_pct = ((c_large - c_small) / c_large * 100.0) if c_large > 0 else 0.0

        print(f"{'Metric':<25} {'Always Small':<16} {'Always Large':<16} {'Comparison / Savings'}")
        print("-" * 80)
        print(f"{'Average Judge Score':<25} {small_summary['avg_score']:<16.2f} {large_summary['avg_score']:<16.2f} {small_summary['avg_score'] - large_summary['avg_score']:+.2f}")
        print(f"{'High Quality (>= 4)':<25} {small_summary['pct_ge_4']:<15.1f}% {large_summary['pct_ge_4']:<15.1f}% {small_summary['pct_ge_4'] - large_summary['pct_ge_4']:+.1f}%")
        print(f"{'Cost per 100 Queries':<25} ${c_small:<15.4f} ${c_large:<15.4f} {savings_pct:.1f}% savings with small")
        print(f"{'Average Latency':<25} {small_summary['avg_latency']:<14.1f}ms {large_summary['avg_latency']:<14.1f}ms {large_summary['avg_latency'] - small_summary['avg_latency']:+.1f}ms faster")
        print(f"{'Avg Thinking Tokens':<25} {small_summary['avg_thinking_tokens']:<16.1f} {large_summary['avg_thinking_tokens']:<16.1f} {large_summary['avg_thinking_tokens'] - small_summary['avg_thinking_tokens']:+.1f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
