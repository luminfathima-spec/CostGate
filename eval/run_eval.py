"""
CostGate evaluation benchmark runner.
Executes evaluation strategies (always_small, always_large) in-process against eval/dataset.csv.
Protects free-tier quota using shared RateLimiter, DailyCallCounter, and resumable output CSVs.
"""

import sys
import csv
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.config import get_settings
from app.llm_client import GeminiClient, ModelRetiredError
from app.cost import compute_cost
from app.limits import RateLimiter, DailyCallCounter, RateLimitError, DailyLimitExceededError
from app.logger import RequestLogger
from eval.metrics import judge_answer

logger = logging.getLogger("costgate.eval_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_dataset(dataset_path: Path, split: str = "all") -> List[Dict[str, Any]]:
    """Load and filter questions from dataset.csv by split."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if split == "all" or r.get("split", "").lower() == split.lower():
                r["id"] = int(r["id"])
                rows.append(r)
    return rows


def get_completed_ids(results_path: Path) -> Set[int]:
    """Read existing results file and return set of completed question IDs."""
    if not results_path.exists():
        return set()

    completed = set()
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "id" in r and r["id"]:
                try:
                    completed.add(int(r["id"]))
                except ValueError:
                    pass
    return completed


def init_results_file(results_path: Path) -> None:
    """Initialize CSV headers if results file does not exist."""
    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "category",
                "answer",
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
                "cost",
                "latency_ms",
                "judge_score",
                "judge_reason",
                "model",
                "price_input",
                "price_output",
                "price_date",
            ])


def run_evaluation():
    parser = argparse.ArgumentParser(description="CostGate In-Process Evaluation Benchmark Runner")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["always_large", "always_small"],
        help="Evaluation strategy to benchmark",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "test", "all"],
        help="Dataset split to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print projected API call counts and quota usage without executing calls",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(root_dir / "eval" / "dataset.csv"),
        help="Path to evaluation dataset CSV",
    )
    args = parser.parse_args()

    settings = get_settings()
    rate_limiter = RateLimiter()
    daily_counter = DailyCallCounter()
    req_logger = RequestLogger()
    client = GeminiClient()

    # Determine strategy model and reference pricing
    if args.strategy == "always_small":
        target_model = settings.model_small
        price_input = settings.price_input_small
        price_output = settings.price_output_small
    else:
        target_model = settings.model_large
        price_input = settings.price_input_large
        price_output = settings.price_output_large

    dataset_path = Path(args.dataset)
    results_path = root_dir / "eval" / f"results_{args.strategy}.csv"

    # Load dataset and identify uncompleted questions
    all_questions = load_dataset(dataset_path, split=args.split)
    completed_ids = get_completed_ids(results_path)
    pending_questions = [q for q in all_questions if q["id"] not in completed_ids]

    if args.limit:
        pending_questions = pending_questions[: args.limit]

    # Check daily usage
    daily_usage = daily_counter.get_all_calls_today()
    calls_used_today = sum(daily_usage.values())
    remaining_daily_budget = max(0, settings.daily_call_limit - calls_used_today)

    num_pending = len(pending_questions)
    # Each question requires 2 API calls: 1 for model generation + 1 for judge
    projected_api_calls = num_pending * 2

    print("=" * 70)
    print(f"CostGate Evaluation: {args.strategy.upper()}")
    print("=" * 70)
    print(f"Model ID              : {target_model}")
    print(f"Reference Pricing     : ${price_input:.2f} in / ${price_output:.2f} out per 1M tokens")
    print(f"Price Snapshot Date   : {settings.price_date}")
    print(f"Split Selection       : {args.split}")
    print(f"Total in Split        : {len(all_questions)}")
    print(f"Already Completed     : {len(completed_ids)}")
    print(f"Pending to Process    : {num_pending}")
    print(f"Projected API Calls   : {projected_api_calls} ({num_pending} answers + {num_pending} judge calls)")
    print(f"Daily Call Quota (PT) : {calls_used_today}/{settings.daily_call_limit} calls used today (Remaining: {remaining_daily_budget})")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No API calls were made.")
        if projected_api_calls > remaining_daily_budget:
            print(f"[WARNING] Projected calls ({projected_api_calls}) exceed remaining daily budget ({remaining_daily_budget})!")
        else:
            print(f"[OK] Remaining daily budget is sufficient for this run.")
        return

    if num_pending == 0:
        print("All questions in this split are already completed!")
        return

    # Check budget before execution
    if projected_api_calls > remaining_daily_budget:
        print(f"ERROR: Projected calls ({projected_api_calls}) exceed remaining daily quota ({remaining_daily_budget}). Aborting.")
        sys.exit(1)

    init_results_file(results_path)

    consecutive_failures = 0
    total_attempted = 0
    total_failed = 0

    with open(results_path, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)

        for i, q in enumerate(pending_questions, start=1):
            q_id = q["id"]
            cat = q["category"]
            question_text = q["question"]
            key_points = q["key_points"]

            # Quota safety guard: Stop cleanly if within 10 calls of daily limit
            current_calls = sum(daily_counter.get_all_calls_today().values())
            if current_calls >= settings.daily_call_limit - 10:
                print(f"\n[QUOTA GUARD] Approaching daily call limit ({current_calls}/{settings.daily_call_limit}). Stopping cleanly.")
                break

            total_attempted += 1
            print(f"[{i}/{num_pending}] Q{q_id:<2} ({cat:<12}) Generating answer with {target_model}...", end=" ", flush=True)

            # 1. Generate answer with model
            answer_result = None
            for retry_attempt in range(4):
                try:
                    rate_limiter.acquire(wait=True)
                    daily_counter.check_and_increment(target_model)
                    answer_result = client.generate(model=target_model, prompt=question_text)
                    break
                except DailyLimitExceededError as dle:
                    print(f"\n[QUOTA] Daily limit exceeded: {dle}")
                    break
                except Exception as err:
                    if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                        backoff = 15 * (retry_attempt + 1)
                        print(f"\n429 Rate Limit. Waiting {backoff}s before retry...", flush=True)
                        time.sleep(backoff)
                    elif any(c in str(err) for c in ["503", "UNAVAILABLE"]):
                        backoff = 10 * (retry_attempt + 1)
                        print(f"\n503 Service Unavailable. Waiting {backoff}s before retry...", flush=True)
                        time.sleep(backoff)
                    else:
                        print(f"\nGeneration error on Q{q_id}: {err}")
                        break

            if not answer_result:
                total_failed += 1
                consecutive_failures += 1
                print("FAILED.")
                if total_attempted >= 5 and (total_failed / total_attempted) > 0.20:
                    print(f"\n[ABORT] Error rate {total_failed}/{total_attempted} (>20%) exceeds tolerance threshold. Aborting evaluation.")
                    break
                continue

            consecutive_failures = 0
            ans_text = answer_result["text"]
            in_tok = answer_result["input_tokens"]
            out_tok = answer_result["output_tokens"]
            think_tok = answer_result["thinking_tokens"]
            lat_ms = answer_result["latency_ms"]

            # Calculate strategy cost (strictly answer call only!)
            strat_cost = compute_cost(
                model=target_model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                thinking_tokens=think_tok,
                settings=settings,
            )

            # Log strategy answer to SQLite
            req_logger.log_request(
                question=question_text,
                route=f"eval_{args.strategy}",
                model=target_model,
                cache_hit=False,
                input_tokens=in_tok,
                output_tokens=out_tok,
                thinking_tokens=think_tok,
                cost=strat_cost,
                latency_ms=lat_ms,
                escalated=False,
                quality_score=None,
                price_date=settings.price_date,
            )

            # 2. Judge Answer
            print(f"done ({lat_ms:.0f}ms). Judging...", end=" ", flush=True)
            judge_res = None
            for retry_attempt in range(4):
                try:
                    judge_res = judge_answer(
                        question=question_text,
                        key_points=key_points,
                        answer=ans_text,
                        client=client,
                        rate_limiter=rate_limiter,
                        daily_counter=daily_counter,
                        logger_instance=req_logger,
                    )
                    break
                except DailyLimitExceededError as dle:
                    print(f"\n[QUOTA] Daily limit exceeded during judging: {dle}")
                    break
                except Exception as err:
                    if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                        backoff = 20 * (retry_attempt + 1)
                        print(f"\n429 Judge rate limit. Backing off {backoff}s...", flush=True)
                        time.sleep(backoff)
                    elif any(c in str(err) for c in ["503", "UNAVAILABLE"]):
                        backoff = 10 * (retry_attempt + 1)
                        print(f"\n503 Judge unavailable. Backing off {backoff}s...", flush=True)
                        time.sleep(backoff)
                    else:
                        print(f"\nJudge error on Q{q_id}: {err}")
                        break

            j_score = judge_res["score"] if judge_res else 1
            j_reason = judge_res["reason"] if judge_res else "Judge evaluation failed"

            print(f"Score: {j_score}/5 | Cost: ${strat_cost:.6f}")

            # Write row to CSV (resumable)
            writer.writerow([
                q_id,
                cat,
                ans_text.replace("\n", " "),
                in_tok,
                out_tok,
                think_tok,
                strat_cost,
                lat_ms,
                j_score,
                j_reason.replace("\n", " "),
                target_model,
                price_input,
                price_output,
                settings.price_date,
            ])
            out_f.flush()

    print("\n" + "=" * 70)
    print(f"Evaluation finished. Results saved to: {results_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
