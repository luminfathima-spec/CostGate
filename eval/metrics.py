"""
Evaluation metrics and LLM-as-a-judge module for CostGate.
Scores answer quality from 1 to 5 based on key points coverage and factual accuracy.
Protects free-tier quota using the shared RateLimiter and DailyCallCounter.
"""

import json
import re
import logging
from typing import Optional, Dict, Any, Tuple
from app.config import get_settings
from app.llm_client import GeminiClient
from app.cost import compute_cost
from app.limits import RateLimiter, DailyCallCounter
from app.logger import RequestLogger

logger = logging.getLogger("costgate.eval")

# Default singletons if not provided
_default_client = GeminiClient()
_default_rate_limiter = RateLimiter()
_default_daily_counter = DailyCallCounter()
_default_logger = RequestLogger()


def parse_judge_json(raw_text: str) -> Optional[Tuple[int, str]]:
    """
    Safely extract and parse JSON object containing 'score' (1-5) and 'reason' (str).
    Handles markdown code blocks and raw JSON text.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    # Attempt direct JSON load
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "score" in data and "reason" in data:
            score = int(data["score"])
            reason = str(data["reason"]).strip()
            if 1 <= score <= 5:
                return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback regex extraction
    score_match = re.search(r'"score"\s*:\s*([1-5])', cleaned)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
    if score_match and reason_match:
        score = int(score_match.group(1))
        reason = reason_match.group(1).strip()
        return score, reason

    return None


def format_judge_prompt(question: str, key_points: str, answer: str) -> str:
    """Construct strict scoring prompt for the evaluation judge."""
    points_list = "\n".join(f"- {p.strip()}" for p in key_points.split("|") if p.strip())
    return (
        "You are an impartial, strict evaluation judge for an AI question-answering gateway.\n"
        "Assess how well the provided candidate answer covers the must-have key points and whether it is factually accurate.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"MUST-HAVE KEY POINTS:\n{points_list}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        "SCORING RUBRIC (1 to 5):\n"
        "5: Excellent - Completely accurate, covers all key points clearly and thoroughly with zero errors.\n"
        "4: Good - Accurate, covers most key points with minor, non-critical omissions.\n"
        "3: Acceptable - Partially accurate or addresses some key points, but misses important details.\n"
        "2: Poor - Major inaccuracies, misses most key points, or contains hallucinations.\n"
        "1: Very Poor - Factually incorrect, irrelevant, contradictory, or refusal.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "score": <integer from 1 to 5>,\n'
        '  "reason": "<1-2 sentence concise explanation>"\n'
        "}"
    )


def judge_answer(
    question: str,
    key_points: str,
    answer: str,
    client: Optional[GeminiClient] = None,
    rate_limiter: Optional[RateLimiter] = None,
    daily_counter: Optional[DailyCallCounter] = None,
    logger_instance: Optional[RequestLogger] = None,
) -> Dict[str, Any]:
    """
    Score candidate answer against key points using the configured JUDGE_MODEL.

    CRITICAL RULE:
    Judge calls are tracked separately with route="judge" in SQLite.
    Judge cost MUST NEVER be included in strategy costs!

    Returns:
        dict: {
            "score": int (1-5),
            "reason": str,
            "judge_cost": float (isolated judge reference cost),
            "input_tokens": int,
            "output_tokens": int,
            "thinking_tokens": int
        }
    """
    settings = get_settings()
    judge_model = settings.judge_model or settings.model_medium

    llm = client or _default_client
    limiter = rate_limiter or _default_rate_limiter
    counter = daily_counter or _default_daily_counter
    log_db = logger_instance or _default_logger

    prompt = format_judge_prompt(question, key_points, answer)

    # 1. First attempt
    limiter.acquire(wait=True)
    counter.check_and_increment(judge_model)

    res = llm.generate(model=judge_model, prompt=prompt)
    judge_cost = compute_cost(
        model=judge_model,
        input_tokens=res["input_tokens"],
        output_tokens=res["output_tokens"],
        thinking_tokens=res["thinking_tokens"],
        settings=settings,
    )

    parsed = parse_judge_json(res["text"])

    # 2. Retry once if JSON was malformed
    if not parsed:
        logger.warning("Judge response was not valid JSON. Retrying once: %r", res["text"][:100])
        retry_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Your previous response was not valid JSON. "
            'Output ONLY the JSON object: {"score": <1-5>, "reason": "<string>"}'
        )
        limiter.acquire(wait=True)
        counter.check_and_increment(judge_model)
        retry_res = llm.generate(model=judge_model, prompt=retry_prompt)
        retry_cost = compute_cost(
            model=judge_model,
            input_tokens=retry_res["input_tokens"],
            output_tokens=retry_res["output_tokens"],
            thinking_tokens=retry_res["thinking_tokens"],
            settings=settings,
        )
        judge_cost += retry_cost
        res = retry_res
        parsed = parse_judge_json(retry_res["text"])

    score, reason = parsed if parsed else (1, "Judge JSON parse failure")

    # 3. Log judge call to SQLite audit trail under route 'judge'
    log_db.log_request(
        question=f"[JUDGE] {question[:100]}",
        route="judge",
        model=judge_model,
        cache_hit=False,
        input_tokens=res["input_tokens"],
        output_tokens=res["output_tokens"],
        thinking_tokens=res["thinking_tokens"],
        cost=judge_cost,
        latency_ms=res["latency_ms"],
        escalated=False,
        quality_score=float(score),
        price_date=settings.price_date,
    )

    return {
        "score": score,
        "reason": reason,
        "judge_cost": judge_cost,
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
        "thinking_tokens": res["thinking_tokens"],
    }
