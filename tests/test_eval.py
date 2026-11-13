"""
Unit tests for Phase 2 evaluation dataset, judge JSON parser, resume behavior, and cost isolation.
All tests use mocks with zero network calls.
"""

import csv
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.config import get_settings
from eval.metrics import parse_judge_json, judge_answer
from eval.run_eval import load_dataset, get_completed_ids
from app.limits import RateLimiter, DailyCallCounter
from app.logger import RequestLogger


@pytest.fixture
def dataset_file():
    """Path to the generated dataset.csv."""
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir / "eval" / "dataset.csv"


def test_dataset_composition(dataset_file):
    """Verify dataset has exactly 60 rows, 15 per category, and 70/30 stratified split."""
    assert dataset_file.exists(), "dataset.csv must exist"

    with open(dataset_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 60, f"Expected 60 rows, got {len(rows)}"

    # Check categories
    categories = [r["category"] for r in rows]
    for cat in ["simple_fact", "explanation", "reasoning", "technical"]:
        assert categories.count(cat) == 15, f"Category {cat} should have 15 rows"

    # Check 70/30 split (42 train / 18 test)
    train_count = sum(1 for r in rows if r["split"] == "train")
    test_count = sum(1 for r in rows if r["split"] == "test")
    assert train_count == 42, f"Expected 42 train rows (70%), got {train_count}"
    assert test_count == 18, f"Expected 18 test rows (30%), got {test_count}"

    # Check key_points format
    for r in rows:
        points = [p.strip() for p in r["key_points"].split("|") if p.strip()]
        assert 2 <= len(points) <= 4, f"Row {r['id']} has invalid key points count: {len(points)}"


def test_judge_json_parser_clean():
    """Verify parser extracts score and reason from clean JSON."""
    raw = '{"score": 5, "reason": "Accurately answers all parts of the question."}'
    res = parse_judge_json(raw)
    assert res is not None
    score, reason = res
    assert score == 5
    assert "Accurately answers" in reason


def test_judge_json_parser_markdown_fence():
    """Verify parser handles markdown code blocks around JSON."""
    raw = """
    Here is my evaluation:
    ```json
    {
      "score": 4,
      "reason": "Covers most points, misses minor detail."
    }
    ```
    """
    res = parse_judge_json(raw)
    assert res is not None
    score, reason = res
    assert score == 4
    assert "Covers most points" in reason


def test_judge_json_parser_invalid():
    """Verify parser returns None for unparseable responses."""
    assert parse_judge_json("I think the answer is a 4 out of 5.") is None
    assert parse_judge_json("") is None
    assert parse_judge_json('{"score": 6, "reason": "out of range"}') is None


def test_judge_cost_isolated_and_logged(tmp_path, monkeypatch):
    """Verify judge cost is tracked separately and logged with route='judge'."""
    test_db = str(tmp_path / "test_judge.db")
    logger = RequestLogger(db_path=test_db)
    limiter = RateLimiter(rpm_limit=50)
    counter = DailyCallCounter(db_path=test_db)

    # Mock client
    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "text": '{"score": 5, "reason": "All key points addressed."}',
        "input_tokens": 150,
        "output_tokens": 30,
        "thinking_tokens": 10,
        "latency_ms": 300.0,
    }

    result = judge_answer(
        question="What is the capital of France?",
        key_points="Paris | capital of France",
        answer="Paris is the capital of France.",
        client=mock_client,
        rate_limiter=limiter,
        daily_counter=counter,
        logger_instance=logger,
    )

    assert result["score"] == 5
    assert result["reason"] == "All key points addressed."
    assert result["judge_cost"] > 0

    # Verify SQLite record has route='judge'
    recent = logger.get_recent_requests(limit=5)
    assert len(recent) == 1
    assert recent[0]["route"] == "judge"
    assert recent[0]["quality_score"] == 5.0


def test_evaluation_resumability(tmp_path):
    """Verify get_completed_ids correctly identifies already finished questions."""
    res_file = tmp_path / "results_test.csv"
    with open(res_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "answer"])
        writer.writerow([1, "simple_fact", "Paris"])
        writer.writerow([2, "simple_fact", "Au"])
        writer.writerow([5, "simple_fact", "0 C"])

    completed = get_completed_ids(res_file)
    assert completed == {1, 2, 5}
    assert 3 not in completed
