"""
Unit tests for cost reference calculation including thinking tokens and embedding costs.
"""

from app.cost import compute_cost, embedding_cost
from app.config import Settings


def test_cost_calculation_small_model():
    """Verify small model pricing ($0.30 input / $2.50 output per 1M)."""
    # 10,000 input tokens = (10,000 / 1M) * 0.30 = $0.003
    # 5,000 output tokens = (5,000 / 1M) * 2.50 = $0.0125
    # Total = 0.0155
    cost = compute_cost(
        model="gemini-3.5-flash-lite",
        input_tokens=10_000,
        output_tokens=5_000,
        thinking_tokens=0,
    )
    assert cost == 0.0155


def test_cost_calculation_including_thinking_tokens():
    """Verify that thinking tokens are counted as billable output tokens."""
    # Large model: $0.75 input / $3.75 output
    # Input: 1,000,000 tokens -> $0.75
    # Output: 500,000 tokens, Thinking: 500,000 tokens -> total output: 1,000,000 tokens -> $3.75
    # Total = $4.50
    cost = compute_cost(
        model="gemini-3.8-flash",
        input_tokens=1_000_000,
        output_tokens=500_000,
        thinking_tokens=500_000,
    )
    assert cost == 4.50


def test_cost_thinking_tokens_added_correctly():
    """Verify thinking tokens increase cost exactly as output tokens do."""
    cost_without_thinking = compute_cost("gemini-3.8-flash", 1000, 1000, thinking_tokens=0)
    cost_with_thinking = compute_cost("gemini-3.8-flash", 1000, 500, thinking_tokens=500)
    assert cost_without_thinking == cost_with_thinking


def test_embedding_cost():
    """Verify embedding cost calculation ($0.20 per 1M tokens)."""
    # 1,000,000 tokens = $0.20
    assert embedding_cost(1_000_000) == 0.20
    # 50,000 tokens = (50,000 / 1M) * 0.20 = $0.01
    assert embedding_cost(50_000) == 0.01
    # 0 tokens = $0.00
    assert embedding_cost(0) == 0.0
