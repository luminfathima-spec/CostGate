"""
Cost calculation module for token usage across small, medium, and large model tiers.

IMPORTANT - Cost Accounting Rule:
Thinking/reasoning tokens returned by Gemini models must always be counted
as output tokens when calculating inference cost.
"""

from typing import Optional


def calculate_cost(
    model_tier: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> float:
    """
    Calculate the estimated USD cost based on token counts and model tier pricing.

    Cost accounting rule:
    Total billable output tokens = output_tokens + thinking_tokens.
    """
    total_output = output_tokens + thinking_tokens
    # Full calculation logic will be implemented in the cost task.
    return 0.0
