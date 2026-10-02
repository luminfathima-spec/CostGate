"""
Cost calculation module for reference token pricing across model tiers.

IMPORTANT - Reference Accounting Notice:
All cost metrics computed in this module are theoretical reference calculations
representing what a paid API user would pay for token consumption (Standard Tier).
Free-tier accounts incur $0.00 from Google; CostGate tracks reference costs to quantify
monetary value saved through caching and intelligent model tiering.

Accounting Rules:
1. Output Tokens: Thinking/reasoning tokens returned by Gemini models MUST be counted
   as billable output tokens:
   billable_output_tokens = output_tokens + thinking_tokens
2. Cache Lookups: Cache lookups are not free; the query embedding generation cost is
   accounted for via embedding_cost(tokens).
"""

from typing import Optional
from app.config import Settings, get_settings


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    settings: Optional[Settings] = None,
) -> float:
    """
    Compute reference USD cost for generation based on token counts and model pricing.

    Args:
        model: Identifier of the Gemini model used
        input_tokens: Prompt token count
        output_tokens: Candidate token count
        thinking_tokens: Reasoning/thinking tokens (thoughts_token_count)
        settings: Optional Settings override (uses active config by default)

    Returns:
        float: Estimated reference cost in USD rounded to 8 decimal places.
    """
    cfg = settings or get_settings()

    # Normalize model identifier
    m = model.strip().lower()

    if m == cfg.model_small.lower() or "flash-lite" in m or "small" in m:
        price_input = cfg.price_input_small
        price_output = cfg.price_output_small
    elif m == cfg.model_medium.lower() or "3.7-flash" in m or "medium" in m:
        price_input = cfg.price_input_medium
        price_output = cfg.price_output_medium
    elif m == cfg.model_large.lower() or "3.8-flash" in m or "pro" in m or "large" in m:
        price_input = cfg.price_input_large
        price_output = cfg.price_output_large
    else:
        # Default to large tier pricing for unrecognized models (conservative)
        price_input = cfg.price_input_large
        price_output = cfg.price_output_large

    billable_output = max(0, output_tokens) + max(0, thinking_tokens)
    billable_input = max(0, input_tokens)

    input_cost = (billable_input / 1_000_000.0) * price_input
    output_cost = (billable_output / 1_000_000.0) * price_output

    total_cost = input_cost + output_cost
    return round(total_cost, 8)


def embedding_cost(tokens: int, settings: Optional[Settings] = None) -> float:
    """
    Compute reference USD cost for generating embeddings (e.g. for semantic cache lookups).

    Args:
        tokens: Number of tokens embedded
        settings: Optional Settings override

    Returns:
        float: Estimated reference cost in USD rounded to 8 decimal places.
    """
    cfg = settings or get_settings()
    billable_tokens = max(0, tokens)
    cost = (billable_tokens / 1_000_000.0) * cfg.price_embedding
    return round(cost, 8)
