"""
Gemini LLM Client wrapper with timeout, exponential backoff retries, and token extraction.
Includes immediate failure handling for retired models (404 'no longer available').
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, Dict, Any
from google import genai
from app.config import get_settings

logger = logging.getLogger("costgate.llm")


class ModelRetiredError(RuntimeError):
    """Raised when a configured model is deprecated or retired by Google API."""
    pass


def check_for_model_retirement(error: Exception, model_name: str) -> None:
    """
    Inspect an API error to determine if a model has been retired.
    If 404 / 'no longer available' is detected, immediately raise ModelRetiredError
    with instructions to update .env instead of triggering exponential backoff retries.
    """
    err_str = str(error)
    if "404" in err_str and ("no longer available" in err_str.lower() or "not_found" in err_str.upper()):
        raise ModelRetiredError(
            f"Configured model '{model_name}' is retired or no longer available (404 NOT_FOUND). "
            f"Please update your model configuration in .env. Details: {err_str}"
        ) from error


import re


def is_retryable_error(error: Exception) -> bool:
    """Determine whether an error qualifies for exponential backoff retries (429, 5xx, timeouts)."""
    err_str = str(error)

    # 404 is strictly non-retryable
    if "404" in err_str or "no longer available" in err_str.lower():
        return False

    # Timeouts (FutureTimeoutError, TimeoutError, network timeouts)
    if isinstance(error, (TimeoutError, FutureTimeoutError)) or any(
        t in err_str.lower() for t in ["timed out", "timeout", "deadline exceeded"]
    ):
        return True

    # 429 Rate limits / Resource Exhausted
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str.upper() or "quota" in err_str.lower():
        return True

    # 5xx Server errors / Service Unavailable
    if any(code in err_str for code in ["500", "502", "503", "504", "UNAVAILABLE", "INTERNAL"]):
        return True

    return False


def extract_retry_delay(error: Exception) -> Optional[float]:
    """Parse recommended retryDelay from Google GenAI error message or response details."""
    err_str = str(error)
    # Check for "retry in 58.302730191s" or "retry in 2s"
    m_s = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s", err_str, re.IGNORECASE)
    if m_s:
        return float(m_s.group(1))
    m_ms = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*ms", err_str, re.IGNORECASE)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    m_sec = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([0-9]+)s['\"]", err_str)
    if m_sec:
        return float(m_sec.group(1))
    return None


class GeminiClient:
    """Wrapper for Google GenAI SDK calls with resilient retry, timeout, and retirement handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_delay: Optional[float] = None,
        client: Optional[Any] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self.retry_base_delay = (
            retry_base_delay if retry_base_delay is not None else settings.llm_retry_base_delay
        )
        self._client = client or (genai.Client(api_key=self.api_key) if self.api_key else None)

    def generate(self, model: str, prompt: str) -> Dict[str, Any]:
        """
        Execute generation call against the specified model with timeout and retry handling.

        Returns:
            dict containing:
                - text (str): generated answer
                - input_tokens (int): prompt token count
                - output_tokens (int): candidate token count
                - thinking_tokens (int): reasoning tokens (thoughts_token_count or 0)
                - latency_ms (float): elapsed round-trip latency in milliseconds
        """
        if self._client is None:
            # Re-initialize if created after config load
            settings = get_settings()
            self._client = genai.Client(api_key=self.api_key or settings.gemini_api_key)

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                # Execute API call within timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._client.models.generate_content,
                        model=model,
                        contents=prompt,
                    )
                    try:
                        response = future.result(timeout=self.timeout_seconds)
                    except FutureTimeoutError as te:
                        raise TimeoutError(
                            f"LLM call to '{model}' timed out after {self.timeout_seconds:.1f}s"
                        ) from te

                latency_ms = (time.perf_counter() - start_time) * 1000.0

                # Extract token usage metadata safely
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

                # Answer text extraction
                answer_text = response.text.strip() if getattr(response, "text", None) else ""

                return {
                    "text": answer_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "latency_ms": round(latency_ms, 2),
                }

            except Exception as err:
                last_error = err
                # 1. Check for model retirement (strictly non-retryable)
                check_for_model_retirement(err, model)

                # 2. Check retry eligibility on 429, 5xx, and timeouts
                if attempt < self.max_retries and is_retryable_error(err):
                    backoff_delay = self.retry_base_delay * (2 ** attempt)
                    parsed_delay = extract_retry_delay(err)
                    if parsed_delay is not None:
                        backoff_delay = max(backoff_delay, parsed_delay + 1.5)
                    elif any(c in str(err) for c in ["503", "UNAVAILABLE"]):
                        backoff_delay = max(backoff_delay, 4.0 * (attempt + 1))

                    logger.warning(
                        "LLM call attempt %d failed for model '%s': %s. Retrying in %.2fs...",
                        attempt + 1,
                        model,
                        err,
                        backoff_delay,
                    )
                    time.sleep(backoff_delay)
                    continue
                else:
                    logger.error(
                        "LLM call failed permanently on attempt %d for model '%s': %s",
                        attempt + 1,
                        model,
                        err,
                    )
                    raise err

        if last_error:
            raise last_error
        raise RuntimeError(f"Unknown generation failure for model '{model}'")
