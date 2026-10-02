"""
Gemini LLM Client wrapper with timeout, exponential backoff retries, and token extraction.
Includes immediate failure handling for retired models (404 'no longer available').
"""

from typing import Optional, Dict, Any


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


class GeminiClient:
    """Wrapper for Google GenAI SDK calls with resilient retry and retirement handling."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate(self, model: str, prompt: str) -> Dict[str, Any]:
        """Placeholder for LLM generation with retry and retirement handling."""
        raise NotImplementedError("LLM generation will be implemented in the client task.")
