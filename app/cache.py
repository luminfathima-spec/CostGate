"""
Semantic Cache for user questions using vector similarity.
"""

from typing import Optional, Tuple


class SemanticCache:
    """In-memory or SQLite-backed vector similarity cache."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def get(self, question: str) -> Optional[Tuple[str, float]]:
        """Lookup answer by question similarity. Returns (answer, similarity) if hit."""
        return None

    def set(self, question: str, answer: str) -> None:
        """Store question and answer in cache."""
        pass
