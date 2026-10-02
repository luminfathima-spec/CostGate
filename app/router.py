"""
Waterfall escalation router: Cache -> Small -> Medium -> Large.
"""

from typing import Dict, Any


class CostGateRouter:
    """Orchestrates query handling through semantic cache and tiered LLMs."""

    def __init__(self):
        pass

    async def route_query(self, question: str) -> Dict[str, Any]:
        """Route a user question through the waterfall pipeline."""
        raise NotImplementedError("Router will be implemented in the router task.")
