from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from .base import BaseAgent, AgentCard

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mcp_servers.citation_validation.server import verify_citations_internal  # type: ignore


class CitationAgent(BaseAgent):
    """Verifies citations in the generated report against sources."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="citation_agent",
                description="Verifies that citations in the text are supported by the provided sources.",
                capabilities=["verify_citations"],
                rate_limit_per_minute=20,
            )
        )

    async def verify(self, draft_answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies citations in the draft answer.
        """
        result = await asyncio.to_thread(verify_citations_internal, draft_answer, sources)
        return result
