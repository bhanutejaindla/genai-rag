from __future__ import annotations

import os
import sys
import asyncio
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

# Fix path for MCP servers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .graph import graph
from .agents.orchestrator_agent import OrchestratorAgent

load_dotenv()


class ResearchAgent:
    """Top-level agent that triggers the orchestrator graph."""

    def __init__(self):
        self.graph = graph
        self.langfuse_handler = CallbackHandler()

        # Initialize orchestrator with graph runner
        self.orchestrator = OrchestratorAgent(graph_runner=self.graph)

        # Load LLM (optional here, used by sub-agents)
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = (
            ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[self.langfuse_handler])
            if api_key else None
        )


    async def run(self, query: str, thread_id: str = "default"):
        """Runs a full research workflow via orchestrator."""

        print(f"\n--- Starting ResearchAgent for Query: {query} ---")

        try:
            # thread_id is used only for chat tracking, not job_id
            result = await self.orchestrator.run_research_flow(
                query=query,
                job_id=None        # IMPORTANT: job_id not tied to thread
            )

            return result

        except Exception as e:
            print(f"❌ ResearchAgent execution failed: {e}")
            return {
                "answer": f"Error occurred: {str(e)}",
                "reports": {}
            }
