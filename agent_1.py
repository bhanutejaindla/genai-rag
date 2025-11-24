from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import asyncio

from .base import BaseAgent, AgentCard
from ..rag import add_document, query_documents


class IngestionRetrievalAgent(BaseAgent):
    """Handles document ingestion and retrieval (RAG)."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="ingestion_rag_agent",
                description="Ingests documents, maintains vector store and serves retrieval results.",
                capabilities=[
                    "ingest_document",
                    "retrieve_context",
                    "list_documents",
                ],
                rate_limit_per_minute=15,
            )
        )

    async def ingest_text(self, content: str, source: str, job_id: int | None = None) -> Dict[str, Any]:
        chunks_added = await asyncio.to_thread(add_document, content, source=source)

        return {"chunks_added": chunks_added}

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = await asyncio.to_thread(query_documents, query)

        return results


from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from .base import BaseAgent, AgentCard


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mcp_servers.research.server import web_search  # type: ignore


class WebResearchAgent(BaseAgent):
    """Performs grounded web searches and returns structured evidence."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="web_research_agent",
                description="Executes grounded web searches and returns structured findings.",
                capabilities=["web_search", "collect_sources", "stream_progress"],
                rate_limit_per_minute=10,
            )
        )

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:

        results = await asyncio.to_thread(web_search, query, max_results=max_results)
        structured = [
            {
                "title": item.get("title"),
                "date": item.get("date"),
                "quote": item.get("snippet") or item.get("description"),
                "url": item.get("url"),
            }
            for item in results
        ]

        return structured


from __future__ import annotations

import asyncio
from typing import Dict, Any

from .base import BaseAgent, AgentCard


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mcp_servers.compliance.server import redact_pii  # type: ignore


class ComplianceAgent(BaseAgent):
    """Detects PII/sensitive content and enforces policy gates."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="compliance_agent",
                description="Applies compliance policy, redacts sensitive text, records approvals.",
                capabilities=["redact", "approve", "block_export"],
                rate_limit_per_minute=20,
            )
        )

    async def enforce(self, draft_text: str, require_approval: bool = False) -> Dict[str, Any]:
        result = await asyncio.to_thread(redact_pii, draft_text)
        event_payload = {"approved": not require_approval, "length": len(draft_text)}
        return {"redacted_text": result}

from __future__ import annotations

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from .base import BaseAgent, AgentCard



class ChatAgent(BaseAgent):
    """Acts as a conversational surface for orchestrator-controlled flows."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="chat_agent",
                description="Maintains conversation history and routes user intents to orchestrator.",
                capabilities=["store_history", "forward_to_orchestrator", "answer"],
            )
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.history: Dict[str, List[BaseMessage]] = {}

    def append_message(self, thread_id: str, message: BaseMessage) -> None:
        self.history.setdefault(thread_id, []).append(message)
        self.history.setdefault(thread_id, []).append(message)

    def get_history(self, thread_id: str) -> List[BaseMessage]:
        return self.history.get(thread_id, [])

    async def summarize(self, thread_id: str) -> str:
        history = self.get_history(thread_id)
        if not history:
            return ""
        response = await self.llm.ainvoke(history[-6:])
        return response.content


from __future__ import annotations

from typing import Dict, Any
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseAgent, AgentCard
from ..report_generator import ReportGenerator



class SynthesisReportAgent(BaseAgent):
    """Generates structured reports from gathered evidence."""

    def __init__(self) -> None:
        super().__init__(
            AgentCard(
                name="synthesis_report_agent",
                description="Creates structured research reports with inline citations and exports.",
                capabilities=[
                    "generate_sections",
                    "produce_tables",
                    "export_report",
                ],
                rate_limit_per_minute=6,
            )
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            """You are the Synthesis & Report agent.
Write a structured report with sections, tables (markdown), and inline citations [1], [2], ...

User Query: {query}
Web Findings: {web_findings}
Retrieved Evidence: {rag_context}

Output JSON with keys:
- summary
- sections: [{title, content}]
- tables: [{title, rows}]
- citations: [{id, source, url, quote}]
"""
        )
        self.generator = ReportGenerator()

    async def generate_report(self, query: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        response = await chain.ainvoke(
            {
                "query": query,
                "web_findings": evidence.get("web_results", ""),
                "rag_context": evidence.get("context", ""),
            }
        )
        report_payload = {
            "summary": response.content,
            "sections": evidence.get("sections"),
            "citations": evidence.get("citations"),
        }

        return report_payload

    async def export(self, final_answer: str, job_id: int | None = None) -> Dict[str, str]:
        filename = f"report_{job_id}" if job_id else "report_preview"
        docx_path = await asyncio.to_thread(self.generator.generate_docx, final_answer, filename)
        pdf_path = await asyncio.to_thread(self.generator.generate_pdf, final_answer, filename)

        return {"docx": docx_path, "pdf": pdf_path}




