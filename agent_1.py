from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import asyncio
import os
from dotenv import load_dotenv

# Import agents
from .agents.ingestion_agent import IngestionRetrievalAgent
from .agents.web_research_agent import WebResearchAgent
from .agents.synthesis_agent import SynthesisReportAgent
from .agents.citation_agent import CitationAgent
from .agents.compliance_agent import ComplianceAgent

load_dotenv()

# ------------------------
# STATE DEFINITION
# ------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    job_id: Optional[int]
    artifacts: Dict[str, Any]
    research_data: Dict[str, Any]
    final_report: Dict[str, str]


# ------------------------
# ROUTER MODEL
# ------------------------

from pydantic import BaseModel
from typing import Literal

class Router(BaseModel):
    next: Literal["research", "synthesis", "citation", "compliance", "report", "FINISH"]


# ------------------------
# SUPERVISOR ROUTER NODE
# ------------------------

async def supervisor_node(state: AgentState):
    next_step = state.get("next_step", "start")
    messages = state["messages"]

    if next_step == "start":
        return {"next_step": "research"}

    if next_step == "report":
        return {"next_step": "end"}

    system_prompt = """
    You are a supervisor managing:
    [research, synthesis, citation, compliance, report]

    Rules:
    1. If research not done → research
    2. If draft missing → synthesis
    3. If citations unverified → citation
    4. If compliance not checked → compliance
    5. Otherwise → report
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    router_llm = llm.with_structured_output(Router)

    response = await router_llm.ainvoke(
        [{"role": "system", "content": system_prompt}] + messages[-5:]
    )

    nxt = response.next
    if nxt == "FINISH":
        nxt = "end"

    return {"next_step": nxt}


# ------------------------
# AGENTS
# ------------------------

ingestion_agent = IngestionRetrievalAgent()
web_agent = WebResearchAgent()
synthesis_agent = SynthesisReportAgent()
citation_agent = CitationAgent()
compliance_agent = ComplianceAgent()


# ------------------------
# WORKER NODES
# ------------------------

async def research_node(state: AgentState):
    print("--- Research Node ---")
    query = state["messages"][0].content

    context = await ingestion_agent.retrieve(query)
    web_results = await web_agent.search(query)

    return {
        "research_data": {
            "context": context,
            "web_results": web_results
        },
        "messages": [AIMessage(content=f"Research complete: {len(web_results)} results.")]
    }


async def synthesis_node(state: AgentState):
    print("--- Synthesis Node ---")
    query = state["messages"][0].content
    data = state["research_data"]

    result = await synthesis_agent.generate_report(
        query,
        {
            "web_results": data.get("web_results", ""),
            "context": data.get("context", ""),
            "sections": [],
            "citations": [],
        },
    )

    return {
        "messages": [AIMessage(content=result["summary"])],
        "artifacts": {"draft_answer": result["summary"]}
    }


async def citation_node(state: AgentState):
    print("--- Citation Node ---")
    draft = state["artifacts"]["draft_answer"]
    web_results = state["research_data"]["web_results"]

    sources = [
        {
            "id": str(i+1),
            "title": s.get("title", "Unknown"),
            "text": s.get("quote", ""),
            "url": s.get("url", "")
        }
        for i, s in enumerate(web_results)
    ]

    result = await citation_agent.verify(draft, sources)

    return {
        "messages": [AIMessage(content=f"Citation verified. Score={result.get('score')}")],
        "artifacts": {"verification_result": result}
    }


async def compliance_node(state: AgentState):
    print("--- Compliance Node ---")
    draft = state["artifacts"]["draft_answer"]

    # Must return interrupt
    return interrupt({"msg": "Approve compliance?", "text": draft})


async def compliance_resume_node(state: AgentState):
    print("--- Compliance Resume Node ---")

    approval_data = state["messages"][-1].content  # resumed message
    draft = state["artifacts"]["draft_answer"]

    if approval_data.get("action") != "approve":
        return {
            "messages": [AIMessage(content="Compliance rejected.")],
            "artifacts": {"final_answer": "[BLOCKED]"}
        }

    result = await compliance_agent.enforce(draft, require_approval=False)

    return {
        "messages": [AIMessage(content="Compliance complete.")],
        "artifacts": {"final_answer": result["redacted_text"]}
    }


async def report_node(state: AgentState):
    print("--- Report Node ---")

    final_answer = state["artifacts"]["final_answer"]
    job_id = state.get("job_id")

    paths = await synthesis_agent.export(final_answer, job_id)

    return {
        "messages": [AIMessage(content="Report generated.")],
        "final_report": paths,
    }


# ------------------------
# GRAPH BUILD
# ------------------------

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research", research_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("citation", citation_node)
workflow.add_node("compliance", compliance_node)
workflow.add_node("compliance_resume", compliance_resume_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("supervisor")

# Router edges
def route_step(state: AgentState):
    return state["next_step"]

workflow.add_conditional_edges(
    "supervisor",
    route_step,
    {
        "research": "research",
        "synthesis": "synthesis",
        "citation": "citation",
        "compliance": "compliance",
        "report": "report",
        "end": END
    }
)

# Return edges
workflow.add_edge("research", "supervisor")
workflow.add_edge("synthesis", "supervisor")
workflow.add_edge("citation", "supervisor")
workflow.add_edge("compliance", "compliance_resume")
workflow.add_edge("compliance_resume", "supervisor")
workflow.add_edge("report", "supervisor")

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
