from typing import TypedDict, Annotated, List, Dict, Any, Optional, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
import asyncio

# Import tools/functions from existing modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from .kafka_client import send_job_event
from .agents.ingestion_agent import IngestionRetrievalAgent
from .agents.web_research_agent import WebResearchAgent
from .agents.synthesis_agent import SynthesisReportAgent
from .agents.compliance_agent import ComplianceAgent
from .a2a import a2a_bus


load_dotenv()

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    job_id: Optional[int]
    artifacts: Dict[str, Any]
    research_data: Dict[str, Any] # Store research results
    final_report: Dict[str, str] # Store paths to generated reports

# --- Nodes ---

from pydantic import BaseModel
from typing import Literal

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["research", "synthesis", "compliance", "report", "FINISH"]

async def supervisor_node(state: AgentState):
    """
    Supervisor node that routes to the next worker based on the conversation state.
    """
    messages = state["messages"]
    next_step = state.get("next_step", "start")
    
    # If we just started, default to research
    if next_step == "start":
        return {"next_step": "research"}
        
    # If we just finished report, we are done
    if next_step == "report":
        return {"next_step": "end"}

    # For other steps, use LLM to decide (or keep simple linear flow if preferred, 
    # but user asked for LLM decision. However, strictly linear dependencies 
    # (Research -> Synthesis -> Compliance -> Report) are often better enforced 
    # by the graph structure itself for this specific pipeline. 
    # But to satisfy "LLM should decide", we can give it the option.)
    
    # Actually, for this specific "Research Agent", the flow is quite linear.
    # But let's implement the Router pattern to allow for loops (e.g. Synthesis -> Research -> Synthesis).
    
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  [research, synthesis, compliance, report].\n"
        "Given the following user request and current state, respond with the worker to act next.\n"
        "1. If research is needed or missing, choose 'research'.\n"
        "2. If research is done but no draft answer exists, choose 'synthesis'.\n"
        "3. If draft exists but not checked for compliance, choose 'compliance'.\n"
        "4. If compliance is done, choose 'report'.\n"
        "5. If everything is complete, choose 'FINISH'."
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Router)
    
    # Create a prompt with history
    # We simplify history for the router to avoid token limits
    response = await structured_llm.ainvoke(
        [{"role": "system", "content": system_prompt}] + messages[-5:]
    )
    
    next_node = response.next
    if next_node == "FINISH":
        next_node = "end"
        
    return {"next_step": next_node}

ingestion_agent = IngestionRetrievalAgent()
web_agent = WebResearchAgent()
synthesis_agent = SynthesisReportAgent()
compliance_agent = ComplianceAgent()


async def research_node(state: AgentState):
    """
    Performs RAG and Web Research.
    """
    print("--- Node: Research ---")
    job_id = state.get("job_id")
    await send_job_event(job_id, "researching", 0.2, "Performing Web Research & RAG")
    
    query = state["messages"][0].content
    
    # 1. RAG
    rag_task = a2a_bus.create_task(
        sender="orchestrator",
        recipient=ingestion_agent.card.name,
        payload={"action": "retrieve_context", "query": query},
    )
    context = await ingestion_agent.retrieve(query)
    a2a_bus.update_task(
        rag_task.task_id,
        status="completed",
        artifacts=[{"type": "context_chunks", "value": context}],
    )
    
    # 2. Web Search
    web_task = a2a_bus.create_task(
        sender="orchestrator",
        recipient=web_agent.card.name,
        payload={"action": "web_search", "query": query},
    )
    try:
        web_results = await web_agent.search(query, max_results=5)
        a2a_bus.update_task(
            web_task.task_id,
            status="completed",
            artifacts=[{"type": "web_results", "value": web_results}],
        )
    except Exception as e:
        web_results = [{"error": str(e)}]
        a2a_bus.update_task(
            web_task.task_id,
            status="failed",
            artifacts=[{"type": "error", "value": str(e)}],
        )
        
    return {
        "research_data": {
            "context": context,
            "web_results": web_results
        },
        "messages": [AIMessage(content=f"Research complete. Found {len(web_results)} chars of web data.")]
    }

async def synthesis_node(state: AgentState):
    """
    Synthesizes the answer using LLM.
    """
    print("--- Node: Synthesis ---")
    job_id = state.get("job_id")
    await send_job_event(job_id, "synthesizing", 0.5, "Synthesizing Answer")
    
    query = state["messages"][0].content
    data = state["research_data"]
    
    response_payload = await synthesis_agent.generate_report(
        query,
        {
            "web_results": data.get("web_results", ""),
            "context": data.get("context", ""),
            "sections": [],
            "citations": [],
        },
    )
    
    return {
        "messages": [AIMessage(content=response_payload["summary"])],
        "artifacts": {"draft_answer": response_payload["summary"]}
    }

async def compliance_node(state: AgentState):
    """
    Checks for PII and compliance.
    """
    print("--- Node: Compliance ---")
    job_id = state.get("job_id")
    await send_job_event(job_id, "compliance", 0.7, "Checking Compliance")
    
    draft = state["artifacts"].get("draft_answer", "")
    
    # Redact PII
    task = a2a_bus.create_task(
        sender="orchestrator",
        recipient=compliance_agent.card.name,
        payload={"action": "enforce_policy", "length": len(draft)},
    )
    compliance_result = await compliance_agent.enforce(draft, require_approval=False)
    a2a_bus.update_task(
        task.task_id,
        status="completed",
        artifacts=[{"type": "redacted", "value": compliance_result["redacted_text"][:200]}],
    )
    redacted = compliance_result["redacted_text"]
    
    return {
        "messages": [AIMessage(content="Compliance check complete.")],
        "artifacts": {"final_answer": redacted}
    }

async def report_node(state: AgentState):
    """
    Generates PDF/DOCX reports.
    """
    print("--- Node: Report ---")
    job_id = state.get("job_id")
    await send_job_event(job_id, "reporting", 0.9, "Generating Reports")
    
    final_answer = state["artifacts"].get("final_answer", "")
    job_id = state.get("job_id")
    
    report_paths = {}
    try:
        report_paths = await synthesis_agent.export(final_answer, job_id=job_id)
    except Exception as e:
        print(f"Report generation failed: {e}")
        
    return {
        "messages": [AIMessage(content=f"Reports generated: {report_paths}")],
        "final_report": report_paths
    }

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research", research_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("compliance", compliance_node)
workflow.add_node("report", report_node)

# Edges
workflow.set_entry_point("supervisor")

# Conditional edges from supervisor
def route_step(state: AgentState):
    return state["next_step"]

workflow.add_conditional_edges(
    "supervisor",
    route_step,
    {
        "research": "research",
        "synthesis": "synthesis",
        "compliance": "compliance",
        "report": "report",
        "end": END
    }
)

# Return to supervisor after each step
workflow.add_edge("research", "supervisor")
workflow.add_edge("synthesis", "supervisor")
workflow.add_edge("compliance", "supervisor")
workflow.add_edge("report", "supervisor")

# Compile
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
