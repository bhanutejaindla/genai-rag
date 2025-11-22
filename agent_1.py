from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from .rag import query_documents
import asyncio

# Direct imports from MCP server files (Python functions)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_servers.research.server import web_search
from mcp_servers.compliance.server import redact_pii
from mcp_servers.citation_validation.server import validate_citation

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def run_agent(query: str):
    """
    Linear Agent Pipeline:
    1. Retrieve Context (RAG)
    2. Web Research (Direct Call)
    3. Synthesize Answer (LLM)
    4. Verify Answer (LLM)
    5. Refine Answer (LLM)
    6. Compliance Check (Direct Call)
    """
    print(f"--- Starting Agent for Query: {query} ---")
    
    # 1. Retrieve Context
    print("1. Retrieving Context...")
    # Run blocking RAG query in a separate thread
    context = await asyncio.to_thread(query_documents, query)
    
    # 2. Web Research
    print("2. Performing Web Research...")
    web_results = ""
    try:
        # Run blocking web search in a separate thread
        web_results = await asyncio.to_thread(web_search, query)
    except Exception as e:
        web_results = f"Search failed: {e}"

    # 3. Synthesize Answer
    print("3. Synthesizing Answer...")
    synthesis_prompt = ChatPromptTemplate.from_template(
        """You are a research analyst. Answer the query based on the provided context and web results.
        
        Query: {query}
        
        Internal Documents (Context):
        {context}
        
        Web Search Results:
        {web_results}
        
        Answer:"""
    )
    chain = synthesis_prompt | llm
    draft_answer = await chain.ainvoke({
        "query": query,
        "context": context,
        "web_results": web_results
    })
    draft_answer = draft_answer.content
    
    # 4. Verify Answer
    print("4. Verifying Answer...")
    try:
        critique = await validate_citation(draft_answer, context, web_results)
    except Exception as e:
        print(f"Verification failed: {e}")
        critique = f"Verification failed: {e}"
    
    # 5. Refine Answer
    print("5. Refining Answer...")
    final_answer = draft_answer
    if "LGTM" not in critique:
        refine_prompt = ChatPromptTemplate.from_template(
            """Refine the answer based on the critique.
            
            Original Answer: {draft_answer}
            Critique: {critique}
            
            Refined Answer:"""
        )
        chain = refine_prompt | llm
        refined_response = await chain.ainvoke({
            "draft_answer": draft_answer,
            "critique": critique
        })
        final_answer = refined_response.content
        
    # 6. Compliance Check
    print("6. Checking Compliance...")
    try:
        # Run blocking compliance check in a separate thread
        final_answer = await asyncio.to_thread(redact_pii, final_answer)
    except Exception as e:
        print(f"Compliance check failed: {e}")
        
    print("--- Agent Finished ---")
    return final_answer
