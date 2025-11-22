from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from .rag import query_documents
from .mcp_client import MCPClient

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

async def run_agent(query: str, mcp_client: MCPClient):
    """
    Linear Agent Pipeline:
    1. Retrieve Context (RAG)
    2. Web Research (MCP)
    3. Synthesize Answer (LLM)
    4. Verify Answer (LLM)
    5. Refine Answer (LLM)
    6. Compliance Check (MCP)
    """
    print(f"--- Starting Agent for Query: {query} ---")
    
    # 1. Retrieve Context
    print("1. Retrieving Context...")
    context = query_documents(query)
    
    # 2. Web Research
    print("2. Performing Web Research...")
    web_results = ""
    try:
        if "research" in mcp_client.sessions:
            result = await mcp_client.call_tool("research", "web_search", {"query": query})
            web_results = result.content[0].text
        else:
            web_results = "Web search unavailable (client not connected)"
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
    draft_answer = chain.invoke({
        "query": query,
        "context": context,
        "web_results": web_results
    }).content
    
    # 4. Verify Answer
    print("4. Verifying Answer...")
    verification_prompt = ChatPromptTemplate.from_template(
        """Verify the following answer for accuracy and proper citation usage based on the sources.
        
        Answer: {draft_answer}
        
        Sources:
        {context}
        {web_results}
        
        Critique (List any missing citations or hallucinations, or say 'LGTM'):"""
    )
    chain = verification_prompt | llm
    critique = chain.invoke({
        "draft_answer": draft_answer,
        "context": context,
        "web_results": web_results
    }).content
    
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
        final_answer = chain.invoke({
            "draft_answer": draft_answer,
            "critique": critique
        }).content
        
    # 6. Compliance Check
    print("6. Checking Compliance...")
    try:
        if "compliance" in mcp_client.sessions:
            result = await mcp_client.call_tool("compliance", "redact_pii", {"text": final_answer})
            final_answer = result.content[0].text
    except Exception as e:
        print(f"Compliance check failed: {e}")
        
    print("--- Agent Finished ---")
    return final_answer
