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
from mcp_servers.citation_validation.server import verify_citations_internal, parse_web_search_results

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def run_agent(query: str):
    """Enhanced Agent Pipeline with Citation Verification"""
    print(f"--- Starting Agent for Query: {query} ---")
    
    # 1. Retrieve Context
    print("1. Retrieving Context...")
    context = await asyncio.to_thread(query_documents, query)
    
    # 2. Web Research
    print("2. Performing Web Research...")
    web_results = ""
    try:
        web_results = await asyncio.to_thread(web_search, query, max_results=5)
        print(f"✓ Found {len(web_results.split('\\n\\n'))} web results")
    except Exception as e:
        web_results = f"Search failed: {e}"

    # 3. Synthesize Answer
    print("3. Synthesizing Answer...")
    synthesis_prompt = ChatPromptTemplate.from_template(
        """You are a research analyst. Answer the query using the provided sources.

Query: {query}

Web Search Results (USE THESE for current information):
{web_results}

Internal Documents:
{context}

IMPORTANT INSTRUCTIONS:
- Prioritize web search results for current/recent information
- Cite sources using [1], [2], etc. matching the numbered sources above
- Every factual claim MUST have a citation
- Example: "The current CM is Revanth Reddy [1]"

Answer with citations:"""
    )
    chain = synthesis_prompt | llm
    draft_answer = await chain.ainvoke({
        "query": query,
        "context": context,
        "web_results": web_results
    })
    draft_answer = draft_answer.content
    print(f"Draft: {draft_answer[:100]}...")
    
    # 4. Verify Citations
    print("4. Verifying Citations...")
    
    # Parse web results into structured sources
    sources = await asyncio.to_thread(parse_web_search_results, web_results)
    
    # Add context as additional source
    if context:
        sources.append({
            'id': 'internal',
            'title': 'Internal Documents',
            'text': context,
            'url': 'internal'
        })
    
    # Run verification
    verification = await asyncio.to_thread(
        verify_citations_internal,
        draft_answer,
        sources,
        strict_mode=False
    )
    
    print(f"\n{verification['summary']}")
    print(f"Score: {verification['score']} ({verification['supported_claims']}/{verification['total_claims']} claims supported)")
    
    if verification['issues']:
        print(f"\n⚠️ Found {len(verification['issues'])} issues:")
        for issue in verification['issues'][:3]:  # Show first 3
            print(f"  {issue}")
    
    # 5. Refine Answer if needed
    final_answer = draft_answer
    
    if not verification['is_valid'] or verification['score'] < 0.8:
        print("\n5. Refining Answer...")
        
        refine_prompt = ChatPromptTemplate.from_template(
            """Fix the citation issues in this answer.

ORIGINAL QUERY: {query}

ORIGINAL ANSWER: 
{draft_answer}

VERIFICATION ISSUES:
{issues}

SOURCES AVAILABLE:
{web_results}

INSTRUCTIONS:
- Fix all issues mentioned above
- Ensure EVERY factual claim has a citation [1], [2], etc.
- Use ONLY information from the sources provided
- For current information (like "current CM"), use the web search results
- Do not introduce information from your training data

Corrected Answer:"""
        )
        chain = refine_prompt | llm
        refined = await chain.ainvoke({
            "query": query,
            "draft_answer": draft_answer,
            "issues": "\n".join(verification['issues']),
            "web_results": web_results
        })
        final_answer = refined.content
        print(f"Refined: {final_answer[:100]}...")
        
        # Verify again
        verification2 = await asyncio.to_thread(
            verify_citations_internal,
            final_answer,
            sources,
            strict_mode=False
        )
        print(f"\nRe-verification: {verification2['summary']}")
    else:
        print("\n5. ✅ No refinement needed")
    
    # 6. Compliance Check
    print("\n6. Checking Compliance...")
    try:
        final_answer = await asyncio.to_thread(redact_pii, final_answer)
    except Exception as e:
        print(f"Compliance check failed: {e}")
        
    print("\n--- Agent Finished ---")
    return final_answer
