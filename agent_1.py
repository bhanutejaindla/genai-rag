from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("citation_validation")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@mcp.tool()
async def validate_citation(draft_answer: str, context: str, web_results: str) -> str:
    """
    Validates the draft answer for accuracy and proper citation usage based on the sources.
    Checks for temporal accuracy (present vs past) and hallucinations.
    """
    verification_prompt = ChatPromptTemplate.from_template(
        """Verify the following answer for accuracy and proper citation usage based on the sources.
        
        CRITICAL INSTRUCTION:
        - You must strictly adhere to the provided sources.
        - Pay special attention to TEMPORAL ACCURACY. If the query asks for the "present" or "current" status, ensure the answer reflects the most recent information from the sources.
        - If the sources contradict the draft answer (e.g., draft says X is current, but sources say Y is current), you MUST point this out.
        - Check for hallucinations.
        
        Draft Answer: {draft_answer}
        
        Sources:
        {context}
        {web_results}
        
        Critique (List any missing citations, factual errors, or hallucinations, or say 'LGTM' if the answer is accurate and supported):"""
    )
    chain = verification_prompt | llm
    critique = await chain.ainvoke({
        "draft_answer": draft_answer,
        "context": context,
        "web_results": web_results
    })
    return critique.content

if __name__ == "__main__":
    mcp.run()
