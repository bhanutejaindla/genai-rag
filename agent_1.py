import asyncio
from backend.graph import graph  # adjust path if needed
from langchain_core.messages import HumanMessage

async def test():
    print("\n==== RUNNING GRAPH TEST ====\n")

    # Build initial state
    initial_state = {
        "messages": [HumanMessage(content="Explain the impact of AI in healthcare")],
        "next_step": "start",
        "job_id": None,
        "artifacts": {},
        "research_data": {},
        "final_report": {}
    }

    # Run the graph
    result = await graph.ainvoke(initial_state)

    print("\n==== FINAL GRAPH STATE ====\n")
    print(result)

    print("\n==== FINAL ANSWER ====\n")
    print(result.get("artifacts", {}).get("final_answer", "(No answer)"))

    print("\n==== REPORT PATHS ====\n")
    print(result.get("final_report", {}))


if __name__ == "__main__":
    asyncio.run(test())
