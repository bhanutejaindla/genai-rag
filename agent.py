from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from tool_declarations import tools

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # LLM decides which tool to call
    verbose=True  # Shows tool usage and reasoning
)
