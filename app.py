# import streamlit as st
# import os
# from langchain.agents import Tool, initialize_agent
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain_postgres import PGVector

# # Import our custom tools from previous steps
# from tools import RAGSearchTool, ReminderTool, ToDoListTool, WeatherTool, SearchTool
# from rag_utils import process_uploaded_files  # function to process/upload files

# # -------------------------
# # CONFIG
# # -------------------------
# POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") 

# st.set_page_config(page_title="Smart Personal Assistant Bot", page_icon="🤖")
# st.title("🤖 Smart Personal Assistant Bot")

# # -------------------------
# # UPLOAD FILES FOR RAG
# # -------------------------
# uploaded_files = st.file_uploader(
#     "Upload documents (PDF, TXT, CSV, XLSX)",
#     accept_multiple_files=True
# )

# vectorstore = None
# if uploaded_files:
#     vectorstore = process_uploaded_files(uploaded_files, POSTGRES_CONNECTION_STRING)
#     st.success(f"Indexed documents into PGVector. Ready to query!")

# # -------------------------
# # INITIALIZE TOOLS
# # -------------------------
# rag_tool = RAGSearchTool(connection_string=POSTGRES_CONNECTION_STRING)
# reminder_tool = ReminderTool()
# todo_tool = ToDoListTool()
# weather_tool = WeatherTool(api_key=OPENWEATHER_API_KEY)
# search_tool = SearchTool(api_key=SERPAPI_API_KEY)

# tools = [
#     Tool(name="RAGSearch", func=rag_tool.run, description="Answer questions from uploaded documents."),
#     Tool(name="Reminder", func=reminder_tool.add, description="Add a reminder for the user."),
#     Tool(name="ToDo", func=todo_tool.add, description="Manage your to-do list tasks."),
#     Tool(name="Weather", func=weather_tool.get_weather, description="Get weather information for a city."),
#     Tool(name="WebSearch", func=search_tool.run, description="Search the web if information is not in documents.")
# ]

# # -------------------------
# # INITIALIZE AGENT
# # -------------------------
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# # -------------------------
# # CHAT INTERFACE
# # -------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# user_input = st.text_input("Type your message here:")

# if st.button("Send") and user_input:
#     st.session_state.history.append({"role": "user", "content": user_input})

#     # Run agent on user input
#     response = agent.run(user_input)

#     st.session_state.history.append({"role": "bot", "content": response})

# # -------------------------
# # DISPLAY CHAT HISTORY
# # -------------------------
# for chat in st.session_state.history:
#     if chat["role"] == "user":
#         st.markdown(f"**You:** {chat['content']}")
#     else:
#         st.markdown(f"**Bot:** {chat['content']}")

import streamlit as st
import os
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# Import our custom tools from previous steps
from tools import RAGSearchTool, ReminderTool, ToDoListTool, WeatherTool, SearchTool
from rag_utils import process_uploaded_files  # function to process/upload files

# -------------------------
# CONFIG
# -------------------------
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") 

st.set_page_config(page_title="Smart Personal Assistant Bot", page_icon="🤖")
st.title("🤖 Smart Personal Assistant Bot")

# -------------------------
# UPLOAD FILES FOR RAG
# -------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT, CSV, XLSX)",
    accept_multiple_files=True
)

vectorstore = None
if uploaded_files:
    with st.spinner("Processing documents..."):
        vectorstore = process_uploaded_files(uploaded_files, POSTGRES_CONNECTION_STRING)
        st.success(f"Indexed {len(uploaded_files)} document(s) into PGVector. Ready to query!")

# -------------------------
# INITIALIZE TOOLS
# -------------------------
rag_tool = RAGSearchTool(connection_string=POSTGRES_CONNECTION_STRING)
reminder_tool = ReminderTool()
todo_tool = ToDoListTool()
weather_tool = WeatherTool(api_key=OPENWEATHER_API_KEY)
search_tool = SearchTool(api_key=SERPAPI_API_KEY)

tools = [
    Tool(
        name="RAGSearch",
        func=rag_tool.run,
        description="Answer questions from uploaded documents. Use this when the user asks about content from their documents."
    ),
    Tool(
        name="Reminder",
        func=reminder_tool.add,
        description="Add a reminder for the user. Input should be the reminder text."
    ),
    Tool(
        name="ToDo",
        func=todo_tool.add,
        description="Manage your to-do list tasks. Input should be the task description."
    ),
    Tool(
        name="Weather",
        func=weather_tool.get_weather,
        description="Get current weather information for a city. Input should be the city name."
    ),
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Search the web for current information not available in documents. Input should be the search query."
    )
]

# -------------------------
# INITIALIZE AGENT
# -------------------------
@st.cache_resource
def create_agent():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful personal assistant. Use the available tools to help the user with their requests. Be concise and friendly."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent using LCEL
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False
    )
    
    return agent_executor

agent_executor = create_agent()

# -------------------------
# CHAT INTERFACE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Create columns for better layout
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input("Type your message here:", key="user_input")

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and user_input:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Show spinner while processing
    with st.spinner("🤔 Thinking..."):
        try:
            # Run agent on user input
            response = agent_executor.invoke({"input": user_input})
            bot_response = response["output"]
        except Exception as e:
            bot_response = f"I encountered an error: {str(e)}"
    
    # Add bot response to history
    st.session_state.history.append({"role": "bot", "content": bot_response})
    
    # Rerun to clear input and update display
    st.rerun()

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
st.markdown("---")
st.markdown("### Chat History")

if st.session_state.history:
    for i, chat in enumerate(st.session_state.history):
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat["content"])
else:
    st.info("👋 Start a conversation by typing a message above!")

# -------------------------
# SIDEBAR - CONTROLS AND INFO
# -------------------------
with st.sidebar:
    st.header("🛠️ Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    
    st.header("📊 Session Info")
    st.metric("Messages", len(st.session_state.history))
    
    st.markdown("---")
    
    # Display reminders if available
    st.subheader("⏰ Reminders")
    try:
        if hasattr(reminder_tool, 'get_all'):
            reminders = reminder_tool.get_all()
            if reminders:
                for idx, reminder in enumerate(reminders, 1):
                    st.text(f"{idx}. {reminder}")
            else:
                st.caption("No reminders yet")
        else:
            st.caption("Reminder tool not configured")
    except Exception as e:
        st.caption("Unable to load reminders")
    
    st.markdown("---")
    
    # Display todos if available
    st.subheader("✅ To-Do List")
    try:
        if hasattr(todo_tool, 'get_all'):
            todos = todo_tool.get_all()
            if todos:
                for idx, todo in enumerate(todos, 1):
                    st.text(f"{idx}. {todo}")
            else:
                st.caption("No tasks yet")
        else:
            st.caption("To-do tool not configured")
    except Exception as e:
        st.caption("Unable to load tasks")
    
    st.markdown("---")
    
    # Available tools
    with st.expander("🔧 Available Tools"):
        for tool in tools:
            st.markdown(f"**{tool.name}**: {tool.description}")