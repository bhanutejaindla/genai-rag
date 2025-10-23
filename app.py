import streamlit as st
import os
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_postgres import PGVector

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
    vectorstore = process_uploaded_files(uploaded_files, POSTGRES_CONNECTION_STRING)
    st.success(f"Indexed documents into PGVector. Ready to query!")

# -------------------------
# INITIALIZE TOOLS
# -------------------------
rag_tool = RAGSearchTool(connection_string=POSTGRES_CONNECTION_STRING)
reminder_tool = ReminderTool()
todo_tool = ToDoListTool()
weather_tool = WeatherTool(api_key=OPENWEATHER_API_KEY)
search_tool = SearchTool(api_key=SERPAPI_API_KEY)

tools = [
    Tool(name="RAGSearch", func=rag_tool.run, description="Answer questions from uploaded documents."),
    Tool(name="Reminder", func=reminder_tool.add, description="Add a reminder for the user."),
    Tool(name="ToDo", func=todo_tool.add, description="Manage your to-do list tasks."),
    Tool(name="Weather", func=weather_tool.get_weather, description="Get weather information for a city."),
    Tool(name="WebSearch", func=search_tool.run, description="Search the web if information is not in documents.")
]

# -------------------------
# INITIALIZE AGENT
# -------------------------
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -------------------------
# CHAT INTERFACE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Type your message here:")

if st.button("Send") and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    # Run agent on user input
    response = agent.run(user_input)

    st.session_state.history.append({"role": "bot", "content": response})

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")
