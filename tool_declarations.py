from langchain.agents import Tool
from tools.rag_search_tool import RAGSearchTool
from tools.remainder_tool import ReminderTool
from tools.to_do_list_tool import ToDoTool
from tools.weather_tool import WeatherTool
from tools.search_tool import SearchTool

from dotenv import load_dotenv
load_dotenv()
import os 

tools = [
    Tool(
        name="RAGSearch",
        func=RAGSearchTool().run,
        description="Use this tool to answer questions based on uploaded documents."
    ),
    Tool(
        name="Reminder",
        func=ReminderTool().add,
        description="Add a reminder for the user."
    ),
    Tool(
        name="ToDo",
        func=ToDoTool().add,
        description="Manage to-do list tasks."
    ),
    Tool(
        name="Weather",
        func=WeatherTool(os.getenv("OPEN_WEATHER_API_KEY")).get_weather,
        description="Get real-time weather information for a city."
    ),
    Tool(
        name="WebSearch",
        func=SearchTool(os.getenv("SERP_API_KEY")).run,
        description="Search the web for information not in uploaded documents."
    )
]
