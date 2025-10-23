from langchain.memory import SQLAlchemyChatMessageHistory
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine

# Postgres connection
engine = create_engine(POSTGRES_CONNECTION_STRING)

# Chat history in Postgres
chat_history = SQLAlchemyChatMessageHistory(
    session=engine.connect(),
    table_name="chat_memory"
)
