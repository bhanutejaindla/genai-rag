CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)  -- or 3072 depending on model
);

import os
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PostgreSQL Connection
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/mydb"
)

conn = psycopg2.connect(DB_URL)
register_vector(conn)
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source TEXT,
    chunk TEXT,
    embedding VECTOR(1536)
);
""")
conn.commit()


# ------------------------
# EMBEDDING FUNCTION
# ------------------------
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ------------------------
# ADD DOCUMENT (Same style as old)
# ------------------------
def add_document(text: str, source: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        return 0

    for chunk in chunks:
        embedding = get_embedding(chunk)

        cur.execute("""
            INSERT INTO documents (source, chunk, embedding)
            VALUES (%s, %s, %s)
        """, (source, chunk, embedding))

    conn.commit()
    return len(chunks)


# ------------------------
# QUERY DOCUMENTS (Same return as old code)
# ------------------------
def query_documents(query: str, n_results: int = 5):
    query_emb = get_embedding(query)

    cur.execute("""
        SELECT source, chunk, (embedding <-> %s) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s
    """, (query_emb, n_results))

    rows = cur.fetchall()

    context = ""

    for src, chunk, dist in rows:
        context += f"[Source: {src}]\n{chunk}\n\n"

    return context
