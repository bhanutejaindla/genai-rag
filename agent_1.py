import os
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Postgres DSN
CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
)

COLLECTION_NAME = "research_docs"

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Vector Store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

def add_document(text: str, source: str):
    """
    Chunks and adds a document to PGVector.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    if not chunks:
        return 0

    metadatas = [{"source": source} for _ in chunks]

    vector_store.add_texts(texts=chunks, metadatas=metadatas)

    return len(chunks)

def query_documents(query: str, n_results: int = 5):
    """
    Queries PGVector for relevant context.
    """
    results = vector_store.similarity_search(query, k=n_results)

    context = ""
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        context += f"[Source: {source}]\n{doc.page_content}\n\n"

    return context
