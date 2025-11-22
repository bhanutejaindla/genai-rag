from vectordb import InMemoryExactNNVectorDB
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import openai
import numpy as np

load_dotenv()

# Define Document Schema
class ResearchDoc(BaseDoc):
    text: str
    source: str
    embedding: NdArray[1536]  # OpenAI embedding size


# Initialize VectorDB
db = InMemoryExactNNVectorDB[ResearchDoc](
    workspace="./vectordb_workspace"
)

# OpenAI Client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def add_document(text: str, source: str):
    """
    Chunks and adds a document to the vector DB.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        return 0

    docs = DocList[ResearchDoc]()

    for chunk in chunks:
        embedding = get_embedding(chunk)
        docs.append(
            ResearchDoc(
                text=chunk,
                source=source,
                embedding=np.array(embedding, dtype=np.float32)
            )
        )

    db.index(docs)
    return len(chunks)


def query_documents(query: str, n_results: int = 5):
    """
    Searches vectordb using a query doc with an embedding.
    """
    query_embedding = get_embedding(query)

    query_doc = ResearchDoc(
        text=query,
        source="user_query",
        embedding=np.array(query_embedding, dtype=np.float32)
    )

    # Correct search call
    results = db.search(
        query=query_doc,
        limit=n_results
    )

    context = ""

    # results[0] contains matches for the query doc
    if len(results) > 0:
        for m in results[0].matches:
            context += f"[Source: {m.source}]\n{m.text}\n\n"

    return context
