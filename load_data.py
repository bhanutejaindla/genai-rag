"""
RAG index builder using PostgreSQL + pgvector

Supports: .pdf, .txt, .csv, .xlsx
Embeds with OpenAIEmbeddings and stores in pgvector DB.
"""

from typing import List, Optional
import os
import pandas as pd
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_postgres import PGVector


def _load_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def _load_text_from_txt(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def _load_text_from_csv(path: str) -> str:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_csv(index=False)


def _load_text_from_xlsx(path: str) -> str:
    xls = pd.read_excel(path, sheet_name=None, dtype=str)
    text = ""
    for name, df in xls.items():
        text += f"\n--- Sheet: {name} ---\n"
        text += df.to_csv(index=False)
    return text


def _load_file_to_document(path: str) -> Document:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        txt = _load_text_from_pdf(path)
    elif ext == ".txt":
        txt = _load_text_from_txt(path)
    elif ext == ".csv":
        txt = _load_text_from_csv(path)
    elif ext in (".xls", ".xlsx"):
        txt = _load_text_from_xlsx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return Document(page_content=txt, metadata={"source": path})


def create_rag_index_pgvector(
    file_paths: List[str],
    collection_name: str = "rag_docs",
    connection_string: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    openai_api_key: Optional[str] = None,
):
    """
    Create a RAG index using Postgres + pgvector.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    if connection_string is None:
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

    if not connection_string:
        raise ValueError("Postgres connection string not provided or found in environment.")

    # Load files
    docs = []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        doc = _load_file_to_document(path)
        if doc.page_content.strip():
            docs.append(doc)

    if not docs:
        raise ValueError("No valid documents loaded!")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for d in docs:
        for i, chunk in enumerate(splitter.split_text(d.page_content)):
            split_docs.append(Document(page_content=chunk, metadata={**d.metadata, "chunk": i}))

    # Create embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Create or connect to PGVector store
    vectorstore = PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        use_jsonb=True,  # allows metadata storage
    )

    print(f"✅ Successfully stored {len(split_docs)} chunks into pgvector collection '{collection_name}'")
    return vectorstore


# Example usage:
# if __name__ == "__main__":
#     db = create_rag_index_pgvector(
#         ["docs/sample.pdf", "data/notes.txt"],
#         connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )
