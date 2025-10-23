# rag_utils.py
import os
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def process_uploaded_files(files, connection_string, collection_name="rag_docs"):
    """Process and upload files to PGVector"""
    try:
        documents = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load based on file type
                if file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                    documents.extend(loader.load())
                elif file.name.endswith('.txt'):
                    loader = TextLoader(tmp_path)
                    documents.extend(loader.load())
                else:
                    print(f"Unsupported file type: {file.name}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        if not documents:
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vectorstore
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            connection=connection_string,
        )
        
        return vectorstore
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None