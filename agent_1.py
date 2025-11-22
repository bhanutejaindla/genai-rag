from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import asyncio
from .rag import add_document
# Direct imports from MCP servers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_servers.ingestion.server import read_pdf, read_docx

app = FastAPI(title="Research Agent Platform API")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Research Agent Platform API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Use Linear Agent Pipeline with direct calls
        from .agent import run_agent
        response = await run_agent(request.message)
        return {"response": response}
    except Exception as e:
        # Fallback if LLM/Agent fails
        return {"response": f"Agent Error: {str(e)}. (Ensure OpenAI Key is set for this demo)"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    upload_dir = "uploads"
    file_location = os.path.join(upload_dir, file.filename)
    os.makedirs(upload_dir, exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger ingestion tool directly
    try:
        text_content = ""
        if file.filename.endswith(".pdf"):
            text_content = await asyncio.to_thread(read_pdf, os.path.abspath(file_location))
        elif file.filename.endswith(".docx"):
            text_content = await asyncio.to_thread(read_docx, os.path.abspath(file_location))
        else:
            return {"message": "File saved, but type not supported for extraction.", "path": file_location}
            
        # Index in Vector DB
        num_chunks = await asyncio.to_thread(add_document, text_content, source=file.filename)
            
        return {
            "message": "File ingested and indexed successfully", 
            "chunks_added": num_chunks,
            "content_preview": text_content[:200]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
