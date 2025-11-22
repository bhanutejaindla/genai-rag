from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from contextlib import asynccontextmanager
from .mcp_client import MCPClient
from .rag import add_document

# Global MCP Client
mcp_client = MCPClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MCP servers
    cwd = os.getcwd()
    await mcp_client.connect_to_server(
        "ingestion", 
        "python3", 
        [f"{cwd}/mcp_servers/ingestion/server.py"]
    )
    await mcp_client.connect_to_server(
        "research", 
        "python3", 
        [f"{cwd}/mcp_servers/research/server.py"]
    )
    await mcp_client.connect_to_server(
        "compliance", 
        "python3", 
        [f"{cwd}/mcp_servers/compliance/server.py"]
    )
    yield
    await mcp_client.cleanup()

app = FastAPI(title="Research Agent Platform API", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Research Agent Platform API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Use Linear Agent Pipeline
        from .agent import run_agent
        response = await run_agent(request.message, mcp_client)
        return {"response": response}
    except Exception as e:
        # Fallback if LLM/Agent fails
        return {"response": f"Agent Error: {str(e)}. (Ensure OpenAI Key is set for this demo)"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger ingestion tool
    try:
        text_content = ""
        if file.filename.endswith(".pdf"):
            result = await mcp_client.call_tool("ingestion", "read_pdf", {"file_path": os.path.abspath(file_location)})
            text_content = result.content[0].text
        elif file.filename.endswith(".docx"):
            result = await mcp_client.call_tool("ingestion", "read_docx", {"file_path": os.path.abspath(file_location)})
            text_content = result.content[0].text
        else:
            return {"message": "File saved, but type not supported for extraction.", "path": file_location}
            
        # Index in Vector DB
        num_chunks = add_document(text_content, source=file.filename)
            
        return {
            "message": "File ingested and indexed successfully", 
            "chunks_added": num_chunks,
            "content_preview": text_content[:200]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
