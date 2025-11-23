from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import asyncio
from .rag import add_document
from .database import engine
from datetime import datetime
# Direct imports from MCP servers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_servers.ingestion.server import read_pdf, read_docx
from contextlib import asynccontextmanager
from .database import create_db_and_tables, get_session
from .models import Job, User, JobStatus
from .kafka_client import consume_events, KafkaProducerClient, TOPIC_NAME
from sqlmodel import Session, select

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # Start Kafka Consumer in background
    task = asyncio.create_task(consume_events())
    yield
    task.cancel()

app = FastAPI(title="Research Agent Platform API", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Research Agent Platform API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Create a new job in DB for this chat request
        with Session(engine) as session:
            # Ensure a user exists for this demo
            statement = select(User).where(User.name == "demo_user")
            user = session.exec(statement).first()
            if not user:
                user = User(name="demo_user")
                session.add(user)
                session.commit()
                session.refresh(user)
                
            job = Job(type="chat", user_id=user.id, status=JobStatus.running)
            session.add(job)
            session.commit()
            session.refresh(job)
            
        # Use Linear Agent Pipeline with direct calls
        from .agent import run_agent
        response = await run_agent(request.message, job_id=job.id)
        return {"response": response, "job_id": job.id}
    except Exception as e:
        # Fallback if LLM/Agent fails
        return {"response": f"Agent Error: {str(e)}. (Ensure OpenAI Key is set for this demo)"}

@app.post("/jobs")
async def create_job():
    # Create a new job in DB
    with Session(engine) as session:
        # Ensure a user exists for this demo
        statement = select(User).where(User.name == "demo_user")
        user = session.exec(statement).first()
        if not user:
            user = User(name="demo_user")
            session.add(user)
            session.commit()
            session.refresh(user)
            
        job = Job(type="research", user_id=user.id)
        session.add(job)
        session.commit()
        session.refresh(job)
    
    # Send initial event
    producer = KafkaProducerClient()
    await producer.start()
    try:
        event = {
            "job_id": job.id,
            "status": "pending",
            "timestamp": datetime.utcnow().isoformat()
        }
        await producer.send_message(TOPIC_NAME, event)
    finally:
        await producer.stop()
        
    return {"job_id": job.id, "status": job.status}

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
