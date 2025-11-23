from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from .rag import add_document
from .database import engine, create_db_and_tables
from .models import Job, User, JobStatus
from .kafka_client import consume_events, KafkaProducerClient, TOPIC_NAME

from sqlmodel import Session, select

# MCP imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_servers.ingestion.server import read_pdf, read_docx

# -------------------------------
# Create Router
# -------------------------------
router = APIRouter()


# -------------------------------
# Lifespan (runs on startup)
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    task = asyncio.create_task(consume_events())   # start Kafka consumer
    yield
    task.cancel()


app = FastAPI(
    title="Research Agent Platform API",
    lifespan=lifespan
)

# Include router
app.include_router(router)


# -------------------------------
# Request Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str


# -------------------------------
# Routes (Now using router)
# -------------------------------

@router.get("/")
async def root():
    return {"message": "Research Agent Platform API is running"}


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Create DB job
        with Session(engine) as session:
            statement = select(User).where(User.name == "demo_user")
            user = session.exec(statement).first()

            if not user:
                user = User(name="demo_user")
                session.add(user)
                session.commit()
                session.refresh(user)

            job = Job(
                type="chat",
                user_id=user.id,
                status=JobStatus.running
            )
            session.add(job)
            session.commit()
            session.refresh(job)

        # Run the agent
        from .agent import run_agent
        response = await run_agent(request.message, job_id=job.id)

        return {
            "response": response,
            "job_id": job.id
        }

    except Exception as e:
        return {"response": f"Agent Error: {str(e)}"}


@router.post("/jobs")
async def create_job():
    # Create DB job
    with Session(engine) as session:
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

    # Publish Kafka event
    producer = KafkaProducerClient()
    await producer.start()

    event = {
        "job_id": job.id,
        "status": "pending",
        "timestamp": datetime.utcnow().isoformat()
    }

    await producer.send_message(TOPIC_NAME, event)
    await producer.stop()

    return {
        "job_id": job.id,
        "status": job.status
    }


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    upload_dir = "uploads"
    file_location = os.path.join(upload_dir, file.filename)
    os.makedirs(upload_dir, exist_ok=True)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    try:
        if file.filename.endswith(".pdf"):
            text_content = await asyncio.to_thread(read_pdf, os.path.abspath(file_location))

        elif file.filename.endswith(".docx"):
            text_content = await asyncio.to_thread(read_docx, os.path.abspath(file_location))

        else:
            return {
                "message": "File saved, but type not supported",
                "path": file_location
            }

        # Store in vector DB
        chunks = await asyncio.to_thread(add_document, text_content, file.filename)

        return {
            "message": "File ingested and indexed",
            "chunks_added": chunks,
            "content_preview": text_content[:200]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )
