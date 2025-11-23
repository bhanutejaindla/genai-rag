@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Ingest a document:
    1. Save file
    2. Extract text (PDF/DOCX)
    3. Split into chunks & store in PGVector
    4. Track job steps + progress
    """

    # 1️⃣ Create an ingestion job
    job = Job(
        name=f"Document Ingestion – {file.filename}",
        type="ingestion",
        status=JobStatus.running,
        user_id=current_user.id,
        progress=0.0,
        tasks=[{"step": "upload_received", "status": "completed"}]
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    upload_dir = "uploads"
    file_location = os.path.join(upload_dir, file.filename)
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # 2️⃣ Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job.progress = 0.20
        job.tasks.append({"step": "file_saved", "status": "completed", "path": file_location})
        session.add(job)
        session.commit()

        # 3️⃣ Extract text using MCP ingestion tools
        if file.filename.endswith(".pdf"):
            text_content = await asyncio.to_thread(read_pdf, os.path.abspath(file_location))
        elif file.filename.endswith(".docx"):
            text_content = await asyncio.to_thread(read_docx, os.path.abspath(file_location))
        else:
            job.status = JobStatus.failed
            job.tasks.append({
                "step": "text_extraction", 
                "status": "failed", 
                "error": "Unsupported file type"
            })
            session.commit()
            return {"message": "Unsupported file type", "job_id": job.id}

        job.progress = 0.60
        job.tasks.append({
            "step": "text_extracted", 
            "status": "completed", 
            "characters": len(text_content)
        })
        session.commit()

        # 4️⃣ Add document to PGVector
        chunks_added = await asyncio.to_thread(add_document, text_content, source=file.filename)

        job.progress = 1.0
        job.status = JobStatus.completed
        job.tasks.append({
            "step": "vector_indexing",
            "status": "completed",
            "chunks": chunks_added
        })
        session.commit()

        # 5️⃣ Return response
        return {
            "message": "File ingested successfully",
            "job_id": job.id,
            "chunks_added": chunks_added,
            "content_preview": text_content[:200]
        }

    except Exception as e:
        job.status = JobStatus.failed
        job.progress = 0.0
        job.tasks.append({
            "step": "ingestion_failed",
            "status": "failed",
            "error": str(e)
        })
        session.commit()

        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
