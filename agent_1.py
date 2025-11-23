async def ingest_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # --- 1. Create Job ---
    job = Job(
        name=f"Ingest {file.filename}",
        type="ingestion",
        status=JobStatus.running,
        user_id=1,          # TEMP — change after adding auth
        progress=0.0,
        tasks=[]
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # --- 2. Save uploaded file ---
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job.tasks.append({"step": "file_saved", "path": file_path})
        job.progress = 0.2
        db.commit()

        # --- 3. Extract text ---
        if file.filename.endswith(".pdf"):
            text = await asyncio.to_thread(read_pdf, file_path)
        elif file.filename.endswith(".docx"):
            text = await asyncio.to_thread(read_docx, file_path)
        else:
            job.status = JobStatus.failed
            db.commit()
            raise HTTPException(400, "Unsupported file type")

        job.tasks.append({"step": "text_extracted", "size": len(text)})
        job.progress = 0.6
        db.commit()

        # --- 4. Add to vector DB ---
        chunks = await asyncio.to_thread(add_document, text, file.filename)

        job.tasks.append({"step": "indexed_chunks", "chunks": chunks})
        job.status = JobStatus.completed
        job.progress = 1.0
        db.commit()

        return {
            "message": "Ingest successful",
            "job_id": job.id,
            "chunks_added": chunks,
            "preview": text[:200]
        }

    except Exception as e:
        job.status = JobStatus.failed
        job.tasks.append({"step": "error", "error": str(e)})
        db.commit()
        raise HTTPException(500, f"Ingestion failed: {e}")
