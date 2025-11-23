async def send_event(job_id: Optional[int], status: str, progress: float):
    if job_id:
        producer = KafkaProducerClient()
        await producer.start()
        try:
            event = {
                "job_id": job_id,
                "status": "running", # Keep overall status as running
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat(),
                "details": status # Add specific stage as details if needed, or map to status
            }
            # If status is completed, update overall status
            if status == "completed":
                event["status"] = "completed"
                
            await producer.send_message(TOPIC_NAME, event)
        except Exception as e:
            print(f"Failed to send event: {e}")
        finally:
            await producer.stop()
