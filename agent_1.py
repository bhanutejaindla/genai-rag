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


print("1. Retrieving Context...")
    await send_event(job_id, "retrieving_context", 0.2)


 # 2. Web Research
    print("2. Performing Web Research...")
    await send_event(job_id, "web_research", 0.35)
# 3. Synthesize Answer
    print("3. Synthesizing Answer...")
    await send_event(job_id, "drafting", 0.55)
# 4. Verify Citations
    print("4. Verifying Citations...")
    await send_event(job_id, "verifying", 0.7)

if not verification['is_valid'] or verification['score'] < 0.8:
        print("\n5. Refining Answer...")
        await send_event(job_id, "refining", 0.85)
    print("\n--- Agent Finished ---")
    await send_event(job_id, "completed", 1.0)
