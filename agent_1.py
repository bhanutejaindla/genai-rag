report_paths = {}
    try:
        generator = ReportGenerator()
        filename = f"report_{job_id}" if job_id else f"report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        docx_path = await asyncio.to_thread(generator.generate_docx, final_answer, filename)
        pdf_path = await asyncio.to_thread(generator.generate_pdf, final_answer, filename)
        
        report_paths = {
            "docx": docx_path,
            "pdf": pdf_path
        }
        print(f"Reports generated: {report_paths}")
    except Exception as e:
        print(f"Report generation failed: {e}")

    print("\n--- Agent Finished ---")
    await send_event(job_id, "completed", 1.0)
    
    return {
        "answer": final_answer,
        "reports": report_paths
    }
