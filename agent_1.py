# 7. Generate Reports
    print("7. Generating Reports...")
    generator = ReportGenerator()
    filename = f"report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    docx_path = await asyncio.to_thread(generator.generate_docx, final_answer, filename)
    pdf_path = await asyncio.to_thread(generator.generate_pdf, final_answer, filename)

    return {
        "answer": final_answer,
        "reports": {"docx": docx_path, "pdf": pdf_path}
    }
