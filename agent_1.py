async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    raw = await asyncio.to_thread(web_search, query, max_results=max_results)

    # Case 1: Already structured list
    if isinstance(raw, list):
        return [
            {
                "title": item.get("title", "Untitled"),
                "quote": item.get("snippet") or item.get("description", ""),
                "url": item.get("url", ""),
                "date": item.get("date", "")
            }
            for item in raw
        ]

    # Case 2: Raw string → convert to list
    if isinstance(raw, str):
        blocks = raw.split("\n\n")
        structured = []
        for i, block in enumerate(blocks):
            structured.append({
                "title": f"Result {i+1}",
                "quote": block.strip(),
                "url": "",
                "date": ""
            })
        return structured

    return []
