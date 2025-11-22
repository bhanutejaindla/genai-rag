from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import asyncio

async def test():
    server_params = StdioServerParameters(
        command="python3",
        args=["/home/user/project/mcp_servers/ingestion/server.py"]  # your ingestion server
    )
    stdio, write = await stdio_client(server_params).__aenter__()
    session = await ClientSession(stdio, write).__aenter__()

    # Call tool
    result = await session.call_tool(
        "read_pdf",
        {"file_path": "/home/user/project/uploads/sample.pdf"}  # your PDF
    )
    print(result.content[0].text)

asyncio.run(test())
