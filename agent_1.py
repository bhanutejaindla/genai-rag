import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def test():
    server_params = StdioServerParameters(
        command="python3",
        args=["/home/user/project/mcp_servers/ingestion/server.py"]  # your ingestion server
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            
            # Call the read_pdf tool
            result = await session.call_tool(
                "read_pdf",
                {"file_path": "/home/user/project/uploads/sample.pdf"}
            )
            print(result.content[0].text)

asyncio.run(test())
