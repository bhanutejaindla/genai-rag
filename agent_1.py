@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MCP servers
    cwd = os.getcwd()
    await mcp_client.connect_to_server(
        "ingestion", 
        "python3", 
        [f"{cwd}/mcp_servers/ingestion/server.py"]
    )
    await mcp_client.connect_to_server(
        "research", 
        "python3", 
        [f"{cwd}/mcp_servers/research/server.py"]
    )
    await mcp_client.connect_to_server(
        "compliance", 
        "python3", 
        [f"{cwd}/mcp_servers/compliance/server.py"]
    )
    yield
    await mcp_client.cleanup()
