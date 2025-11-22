@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to /root

    await mcp_client.connect_to_server(
        "ingestion",
        "python3",
        [os.path.join(BASE_DIR, "mcp_servers", "ingestion", "server.py")]
    )

    await mcp_client.connect_to_server(
        "research",
        "python3",
        [os.path.join(BASE_DIR, "mcp_servers", "research", "server.py")]
    )

    await mcp_client.connect_to_server(
        "compliance",
        "python3",
        [os.path.join(BASE_DIR, "mcp_servers", "compliance", "server.py")]
    )

    yield
    await mcp_client.cleanup()
