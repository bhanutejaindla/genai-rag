import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, name: str, command: str, args: list[str], env: dict = None):
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        
        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self.sessions[name] = session
        print(f"Connected to MCP server: {name}")

    async def list_tools(self, server_name: str):
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        return await self.sessions[server_name].list_tools()

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        return await self.sessions[server_name].call_tool(tool_name, arguments)

    async def cleanup(self):
        await self.exit_stack.aclose()
