# MCP (Model Context Protocol) Integration with LangGraph - Comprehensive Guide

**Version:** 2025-03
**Author:** Research Compilation
**Status:** Complete Integration Guide

---

## Table of Contents

1. [Working Examples of MCP with Python/LangGraph](#1-working-examples)
2. [Connecting MCP Server to LangGraph Node](#2-connection-steps)
3. [Tool Discovery and Dynamic Registration](#3-tool-discovery)
4. [Error Handling Patterns](#4-error-handling)
5. [Running Multiple MCP Servers with Docker](#5-docker-setup)
6. [Authentication and Credentials](#6-authentication)
7. [Streaming Responses](#7-streaming-responses)

---

## 1. Working Examples of MCP with Python/LangGraph {#1-working-examples}

### Example 1: Basic Math Tools Server with LangGraph Agent

#### Server Implementation (math_server.py)

```python
from mcp.server.fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("MathTools")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    try:
        result = a + b
        logger.info(f"Addition: {a} + {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"Addition failed: {e}")
        raise

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    try:
        result = a * b
        logger.info(f"Multiplication: {a} * {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"Multiplication failed: {e}")
        raise

if __name__ == "__main__":
    # For stdio transport
    mcp.run(transport="stdio")

    # OR for HTTP transport
    # mcp.run(
    #     transport="streamable-http",
    #     host="0.0.0.0",
    #     port=8000
    # )
```

#### LangGraph Client Implementation

```python
import asyncio
from langchain_mcp_adapters import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_mcp_client():
    """Initialize a client with connections to multiple MCP servers."""

    server_configs = {
        "math_tools": {
            "command": "python",
            "args": ["math_server.py"],
            "transport": "stdio"
        },
        "weather_service": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http"
        }
    }

    try:
        client = MultiServerMCPClient(server_configs)
        await asyncio.wait_for(client.connect(), timeout=30.0)
        logger.info("Successfully connected to all MCP servers")
        return client
    except asyncio.TimeoutError:
        logger.error("Connection timeout after 30 seconds")
        raise
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        raise

async def create_langgraph_agent():
    """Set up a LangGraph agent with MCP tools."""

    mcp_client = await setup_mcp_client()

    try:
        # Get all tools from connected MCP servers
        tools = await mcp_client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from MCP servers")

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            timeout=60.0
        )

        # Create LangGraph agent with MCP tools
        agent = create_react_agent(
            model=llm,
            tools=tools,
            debug=True
        )

        return agent, mcp_client
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        await mcp_client.disconnect()
        raise

async def run_agent_example():
    """Run example queries using the LangGraph agent."""

    agent, mcp_client = await create_langgraph_agent()

    try:
        # Example 1: Math operations
        math_query = "Calculate (15 + 25) * 3 and tell me the result"
        math_result = await agent.ainvoke({"messages": [("user", math_query)]})
        print(f"Math Result: {math_result}")

        # Example 2: Weather queries
        weather_query = "What's the weather like in San Francisco?"
        weather_result = await agent.ainvoke({"messages": [("user", weather_query)]})
        print(f"Weather Result: {weather_result}")

        # Example 3: Combined operations
        combined_query = "If sunny in Miami, multiply 12 by 8, else add 10 and 5"
        combined_result = await agent.ainvoke({"messages": [("user", combined_query)]})
        print(f"Combined Result: {combined_result}")
    finally:
        await mcp_client.disconnect()

# Run the example
if __name__ == "__main__":
    asyncio.run(run_agent_example())
```

### Example 2: Weather Service with Async Tools

```python
from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP("WeatherService")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Retrieve weather information for a given location."""
    try:
        # Simulate API call
        await asyncio.sleep(0.1)
        weather_data = f"Current weather in {location}: 72°F, partly cloudy"
        return weather_data
    except Exception as e:
        raise Exception(f"Weather request failed: {e}")

@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
    """Access weather data as a resource."""
    return f"Weather resource for {location}"

@mcp.prompt()
def weather_prompt(location: str) -> str:
    """Create a weather query prompt."""
    return f"Please provide detailed weather information for {location}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Example 3: Multi-Server LangGraph Integration

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

async def create_custom_graph():
    """Create a custom LangGraph with MCP tools."""

    # Initialize MCP client
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/path/to/math_server.py"]
        },
        "weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp"
        }
    })

    # Get tools
    tools = await client.get_tools()

    # Create graph
    workflow = StateGraph(AgentState)

    # Define nodes
    async def call_model(state: AgentState):
        # Your model logic here
        pass

    async def call_tools(state: AgentState):
        # Tool execution logic
        pass

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_edge("tools", "agent")

    # Compile
    app = workflow.compile()

    return app, client
```

---

## 2. Connecting MCP Server to LangGraph Node {#2-connection-steps}

### Step-by-Step Connection Guide

#### Step 1: Install Required Packages

```bash
# Install core packages
pip install langchain-mcp-adapters langgraph "langchain[openai]"

# For custom server development
pip install mcp

# Optional: For async operations
pip install asyncio
```

#### Step 2: Create MCP Server

Choose your transport method:

**Option A: stdio Transport (Local Development)**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Option B: Streamable HTTP Transport (Production)**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000
    )
```

#### Step 3: Configure Client Connection

```python
from langchain_mcp_adapters import MultiServerMCPClient

# Single server configuration
single_server = {
    "my_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["path/to/server.py"]
    }
}

# Multiple servers configuration
multi_servers = {
    "server1": {
        "transport": "stdio",
        "command": "python",
        "args": ["server1.py"]
    },
    "server2": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp"
    },
    "server3": {
        "transport": "sse",
        "url": "http://localhost:8001/sse"
    }
}

client = MultiServerMCPClient(multi_servers)
```

#### Step 4: Initialize Connection

```python
import asyncio

async def initialize_client():
    """Initialize and connect to MCP servers."""
    client = MultiServerMCPClient(server_configs)

    # Connect with timeout
    await asyncio.wait_for(client.connect(), timeout=30.0)

    return client
```

#### Step 5: Load Tools for LangGraph

```python
from langchain_mcp_adapters import load_mcp_tools

async def load_tools():
    """Load tools from MCP servers."""
    client = await initialize_client()

    # Option 1: Get all tools (stateless)
    tools = await client.get_tools()

    # Option 2: Use persistent session (stateful)
    async with client.session("server_name") as session:
        tools = await load_mcp_tools(session)

    return tools, client
```

#### Step 6: Create LangGraph Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def create_agent():
    """Create LangGraph agent with MCP tools."""

    # Load tools
    tools, client = await load_tools()

    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    # Create agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        debug=True
    )

    return agent, client
```

#### Step 7: Integrate with Custom LangGraph Node

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    messages: list
    mcp_results: dict

async def mcp_tool_node(state: State):
    """Custom node that uses MCP tools."""

    # Get tools
    tools, client = await load_tools()

    # Execute tool
    tool_name = "my_tool"
    tool = next(t for t in tools if t.name == tool_name)

    result = await tool.ainvoke({"param": "value"})

    state["mcp_results"] = result
    return state

# Create graph
workflow = StateGraph(State)
workflow.add_node("mcp_tools", mcp_tool_node)
workflow.set_entry_point("mcp_tools")

app = workflow.compile()
```

### Transport Mechanisms Comparison

| Transport | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **stdio** | Local development, desktop apps | Simple, no network config | Single client only |
| **Streamable HTTP** | Production, remote servers | Multi-client, scalable | Requires HTTP server |
| **SSE** | Real-time streaming | Efficient for long responses | More complex setup |

---

## 3. Tool Discovery and Dynamic Registration {#3-tool-discovery}

### Automatic Tool Discovery

The `MultiServerMCPClient` automatically discovers all tools from connected servers:

```python
from langchain_mcp_adapters import MultiServerMCPClient

async def discover_tools():
    """Automatically discover all tools from MCP servers."""

    client = MultiServerMCPClient({
        "server1": {"transport": "stdio", "command": "python", "args": ["server1.py"]},
        "server2": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"}
    })

    await client.connect()

    # Automatic discovery - fetches tools from all servers
    tools = await client.get_tools()

    print(f"Discovered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    return tools
```

### Dynamic Tool Registration Pattern

```python
from mcp.server.fastmcp import FastMCP
import importlib
import inspect

class DynamicMCPServer:
    """MCP Server with dynamic tool registration."""

    def __init__(self, name: str):
        self.mcp = FastMCP(name)
        self.registered_tools = {}

    def register_tool(self, func, name: str = None, description: str = None):
        """Dynamically register a tool."""
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "No description"

        # Register with FastMCP
        decorated_func = self.mcp.tool()(func)

        self.registered_tools[tool_name] = {
            "function": decorated_func,
            "description": tool_desc
        }

        return decorated_func

    def register_module_tools(self, module_path: str):
        """Register all functions from a module as tools."""
        module = importlib.import_module(module_path)

        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                self.register_tool(obj)

    def run(self, transport="stdio", **kwargs):
        """Run the MCP server."""
        self.mcp.run(transport=transport, **kwargs)

# Usage
server = DynamicMCPServer("DynamicServer")

# Register individual tool
def custom_tool(x: int) -> int:
    """Custom tool description"""
    return x * 2

server.register_tool(custom_tool)

# Register all tools from a module
# server.register_module_tools("my_tools_module")

server.run()
```

### Runtime Tool Loading

```python
from langchain_mcp_adapters import MultiServerMCPClient
from typing import List, Dict

class DynamicToolManager:
    """Manage dynamic tool loading and reloading."""

    def __init__(self):
        self.client = None
        self.tools = []
        self.server_configs = {}

    async def add_server(self, name: str, config: Dict):
        """Add a new server configuration."""
        self.server_configs[name] = config
        await self.reload_client()

    async def remove_server(self, name: str):
        """Remove a server configuration."""
        if name in self.server_configs:
            del self.server_configs[name]
            await self.reload_client()

    async def reload_client(self):
        """Reload client with updated server configurations."""
        if self.client:
            await self.client.disconnect()

        self.client = MultiServerMCPClient(self.server_configs)
        await self.client.connect()
        self.tools = await self.client.get_tools()

    async def get_tools(self) -> List:
        """Get current list of tools."""
        if not self.tools:
            self.tools = await self.client.get_tools()
        return self.tools

    async def refresh_tools(self):
        """Refresh tool list from servers."""
        self.tools = await self.client.get_tools()
        return self.tools

# Usage
manager = DynamicToolManager()

# Add servers dynamically
await manager.add_server("math", {
    "transport": "stdio",
    "command": "python",
    "args": ["math_server.py"]
})

# Get tools
tools = await manager.get_tools()

# Add another server at runtime
await manager.add_server("weather", {
    "transport": "streamable_http",
    "url": "http://localhost:8000/mcp"
})

# Refresh to get new tools
tools = await manager.refresh_tools()
```

### Tool Metadata and Schema

```python
async def inspect_tool_schemas():
    """Inspect tool schemas and metadata."""

    client = MultiServerMCPClient(server_configs)
    await client.connect()

    tools = await client.get_tools()

    for tool in tools:
        print(f"\nTool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Schema: {tool.args_schema.schema() if hasattr(tool, 'args_schema') else 'N/A'}")

        # Access tool metadata
        if hasattr(tool, 'metadata'):
            print(f"Metadata: {tool.metadata}")
```

---

## 4. Error Handling Patterns {#4-error-handling}

### Connection Error Handling

```python
import asyncio
import logging
from langchain_mcp_adapters import MultiServerMCPClient

logger = logging.getLogger(__name__)

async def robust_client_connection(
    server_configs: dict,
    max_retries: int = 3,
    retry_delay: float = 2.0
):
    """Connect to MCP servers with exponential backoff retry."""

    for attempt in range(max_retries):
        try:
            client = MultiServerMCPClient(server_configs)

            # Connect with timeout
            await asyncio.wait_for(
                client.connect(),
                timeout=30.0
            )

            logger.info("Successfully connected to MCP servers")
            return client

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                raise Exception("Max retries exceeded for connection")

        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                raise

        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            raise
```

### Tool Execution Error Handling

```python
from typing import Optional, Any
from enum import Enum

class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MCPToolExecutor:
    """Execute MCP tools with comprehensive error handling."""

    def __init__(self, client):
        self.client = client
        self.error_log = []

    async def execute_tool_safe(
        self,
        tool_name: str,
        args: dict,
        fallback_value: Optional[Any] = None,
        retry_count: int = 2
    ) -> tuple[Optional[Any], Optional[Exception]]:
        """Execute tool with error handling and fallback."""

        tools = await self.client.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            error = ValueError(f"Tool '{tool_name}' not found")
            self._log_error(tool_name, error, ErrorSeverity.ERROR)
            return fallback_value, error

        for attempt in range(retry_count):
            try:
                result = await tool.ainvoke(args)
                return result, None

            except TimeoutError as e:
                self._log_error(tool_name, e, ErrorSeverity.WARNING)
                if attempt < retry_count - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return fallback_value, e

            except ValueError as e:
                # Don't retry on validation errors
                self._log_error(tool_name, e, ErrorSeverity.ERROR)
                return fallback_value, e

            except Exception as e:
                self._log_error(tool_name, e, ErrorSeverity.CRITICAL)
                if attempt < retry_count - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return fallback_value, e

        return fallback_value, Exception("Max retries exceeded")

    def _log_error(self, tool_name: str, error: Exception, severity: ErrorSeverity):
        """Log error with metadata."""
        import time

        error_entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "error": str(error),
            "type": type(error).__name__,
            "severity": severity.value
        }

        self.error_log.append(error_entry)
        logger.error(f"[{severity.value.upper()}] Tool '{tool_name}': {error}")

# Usage
executor = MCPToolExecutor(client)
result, error = await executor.execute_tool_safe(
    "add",
    {"a": 5, "b": 3},
    fallback_value=0,
    retry_count=3
)
```

### Server Health Monitoring

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

@dataclass
class ServerHealth:
    name: str
    status: str  # "healthy", "degraded", "down"
    last_check: datetime
    response_time_ms: float
    error_count: int

class MCPHealthMonitor:
    """Monitor health of MCP servers."""

    def __init__(self, client: MultiServerMCPClient):
        self.client = client
        self.health_status: Dict[str, ServerHealth] = {}

    async def check_server_health(self, server_name: str) -> ServerHealth:
        """Check health of a specific server."""
        start_time = datetime.now()

        try:
            # Try to get tools from specific server
            async with self.client.session(server_name) as session:
                tools = await asyncio.wait_for(
                    load_mcp_tools(session),
                    timeout=5.0
                )

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            status = ServerHealth(
                name=server_name,
                status="healthy" if response_time < 1000 else "degraded",
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=0
            )

        except asyncio.TimeoutError:
            status = ServerHealth(
                name=server_name,
                status="degraded",
                last_check=datetime.now(),
                response_time_ms=5000.0,
                error_count=1
            )

        except Exception as e:
            logger.error(f"Health check failed for {server_name}: {e}")
            status = ServerHealth(
                name=server_name,
                status="down",
                last_check=datetime.now(),
                response_time_ms=-1.0,
                error_count=1
            )

        self.health_status[server_name] = status
        return status

    async def check_all_servers(self) -> Dict[str, ServerHealth]:
        """Check health of all configured servers."""
        # Get server names from client config
        server_names = list(self.client._server_configs.keys())

        results = await asyncio.gather(
            *[self.check_server_health(name) for name in server_names],
            return_exceptions=True
        )

        return self.health_status

    def get_healthy_servers(self) -> List[str]:
        """Get list of healthy server names."""
        return [
            name for name, health in self.health_status.items()
            if health.status == "healthy"
        ]

# Usage
monitor = MCPHealthMonitor(client)
health_status = await monitor.check_all_servers()
healthy_servers = monitor.get_healthy_servers()
```

### Protocol Error Handling

```python
from enum import Enum

class MCPErrorType(Enum):
    PROTOCOL_VIOLATION = "protocol_violation"
    SERIALIZATION_ERROR = "serialization_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    INTERNAL_ERROR = "internal_error"

class MCPErrorHandler:
    """Handle various MCP protocol errors."""

    @staticmethod
    def handle_error(error: Exception) -> tuple[MCPErrorType, str, bool]:
        """
        Analyze error and return (error_type, message, is_recoverable).
        """
        error_str = str(error).lower()

        # Timeout errors
        if isinstance(error, asyncio.TimeoutError) or "timeout" in error_str:
            return (
                MCPErrorType.TIMEOUT,
                "Request timed out. Server may be overloaded.",
                True  # Recoverable with retry
            )

        # Authentication errors
        if "401" in error_str or "unauthorized" in error_str:
            return (
                MCPErrorType.AUTHENTICATION,
                "Authentication failed. Check credentials.",
                False  # Not recoverable without fixing auth
            )

        # Authorization errors
        if "403" in error_str or "forbidden" in error_str:
            return (
                MCPErrorType.AUTHORIZATION,
                "Insufficient permissions for this operation.",
                False
            )

        # Not found errors
        if "404" in error_str or "not found" in error_str:
            return (
                MCPErrorType.NOT_FOUND,
                "Requested resource or tool not found.",
                False
            )

        # Serialization errors
        if any(keyword in error_str for keyword in ["json", "decode", "encode", "utf-8"]):
            return (
                MCPErrorType.SERIALIZATION_ERROR,
                "Data serialization failed. Check data format.",
                False
            )

        # Protocol violations
        if "protocol" in error_str or "schema" in error_str:
            return (
                MCPErrorType.PROTOCOL_VIOLATION,
                "MCP protocol violation detected.",
                False
            )

        # Default to internal error
        return (
            MCPErrorType.INTERNAL_ERROR,
            f"Internal error: {str(error)}",
            True  # Might be recoverable
        )

    @staticmethod
    async def execute_with_handling(
        coroutine,
        fallback_value=None,
        max_retries=2
    ):
        """Execute coroutine with automatic error handling and retry."""

        for attempt in range(max_retries):
            try:
                return await coroutine

            except Exception as e:
                error_type, message, is_recoverable = MCPErrorHandler.handle_error(e)

                logger.error(f"[{error_type.value}] {message}")

                if not is_recoverable or attempt >= max_retries - 1:
                    logger.error(f"Failed after {attempt + 1} attempts")
                    return fallback_value

                # Exponential backoff for recoverable errors
                await asyncio.sleep(2 ** attempt)
                logger.info(f"Retrying (attempt {attempt + 2}/{max_retries})...")

# Usage
result = await MCPErrorHandler.execute_with_handling(
    client.get_tools(),
    fallback_value=[],
    max_retries=3
)
```

---

## 5. Running Multiple MCP Servers with Docker {#5-docker-setup}

### Docker Compose Configuration

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # MCP Gateway (coordinates multiple servers)
  mcp-gateway:
    image: mcp-gateway:latest
    container_name: mcp-gateway
    ports:
      - "8080:8080"
    environment:
      - GATEWAY_PORT=8080
      - LOG_LEVEL=INFO
    networks:
      - mcp-network
    depends_on:
      - math-server
      - weather-server
      - database-server

  # Math Tools Server
  math-server:
    build:
      context: ./servers/math
      dockerfile: Dockerfile
    container_name: mcp-math-server
    ports:
      - "8001:8000"
    environment:
      - SERVER_NAME=math-tools
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    networks:
      - mcp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Weather Service Server
  weather-server:
    build:
      context: ./servers/weather
      dockerfile: Dockerfile
    container_name: mcp-weather-server
    ports:
      - "8002:8000"
    environment:
      - SERVER_NAME=weather-service
      - TRANSPORT=streamable-http
      - API_KEY=${WEATHER_API_KEY}
      - HOST=0.0.0.0
      - PORT=8000
    env_file:
      - .env.weather
    networks:
      - mcp-network
    restart: unless-stopped
    secrets:
      - weather_api_key

  # Database Tools Server
  database-server:
    build:
      context: ./servers/database
      dockerfile: Dockerfile
    container_name: mcp-database-server
    ports:
      - "8003:8000"
    environment:
      - SERVER_NAME=database-tools
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
    networks:
      - mcp-network
    depends_on:
      postgres:
        condition: service_healthy
    secrets:
      - db_password

  # PostgreSQL Database (for database server)
  postgres:
    image: postgres:15
    container_name: mcp-postgres
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - mcp-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    secrets:
      - db_password

networks:
  mcp-network:
    driver: bridge

volumes:
  postgres-data:

secrets:
  weather_api_key:
    file: ./secrets/weather_api_key.txt
  db_password:
    file: ./secrets/db_password.txt
```

### Individual Server Dockerfile

#### servers/math/Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

USER mcpuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run server
CMD ["python", "server.py"]
```

#### servers/math/requirements.txt

```txt
mcp>=1.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
```

#### servers/math/server.py

```python
from mcp.server.fastmcp import FastMCP
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(os.getenv("SERVER_NAME", "math-tools"))

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    transport = os.getenv("TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting {mcp.name} on {host}:{port} using {transport}")

    mcp.run(
        transport=transport,
        host=host,
        port=port
    )
```

### Environment Configuration

#### .env

```bash
# Postgres Configuration
POSTGRES_DB=mcpdb
POSTGRES_USER=mcpuser

# Weather API
WEATHER_API_KEY=your_api_key_here

# Gateway Configuration
GATEWAY_PORT=8080
LOG_LEVEL=INFO
```

#### secrets/db_password.txt

```
your_secure_database_password_here
```

#### secrets/weather_api_key.txt

```
your_weather_api_key_here
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d math-server weather-server

# Start with gateway profile
docker-compose --profile gateway up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f math-server

# Check service status
docker-compose ps

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build

# Scale a service
docker-compose up -d --scale math-server=3
```

### Client Configuration for Docker Services

```python
from langchain_mcp_adapters import MultiServerMCPClient

# Configuration for Docker Compose services
docker_servers = {
    "math": {
        "transport": "streamable_http",
        "url": "http://localhost:8001/mcp"
    },
    "weather": {
        "transport": "streamable_http",
        "url": "http://localhost:8002/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('WEATHER_API_KEY')}"
        }
    },
    "database": {
        "transport": "streamable_http",
        "url": "http://localhost:8003/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('DB_TOKEN')}"
        }
    }
}

client = MultiServerMCPClient(docker_servers)
```

### Docker MCP Toolkit (Official Solution)

Docker Desktop provides an official MCP Toolkit for managing multiple servers:

1. **Install Docker Desktop** (includes MCP Toolkit)
2. **Enable MCP Toolkit** in Docker Desktop settings
3. **Configure Gateway** in `docker-compose.yml`:

```yaml
services:
  mcp-gateway:
    image: docker/mcp-gateway:latest
    ports:
      - "8080:8080"
    volumes:
      - ./mcp-config.json:/config/mcp-config.json
    environment:
      - MCP_CONFIG=/config/mcp-config.json
```

4. **Gateway Configuration** (`mcp-config.json`):

```json
{
  "servers": {
    "math-tools": {
      "url": "http://math-server:8000/mcp",
      "enabled": true
    },
    "weather-service": {
      "url": "http://weather-server:8000/mcp",
      "enabled": true,
      "auth": {
        "type": "bearer",
        "token_env": "WEATHER_API_KEY"
      }
    }
  },
  "gateway": {
    "port": 8080,
    "timeout": 30,
    "max_connections": 100
  }
}
```

---

## 6. Authentication and Credentials {#6-authentication}

### OAuth 2.1 with Bearer Tokens

MCP follows OAuth 2.1 specification for authentication.

#### Authorization Specification

```python
# MCP requires Bearer token in Authorization header
# Format: Authorization: Bearer <access-token>
```

#### Server Configuration with Auth

```python
from mcp.server.fastmcp import FastMCP
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

mcp = FastMCP("SecureServer")
security = HTTPBearer()

# Token validation
VALID_TOKENS = {
    "secret_token_123": {"user": "client1", "scopes": ["read", "write"]},
    "secret_token_456": {"user": "client2", "scopes": ["read"]}
}

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token."""
    token = credentials.credentials

    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return VALID_TOKENS[token]

# Apply authentication to tools
@mcp.tool()
def secure_operation(data: str, user_info = Depends(verify_token)) -> str:
    """Secure operation requiring authentication."""
    return f"Processed by {user_info['user']}: {data}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

#### Client with Bearer Token

```python
from langchain_mcp_adapters import MultiServerMCPClient
import os

# Configuration with authentication
authenticated_servers = {
    "secure_server": {
        "transport": "streamable_http",
        "url": "https://api.example.com/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('MCP_ACCESS_TOKEN')}"
        }
    }
}

client = MultiServerMCPClient(authenticated_servers)
```

### API Key Authentication

```python
from mcp.server.fastmcp import FastMCP
from fastapi import Header, HTTPException

mcp = FastMCP("APIKeyServer")

VALID_API_KEYS = {
    "api_key_123": {"client": "app1", "rate_limit": 1000},
    "api_key_456": {"client": "app2", "rate_limit": 500}
}

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header."""
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return VALID_API_KEYS[x_api_key]

@mcp.tool()
def api_protected_tool(param: str, client_info = Depends(verify_api_key)) -> str:
    """Tool protected by API key."""
    return f"Client {client_info['client']}: {param}"
```

#### Client with API Key

```python
servers_with_api_key = {
    "api_server": {
        "transport": "streamable_http",
        "url": "https://api.example.com/mcp",
        "headers": {
            "X-API-Key": os.getenv('API_KEY')
        }
    }
}

client = MultiServerMCPClient(servers_with_api_key)
```

### OAuth 2.1 Authorization Code Flow

#### Server Implementation

```python
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import secrets

app = FastAPI()
mcp = FastMCP("OAuthServer")

# OAuth configuration
OAUTH_CONFIG = {
    "authorization_endpoint": "/authorize",
    "token_endpoint": "/token",
    "client_id": "mcp_client",
    "client_secret": "secret_123",
    "redirect_uri": "http://localhost:3000/callback"
}

# Temporary storage (use database in production)
authorization_codes = {}
access_tokens = {}

@app.get("/authorize")
async def authorize(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str = "S256"
):
    """OAuth authorization endpoint."""

    if client_id != OAUTH_CONFIG["client_id"]:
        return {"error": "invalid_client"}

    # Generate authorization code
    code = secrets.token_urlsafe(32)
    authorization_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method
    }

    # Redirect with code
    return RedirectResponse(f"{redirect_uri}?code={code}&state={state}")

@app.post("/token")
async def token(
    grant_type: str,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str
):
    """OAuth token endpoint."""

    if grant_type != "authorization_code":
        return {"error": "unsupported_grant_type"}

    if code not in authorization_codes:
        return {"error": "invalid_grant"}

    auth_data = authorization_codes[code]

    # Verify PKCE
    import hashlib
    import base64

    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip('=')

    if challenge != auth_data["code_challenge"]:
        return {"error": "invalid_grant"}

    # Generate access token
    access_token = secrets.token_urlsafe(32)
    access_tokens[access_token] = {
        "client_id": client_id,
        "scope": "read write"
    }

    # Clean up authorization code
    del authorization_codes[code]

    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "read write"
    }

# Mount MCP server
app.mount("/mcp", mcp)
```

### Environment Variable Management

#### Secure Credential Storage

```python
import os
from pathlib import Path
from dotenv import load_dotenv

class SecureCredentials:
    """Manage MCP server credentials securely."""

    def __init__(self, env_file: str = ".env.mcp"):
        self.env_file = Path(env_file)
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from environment file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)

    def get_server_config(self, server_name: str) -> dict:
        """Get server configuration with credentials."""

        base_config = {
            "transport": os.getenv(f"{server_name.upper()}_TRANSPORT", "streamable_http"),
            "url": os.getenv(f"{server_name.upper()}_URL")
        }

        # Add authentication if available
        token = os.getenv(f"{server_name.upper()}_TOKEN")
        api_key = os.getenv(f"{server_name.upper()}_API_KEY")

        if token:
            base_config["headers"] = {
                "Authorization": f"Bearer {token}"
            }
        elif api_key:
            base_config["headers"] = {
                "X-API-Key": api_key
            }

        return base_config

    def get_all_servers(self) -> dict:
        """Get all configured servers."""
        servers = {}

        # Parse environment variables for server configurations
        server_names = set()
        for key in os.environ:
            if "_URL" in key:
                server_name = key.replace("_URL", "").lower()
                server_names.add(server_name)

        for name in server_names:
            servers[name] = self.get_server_config(name)

        return servers

# Usage
creds = SecureCredentials()
server_configs = creds.get_all_servers()
client = MultiServerMCPClient(server_configs)
```

#### .env.mcp Example

```bash
# Math Server
MATH_TRANSPORT=streamable_http
MATH_URL=http://localhost:8001/mcp

# Weather Server (with Bearer token)
WEATHER_TRANSPORT=streamable_http
WEATHER_URL=https://api.weather.com/mcp
WEATHER_TOKEN=eyJhbGciOiJIUzI1NiIs...

# Database Server (with API key)
DATABASE_TRANSPORT=streamable_http
DATABASE_URL=https://db.example.com/mcp
DATABASE_API_KEY=sk_live_123456789

# Production Server (with OAuth)
PRODUCTION_TRANSPORT=streamable_http
PRODUCTION_URL=https://prod.example.com/mcp
PRODUCTION_TOKEN=oauth_token_here
```

### Custom Authentication Middleware

```python
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import jwt

class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Custom authentication middleware for MCP server."""

    def __init__(self, app, secret_key: str):
        super().__init__(app)
        self.secret_key = secret_key

    async def dispatch(self, request, call_next):
        # Skip auth for health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid authorization header"}
            )

        token = auth_header.replace("Bearer ", "")

        try:
            # Verify JWT
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Add user info to request state
            request.state.user = payload
            request.state.authenticated = True

            # Continue processing
            response = await call_next(request)
            return response

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"error": "Token expired"}
            )

        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token"}
            )

# Apply middleware
app = FastAPI()
app.add_middleware(MCPAuthMiddleware, secret_key="your-secret-key")
```

---

## 7. Streaming Responses from MCP Tools {#7-streaming-responses}

### Server-Sent Events (SSE) Implementation

#### SSE Server with FastAPI

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import StreamingResponse
import asyncio

mcp = FastMCP("StreamingServer")

def create_sse_server():
    """Create Starlette app with SSE support."""
    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        """Handle SSE connection."""
        async with transport.connect_sse(
            request.scope,
            request.receive,
            request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options()
            )

    routes = [
        Route("/sse/", endpoint=handle_sse),
        Mount("/messages/", app=transport.handle_post_message),
    ]

    return Starlette(routes=routes)

@mcp.tool()
async def stream_data(count: int) -> str:
    """Tool that processes data in streaming fashion."""
    results = []
    for i in range(count):
        await asyncio.sleep(0.1)  # Simulate processing
        results.append(f"Item {i + 1}")
    return ", ".join(results)

@mcp.resource("stream://data/{id}")
async def stream_resource(id: str) -> str:
    """Stream resource data."""
    # Simulate streaming large resource
    data_chunks = []
    for i in range(10):
        await asyncio.sleep(0.05)
        data_chunks.append(f"chunk_{i}")
    return "".join(data_chunks)

if __name__ == "__main__":
    import uvicorn
    app = create_sse_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### SSE Client Configuration

```python
from langchain_mcp_adapters import MultiServerMCPClient

# Configure SSE transport
sse_servers = {
    "streaming_server": {
        "transport": "sse",
        "url": "http://localhost:8000/sse"
    }
}

client = MultiServerMCPClient(sse_servers)
```

### Streamable HTTP Transport

#### Server Implementation

```python
from mcp.server.fastmcp import FastMCP
from typing import AsyncIterator
import asyncio

mcp = FastMCP("StreamableHTTPServer")

@mcp.tool()
async def generate_report(topic: str) -> str:
    """Generate a report with streaming updates."""

    # Simulate streaming generation
    sections = ["Introduction", "Analysis", "Conclusion"]
    report_parts = []

    for section in sections:
        await asyncio.sleep(0.5)  # Simulate processing
        part = f"\n## {section}\n\nContent for {section} about {topic}.\n"
        report_parts.append(part)

    return "".join(report_parts)

@mcp.tool()
async def process_large_dataset(dataset_id: str) -> dict:
    """Process large dataset with progress updates."""

    total_items = 100
    processed = 0
    results = []

    for i in range(total_items):
        await asyncio.sleep(0.01)  # Simulate processing
        results.append(f"result_{i}")
        processed += 1

        # In real implementation, server would send progress updates
        if processed % 10 == 0:
            print(f"Progress: {processed}/{total_items}")

    return {
        "dataset_id": dataset_id,
        "processed": processed,
        "results": results[:10]  # Return sample
    }

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000
    )
```

#### Client with Streaming

```python
from langchain_mcp_adapters import MultiServerMCPClient
import asyncio

async def stream_tool_execution():
    """Execute tool and handle streaming response."""

    client = MultiServerMCPClient({
        "streaming": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp"
        }
    })

    await client.connect()
    tools = await client.get_tools()

    # Get streaming tool
    generate_tool = next(t for t in tools if t.name == "generate_report")

    # Execute (server handles streaming internally)
    result = await generate_tool.ainvoke({"topic": "AI Technology"})

    print("Final result:", result)

    await client.disconnect()

asyncio.run(stream_tool_execution())
```

### LangGraph with Streaming MCP Tools

```python
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters import MultiServerMCPClient
from typing import TypedDict, Annotated
import operator

class StreamingState(TypedDict):
    messages: Annotated[list, operator.add]
    streaming_results: list
    current_progress: float

async def streaming_tool_node(state: StreamingState):
    """Node that executes streaming MCP tools."""

    # Setup MCP client
    client = MultiServerMCPClient({
        "streaming": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp"
        }
    })

    await client.connect()
    tools = await client.get_tools()

    # Execute streaming tool
    tool = next(t for t in tools if t.name == "generate_report")
    result = await tool.ainvoke({"topic": "Technology Trends"})

    state["streaming_results"].append(result)
    state["current_progress"] = 1.0

    await client.disconnect()

    return state

async def create_streaming_graph():
    """Create LangGraph with streaming MCP integration."""

    workflow = StateGraph(StreamingState)

    workflow.add_node("streaming_tools", streaming_tool_node)
    workflow.set_entry_point("streaming_tools")
    workflow.add_edge("streaming_tools", END)

    return workflow.compile()

# Run
app = await create_streaming_graph()
result = await app.ainvoke({
    "messages": [],
    "streaming_results": [],
    "current_progress": 0.0
})
```

### Advanced: Custom Streaming Protocol

```python
from mcp.server.fastmcp import FastMCP
from starlette.responses import StreamingResponse
from typing import AsyncIterator
import json
import asyncio

mcp = FastMCP("CustomStreamingServer")

async def stream_generator(data: list) -> AsyncIterator[str]:
    """Generate streaming responses."""
    for item in data:
        await asyncio.sleep(0.1)

        # Send as Server-Sent Event format
        event_data = {
            "type": "data",
            "content": item
        }
        yield f"data: {json.dumps(event_data)}\n\n"

    # Send completion event
    completion = {
        "type": "complete",
        "total": len(data)
    }
    yield f"data: {json.dumps(completion)}\n\n"

@mcp.tool()
async def custom_stream(query: str) -> str:
    """Tool with custom streaming logic."""

    # Generate data
    data_items = [f"Result {i} for {query}" for i in range(10)]

    # In actual implementation, this would stream
    # For now, return all results
    return "\n".join(data_items)

# Custom endpoint for true streaming
from fastapi import FastAPI

app = FastAPI()

@app.get("/stream/{query}")
async def stream_endpoint(query: str):
    """Custom streaming endpoint."""
    data = [f"Result {i} for {query}" for i in range(10)]
    return StreamingResponse(
        stream_generator(data),
        media_type="text/event-stream"
    )

app.mount("/mcp", mcp)
```

#### Client for Custom Streaming

```python
import httpx
import json

async def consume_stream(url: str, query: str):
    """Consume custom streaming endpoint."""

    async with httpx.AsyncClient() as client:
        async with client.stream('GET', f"{url}/stream/{query}") as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])

                    if data['type'] == 'data':
                        print(f"Received: {data['content']}")
                    elif data['type'] == 'complete':
                        print(f"Stream complete. Total items: {data['total']}")
                        break

# Usage
await consume_stream("http://localhost:8000", "AI trends")
```

### Streaming with Progress Callbacks

```python
from typing import Callable, Optional
import asyncio

class StreamingMCPExecutor:
    """Execute MCP tools with streaming progress callbacks."""

    def __init__(self, client: MultiServerMCPClient):
        self.client = client

    async def execute_with_progress(
        self,
        tool_name: str,
        args: dict,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """Execute tool with progress updates."""

        tools = await self.client.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Simulate progress (in real implementation, server would provide this)
        if progress_callback:
            progress_callback(0.0, "Starting execution")

        # Execute tool
        result = await tool.ainvoke(args)

        if progress_callback:
            progress_callback(1.0, "Execution complete")

        return result

# Usage
async def progress_handler(progress: float, message: str):
    """Handle progress updates."""
    print(f"[{progress*100:.0f}%] {message}")

executor = StreamingMCPExecutor(client)
result = await executor.execute_with_progress(
    "generate_report",
    {"topic": "AI"},
    progress_callback=progress_handler
)
```

---

## Summary and Best Practices

### Quick Start Checklist

1. **Install packages**: `pip install langchain-mcp-adapters langgraph mcp`
2. **Create MCP server** with FastMCP and `@mcp.tool()` decorators
3. **Configure client** with `MultiServerMCPClient` and server configs
4. **Load tools** using `await client.get_tools()`
5. **Create agent** with `create_react_agent(model, tools)`
6. **Handle errors** with try-except and retry logic
7. **Secure with auth** using Bearer tokens or API keys
8. **Use Docker** for production deployments

### Best Practices

**Server Development:**
- Use FastMCP for rapid server development
- Implement proper error handling in tools
- Add logging for debugging
- Use type hints for automatic schema generation
- Choose appropriate transport (stdio for local, HTTP for production)

**Client Integration:**
- Use `MultiServerMCPClient` for managing multiple servers
- Implement connection retry logic with exponential backoff
- Monitor server health regularly
- Handle timeouts gracefully
- Use environment variables for credentials

**Security:**
- Never hardcode credentials
- Use OAuth 2.1 for user-delegated access
- Implement PKCE for all OAuth flows
- Use HTTPS for all remote connections
- Store secrets in Docker secrets or environment files
- Validate tokens on every request

**Production Deployment:**
- Use Docker Compose for multi-server setups
- Implement health checks for all services
- Use non-root users in containers
- Set up proper logging and monitoring
- Configure resource limits
- Use connection pooling for better performance

**Performance:**
- Use async/await for I/O-bound operations
- Implement connection pooling
- Cache tool schemas when possible
- Monitor context size in LangGraph
- Use streaming for large responses
- Set appropriate timeouts

### Common Pitfalls to Avoid

1. **Don't** use stdio transport for multi-client scenarios
2. **Don't** forget to disconnect clients after use
3. **Don't** hardcode sensitive credentials
4. **Don't** skip error handling
5. **Don't** ignore server health monitoring
6. **Don't** use verbose logging in production
7. **Don't** forget PKCE in OAuth flows
8. **Don't** put access tokens in URL query strings

### Resources

- **Official Docs**: https://docs.langchain.com/oss/python/langchain/mcp
- **GitHub**: https://github.com/langchain-ai/langchain-mcp-adapters
- **MCP Specification**: https://modelcontextprotocol.io/specification/
- **Docker MCP Toolkit**: https://www.docker.com/blog/mcp-toolkit-mcp-servers-that-just-work/

---

**Document Version**: 1.0
**Last Updated**: 2025-03
**Compatibility**: MCP Protocol 2025-03-26, LangGraph 0.2+, Python 3.9+
