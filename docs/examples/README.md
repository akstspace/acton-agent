# Examples

This directory contains practical examples demonstrating various features and use cases of Acton Agent.

## Quick Navigation

### Basic Examples
- [Simple Agent](#simple-agent) - Get started with a basic agent
- [Function Tools](#function-tools) - Wrap Python functions as tools
- [API Integration](#api-integration) - Connect to REST APIs

### Intermediate Examples
- [Multi-Tool Agent](#multi-tool-agent) - Agent with multiple tools
- [Organizing Tools with ToolSets](#organizing-tools-with-toolsets) - Group related tools
- [Conversation Context](#conversation-context) - Multi-turn conversations
- [Custom Memory](#custom-memory) - Custom memory management
- [Streaming Responses](#streaming-responses) - Real-time output

### Advanced Examples
- [Custom Tool Class](#custom-tool-class) - Build custom tools
- [OpenAPI Integration](#openapi-integration) - Auto-generate tools from specs
- [Error Handling](#error-handling) - Robust error management
- [Production Patterns](#production-patterns) - Production-ready code

---

## Basic Examples

### Simple Agent

The simplest possible agent setup:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient

# Initialize client and agent
client = OpenAIClient(api_key="your-key", model="gpt-4o")
agent = Agent(llm_client=client)

# Run a simple query
response = agent.run("What is the capital of France?")
print(response)
# Output: "The capital of France is Paris."
```

### Function Tools

Turn any Python function into a tool:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool

# Define your function
def calculate(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else 0
    }
    return ops.get(operation, 0)

# Wrap it as a tool
calculator = FunctionTool(
    name="calculator",
    description="Perform arithmetic: add, subtract, multiply, divide",
    func=calculate,
    schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            }
        },
        "required": ["a", "b", "operation"]
    }
)

# Create agent and register tool
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)
agent.register_tool(calculator)

# Use it
result = agent.run("What is 156 multiplied by 23?")
print(result)
# Output: "The result of 156 multiplied by 23 is 3,588."
```

### API Integration

Connect to REST APIs easily:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import create_api_tool

# Create API tools
weather_tool = create_api_tool(
    name="get_weather",
    description="Get current weather for a city",
    endpoint="https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={city}",
    method="GET"
)

user_tool = create_api_tool(
    name="get_user",
    description="Fetch user information by ID",
    endpoint="https://jsonplaceholder.typicode.com/users/{user_id}",
    method="GET"
)

# Set up agent
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)
agent.register_tool(weather_tool)
agent.register_tool(user_tool)

# Use the tools
response = agent.run("Get information about user 5")
print(response)
```

---

## Intermediate Examples

### Multi-Tool Agent

Agent with multiple tools working together:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool
import datetime

# Define multiple tools
def get_current_time(timezone: str = "UTC") -> str:
    """Get current time."""
    return datetime.datetime.now().isoformat()

def search_web(query: str) -> str:
    """Simulate web search."""
    return f"Search results for '{query}': [Sample results...]"

def calculate(expression: str) -> float:
    """Evaluate mathematical expression."""
    try:
        return eval(expression)  # Note: Use safely in production!
    except:
        return 0

# Create tools
time_tool = FunctionTool(
    name="get_time",
    description="Get current time",
    func=get_current_time,
    schema={"type": "object", "properties": {}, "required": []}
)

search_tool = FunctionTool(
    name="search",
    description="Search the web",
    func=search_web,
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
)

calc_tool = FunctionTool(
    name="calculate",
    description="Evaluate math expression",
    func=calculate,
    schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    }
)

# Create agent with all tools
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)
agent.register_tool(time_tool)
agent.register_tool(search_tool)
agent.register_tool(calc_tool)

# Complex multi-step query
response = agent.run(
    "What time is it? Then calculate 25 * 4 and search for information about that number."
)
print(response)
```

### Organizing Tools with ToolSets

Group related tools together with ToolSets for better organization:

```python
from acton_agent import Agent, ToolSet
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool

# Define weather-related functions
def get_current_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast."""
    return f"{days}-day forecast for {city}: Sunny with highs around 75°F"

# Create a ToolSet for weather tools
weather_toolset = ToolSet(
    name="weather_tools",
    description="Tools for fetching weather data and forecasts",
    tools=[
        FunctionTool(
            name="current_weather",
            description="Get current weather for any city",
            func=get_current_weather,
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        ),
        FunctionTool(
            name="weather_forecast",
            description="Get multi-day weather forecast",
            func=get_forecast,
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of days", "default": 3}
                },
                "required": ["city"]
            }
        )
    ]
)

# Create agent and register the entire toolset
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)
agent.register_toolset(weather_toolset)

# Use the tools
response = agent.run("What's the weather in Seattle? Also give me a 5-day forecast.")
print(response)

# List registered toolsets
toolsets = agent.tool_registry.list_toolsets()
print(f"Available toolsets: {toolsets}")  # ["weather_tools"]
```

**Benefits of ToolSets:**
- Organize related tools by domain (weather, database, math, etc.)
- Provide shared context via toolset description
- Register/unregister multiple tools at once
- Improve LLM understanding of tool relationships

### Conversation Context

Multi-turn conversation with context:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient

client = OpenAIClient(model="gpt-4o")
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful tutor for Python programming."
)

# First question
response1 = agent.run("What are Python decorators?")
print("Q1:", response1)

# Follow-up - agent remembers context
response2 = agent.run("Can you show me an example?")
print("Q2:", response2)

# Another follow-up
response3 = agent.run("What are common use cases for them?")
print("Q3:", response3)

# View conversation history
history = agent.get_conversation_history()
print(f"\nConversation has {len(history)} messages")

# Reset to start fresh
agent.reset()
```

### Custom Memory

Implement custom memory management:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import AgentMemory, Message
from typing import List

class SlidingWindowMemory(AgentMemory):
    """Keep only the last N messages."""

    def __init__(self, window_size: int = 6):
        self.window_size = window_size

    def manage_history(self, history: List[Message]) -> List[Message]:
        if len(history) <= self.window_size:
            return history
        return history[-self.window_size:]

# Use custom memory
client = OpenAIClient(model="gpt-4o")
memory = SlidingWindowMemory(window_size=4)
agent = Agent(llm_client=client, memory=memory)

# Have a long conversation
for i in range(10):
    response = agent.run(f"Tell me fact {i+1} about Python")
    print(f"Fact {i+1}: {response[:50]}...")

# Only last 4 messages are kept
history = agent.get_conversation_history()
print(f"\nMemory contains {len(history)} messages")
```

### Streaming Responses

Real-time streaming output:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import AgentToken, AgentFinalResponseEvent
import sys

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client, stream=True)

print("Assistant: ", end="", flush=True)

for event in agent.run_stream("Write a short poem about programming"):
    if isinstance(event, AgentToken):
        # Print each token as it arrives
        print(event.content, end="", flush=True)
        sys.stdout.flush()

    elif isinstance(event, AgentFinalResponseEvent):
        print("\n\n[Complete]")

# Output streams word-by-word in real-time
```

---

## Advanced Examples

### Custom Tool Class

Build sophisticated custom tools:

```python
from acton_agent.agent import Tool
from typing import Dict, Any
import json

class FileSystemTool(Tool):
    """Tool for safe file system operations."""

    ALLOWED_DIR = "/tmp/agent_workspace"

    def __init__(self):
        super().__init__(
            name="file_ops",
            description="Read and write files in workspace"
        )
        # Create workspace if needed
        import os
        os.makedirs(self.ALLOWED_DIR, exist_ok=True)

    def execute(self, parameters: Dict[str, Any]) -> str:
        import os

        operation = parameters.get("operation")
        filename = parameters.get("filename", "")

        # Security: Prevent path traversal
        if ".." in filename or filename.startswith("/"):
            return "Error: Invalid filename"

        filepath = os.path.join(self.ALLOWED_DIR, filename)

        if operation == "read":
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                return f"File content:\n{content}"
            except FileNotFoundError:
                return "Error: File not found"
            except Exception as e:
                return f"Error: {str(e)}"

        elif operation == "write":
            content = parameters.get("content", "")
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {filename}"
            except Exception as e:
                return f"Error: {str(e)}"

        elif operation == "list":
            try:
                files = os.listdir(self.ALLOWED_DIR)
                return f"Files: {', '.join(files)}"
            except Exception as e:
                return f"Error: {str(e)}"

        return "Error: Unknown operation"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "File operation to perform"
                },
                "filename": {
                    "type": "string",
                    "description": "Name of file (for read/write)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                }
            },
            "required": ["operation"]
        }

# Use the custom tool
from acton_agent import Agent
from acton_agent.client import OpenAIClient

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)
agent.register_tool(FileSystemTool())

response = agent.run("Write 'Hello World' to a file called test.txt")
print(response)
```

### OpenAPI Integration

Auto-generate tools from OpenAPI specs:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import create_tools_from_openapi

# Generate tools from OpenAPI spec
tools = create_tools_from_openapi(
    spec="https://petstore3.swagger.io/api/v3/openapi.json",
    tags=["pet"],  # Only include 'pet' operations
    max_tools=10   # Limit number of tools
)

# Create agent and register all tools
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)

for tool in tools:
    agent.register_tool(tool)
    print(f"Registered: {tool.name}")

# Use auto-generated tools
response = agent.run("Find available pets")
print(response)
```

### Error Handling

Robust error handling patterns:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import (
    MaxIterationsError,
    LLMCallError,
    ToolExecutionError,
    RetryConfig
)

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    min_wait=1.0,
    max_wait=30.0,
    multiplier=2.0
)

client = OpenAIClient(model="gpt-4o")
agent = Agent(
    llm_client=client,
    max_iterations=10,
    retry_config=retry_config
)

def safe_run(query: str, default_response: str = None) -> str:
    """Run agent with comprehensive error handling."""
    try:
        return agent.run(query)

    except MaxIterationsError as e:
        print(f"Warning: Max iterations ({e.max_iterations}) reached")
        return default_response or "Could not complete request in time."

    except LLMCallError as e:
        print(f"Error: LLM call failed after {e.retry_count} retries")
        print(f"Reason: {e.original_error}")
        return default_response or "Service temporarily unavailable."

    except ToolExecutionError as e:
        print(f"Error: Tool '{e.tool_name}' failed")
        print(f"Reason: {e.original_error}")
        return default_response or "Tool execution failed."

    except Exception as e:
        print(f"Unexpected error: {e}")
        return default_response or "An error occurred."

# Usage
response = safe_run(
    "Complex query that might fail",
    default_response="Unable to process request at this time."
)
print(response)
```

### Production Patterns

Production-ready agent setup:

```python
from acton_agent import Agent, SimpleAgentMemory
from acton_agent.client import OpenAIClient
from acton_agent.agent import RetryConfig
import os
import logging

class ProductionAgent:
    """Production-ready agent wrapper."""

    def __init__(self):
        # Configuration from environment
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o")

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create agent with production settings
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create agent with production configuration."""

        # Production retry config
        retry_config = RetryConfig(
            max_attempts=5,
            min_wait=2.0,
            max_wait=120.0,
            multiplier=2.5
        )

        # Client with error handling
        try:
            client = OpenAIClient(
                api_key=self.api_key,
                model=self.model
            )
        except ValueError as e:
            self.logger.error(f"Failed to create client: {e}")
            raise

        # Agent with conservative settings
        agent = Agent(
            llm_client=client,
            max_iterations=15,
            retry_config=retry_config,
            memory=SimpleAgentMemory(max_history_tokens=10000),
            system_prompt=self._get_system_prompt()
        )

        # Register tools
        self._register_tools(agent)

        return agent

    def _get_system_prompt(self) -> str:
        """Load system prompt from file or environment."""
        prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file) as f:
                return f.read()
        return "You are a helpful assistant."

    def _register_tools(self, agent: Agent):
        """Register all tools."""
        # Register your tools here
        pass

    def run(self, query: str, user_id: str = None) -> dict:
        """
        Run agent with full production features.

        Returns:
            dict with response, metadata, and status
        """
        import time
        start_time = time.time()

        try:
            # Log request
            self.logger.info(f"Request from user={user_id}: {query[:100]}")

            # Run agent
            response = self.agent.run(query)

            # Log success
            duration = time.time() - start_time
            self.logger.info(f"Completed in {duration:.2f}s")

            return {
                "status": "success",
                "response": response,
                "duration_seconds": duration,
                "tokens_used": None,  # Could track this
            }

        except Exception as e:
            # Log error
            self.logger.error(f"Error: {e}", exc_info=True)

            return {
                "status": "error",
                "response": "An error occurred processing your request.",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }

    def health_check(self) -> dict:
        """Check agent health."""
        try:
            test_response = self.agent.run("Say 'OK'")
            return {
                "status": "healthy",
                "details": {
                    "tools": len(self.agent.list_tools()),
                    "model": self.model
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Usage
if __name__ == "__main__":
    agent = ProductionAgent()

    # Health check
    health = agent.health_check()
    print(f"Health: {health}")

    # Process request
    result = agent.run("What is the capital of France?", user_id="user123")
    print(f"Result: {result}")
```

---

## Running the Examples

All examples assume you have set up your API keys:

```bash
export OPENAI_API_KEY="your-key-here"
```

Run any example:

```bash
python example_name.py
```

## See Also

- [Getting Started](../getting-started.md) - Installation and setup
- [Core Concepts](../core-concepts.md) - Understanding the framework
- [API Reference](../api-reference.md) - Complete API docs
