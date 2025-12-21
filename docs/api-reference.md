# API Reference

Complete API documentation for Acton Agent.

## Table of Contents

- [Agent](#agent)
- [LLM Clients](#llm-clients)
- [Tools](#tools)
- [Models](#models)
- [Memory](#memory)
- [Exceptions](#exceptions)
- [Utilities](#utilities)

## Agent

### Agent

The main agent class that orchestrates LLM interactions and tool execution.

```python
class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        retry_config: Optional[RetryConfig] = None,
        stream: bool = False,
        final_answer_format_instructions: Optional[str] = None,
        timezone: str = "UTC",
        memory: Optional[AgentMemory] = None,
    )
```

**Parameters:**
- `llm_client` (LLMClient): LLM client for model calls
- `system_prompt` (Optional[str]): Custom system instructions. Default: None
- `max_iterations` (int): Maximum reasoning iterations. Default: 10
- `retry_config` (Optional[RetryConfig]): Retry configuration. Default: Default RetryConfig
- `stream` (bool): Enable streaming responses. Default: False
- `final_answer_format_instructions` (Optional[str]): Custom format instructions. Default: Default format
- `timezone` (str): Timezone for timestamps. Default: "UTC"
- `memory` (Optional[AgentMemory]): Memory manager. Default: SimpleAgentMemory(8000)

**Example:**
```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import RetryConfig, SimpleAgentMemory

client = OpenAIClient(api_key="sk-...", model="gpt-4o")
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful coding assistant",
    max_iterations=15,
    retry_config=RetryConfig(max_attempts=5),
    memory=SimpleAgentMemory(max_history_tokens=10000),
    timezone="America/New_York"
)
```

#### Methods

##### run

Run the agent on user input and return the final answer.

```python
def run(self, user_input: str) -> str
```

**Parameters:**
- `user_input` (str): The user's query or request

**Returns:**
- `str`: The agent's final answer

**Raises:**
- `MaxIterationsError`: If no final answer within max_iterations
- `LLMCallError`: If LLM calls fail

**Example:**
```python
response = agent.run("What is the weather in Tokyo?")
print(response)
```

##### run_stream

Stream the agent's processing as structured events.

```python
def run_stream(self, user_input: str) -> Generator[StreamingEvent, None, None]
```

**Parameters:**
- `user_input` (str): The user's query or request

**Yields:**
- `StreamingEvent`: Various event types (AgentToken, AgentStepEvent, etc.)

**Example:**
```python
for event in agent.run_stream("Tell me a story"):
    if isinstance(event, AgentToken):
        print(event.content, end="", flush=True)
    elif isinstance(event, AgentFinalResponseEvent):
        print(f"\n\nDone: {event.response.final_answer}")
```

##### register_tool

Register a tool with the agent.

```python
def register_tool(self, tool: Tool) -> None
```

**Parameters:**
- `tool` (Tool): Tool instance to register

**Example:**
```python
from acton_agent.agent import FunctionTool

tool = FunctionTool(name="calc", description="Calculator", func=calculate, schema={...})
agent.register_tool(tool)
```

##### register_toolset

Register a ToolSet with the agent's ToolRegistry.

```python
def register_toolset(self, toolset: ToolSet) -> None
```

**Parameters:**
- `toolset` (ToolSet): Collection of related tools with a shared description

**Example:**
```python
from acton_agent.agent import ToolSet, FunctionTool

toolset = ToolSet(
    name="math_tools",
    description="Mathematical operations and calculations",
    tools=[
        FunctionTool(name="add", description="Add two numbers", func=add, schema={...}),
        FunctionTool(name="multiply", description="Multiply two numbers", func=multiply, schema={...})
    ]
)
agent.register_toolset(toolset)
```

##### unregister_tool

Remove a tool from the agent's registry.

```python
def unregister_tool(self, tool_name: str) -> None
```

**Parameters:**
- `tool_name` (str): Name of the tool to remove

**Raises:**
- `ToolNotFoundError`: If tool not found

**Example:**
```python
agent.unregister_tool("calculator")
```

##### list_tools

Get list of registered tool names.

```python
def list_tools(self) -> List[str]
```

**Returns:**
- `List[str]`: List of tool names

**Example:**
```python
tools = agent.list_tools()
print(f"Available tools: {', '.join(tools)}")
```

##### reset

Clear the agent's conversation history.

```python
def reset(self) -> None
```

**Example:**
```python
agent.reset()  # Start fresh conversation
```

##### add_message

Add a message to conversation history.

```python
def add_message(self, role: str, content: str) -> None
```

**Parameters:**
- `role` (str): Message role ("user", "assistant", "system")
- `content` (str): Message content

**Example:**
```python
agent.add_message("user", "Hello!")
agent.add_message("assistant", "Hi there!")
```

##### get_conversation_history

Get a copy of the conversation history.

```python
def get_conversation_history(self) -> List[Message]
```

**Returns:**
- `List[Message]`: Copy of conversation history

**Example:**
```python
history = agent.get_conversation_history()
for msg in history:
    print(f"{msg.role}: {msg.content}")
```

##### set_system_prompt

Update the agent's system prompt.

```python
def set_system_prompt(self, prompt: str) -> None
```

**Parameters:**
- `prompt` (str): New system prompt

**Example:**
```python
agent.set_system_prompt("You are now a Python expert")
```

##### set_timezone

Update the agent's timezone.

```python
def set_timezone(self, timezone: str) -> None
```

**Parameters:**
- `timezone` (str): IANA timezone name (e.g., "UTC", "America/New_York")

**Raises:**
- `ValueError`: If timezone is invalid

**Example:**
```python
agent.set_timezone("Europe/London")
```

## LLM Clients

### OpenAIClient

Client for OpenAI and OpenAI-compatible APIs.

```python
class OpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        default_headers: Optional[dict] = None,
    )
```

**Parameters:**
- `api_key` (Optional[str]): API key (or use OPENAI_API_KEY env var). Default: None
- `model` (str): Model identifier. Default: "gpt-4o"
- `base_url` (str): API base URL. Default: "https://api.openai.com/v1"
- `organization` (Optional[str]): Organization ID. Default: None
- `default_headers` (Optional[dict]): Default headers. Default: None

**Raises:**
- `ValueError`: If no API key provided

**Example:**
```python
from acton_agent.client import OpenAIClient

# Using environment variable
client = OpenAIClient(model="gpt-4o")

# Explicit API key
client = OpenAIClient(api_key="sk-...", model="gpt-3.5-turbo")

# Local/custom endpoint
client = OpenAIClient(
    api_key="not-needed",
    model="llama-3",
    base_url="http://localhost:8000/v1"
)
```

#### Methods

##### call

Make a chat completion request.

```python
def call(self, messages: List[Message], **kwargs) -> str
```

**Parameters:**
- `messages` (List[Message]): Conversation messages
- `**kwargs`: Additional parameters (temperature, max_tokens, etc.)

**Returns:**
- `str`: The assistant's response

**Example:**
```python
from acton_agent.agent import Message

messages = [Message(role="user", content="Hello!")]
response = client.call(messages, temperature=0.7)
```

##### call_stream

Stream chat completion tokens.

```python
def call_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]
```

**Parameters:**
- `messages` (List[Message]): Conversation messages
- `**kwargs`: Additional parameters

**Yields:**
- `str`: Token chunks

**Example:**
```python
for chunk in client.call_stream(messages):
    print(chunk, end="", flush=True)
```

### OpenRouterClient

Client for OpenRouter API (access multiple model providers).

```python
class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    )
```

**Parameters:**
- `api_key` (Optional[str]): OpenRouter API key. Default: None (uses OPENROUTER_API_KEY)
- `model` (str): Model identifier. Default: "openai/gpt-4o"
- `site_url` (Optional[str]): Your site URL for rankings. Default: None
- `app_name` (Optional[str]): Your app name for rankings. Default: None

**Example:**
```python
from acton_agent import OpenRouterClient

client = OpenRouterClient(
    api_key="sk-or-...",
    model="anthropic/claude-3-opus",
    site_url="https://myapp.com",
    app_name="My App"
)
```

## Tools

### Tool

Abstract base class for all tools.

```python
class Tool(ABC):
    def __init__(self, name: str, description: str)
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        pass
```

**Parameters:**
- `name` (str): Unique tool identifier
- `description` (str): Human-readable description

**Example:**
```python
from acton_agent.agent import Tool
from typing import Dict, Any

class MyTool(Tool):
    def __init__(self):
        super().__init__(name="my_tool", description="Does something useful")
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        # Implementation
        return "Result"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
```

### FunctionTool

Wraps a Python function as a tool.

```python
class FunctionTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        schema: Dict[str, Any]
    )
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `func` (Callable): Function to wrap
- `schema` (Dict[str, Any]): JSON Schema for parameters

**Raises:**
- `InvalidToolSchemaError`: If schema is invalid

**Example:**
```python
from acton_agent.agent import FunctionTool

def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

tool = FunctionTool(
    name="greeter",
    description="Greet someone with a custom greeting",
    func=greet,
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "greeting": {"type": "string", "description": "Greeting word", "default": "Hello"}
        },
        "required": ["name"]
    }
)
```

### RequestsTool

Tool for making HTTP API requests.

```python
class RequestsTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        url_template: str = "",
        headers: Optional[Dict[str, str]] = None,
        query_params_schema: Optional[Dict[str, Any]] = None,
        body_schema: Optional[Dict[str, Any]] = None,
        path_params: Optional[List[str]] = None,
        path_params_schema: Optional[Dict[str, Any]] = None,
        header_params_schema: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        auth: Optional[tuple] = None,
    )
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `method` (str): HTTP method. Default: "GET"
- `url_template` (str): URL template with {param} placeholders
- `headers` (Optional[Dict[str, str]]): Default headers
- `query_params_schema` (Optional[Dict[str, Any]]): Query parameter schema
- `body_schema` (Optional[Dict[str, Any]]): Request body schema
- `path_params` (Optional[List[str]]): Path parameters (auto-extracted if None)
- `path_params_schema` (Optional[Dict[str, Any]]): Path parameter schema
- `header_params_schema` (Optional[Dict[str, Any]]): Header parameter schema
- `timeout` (int): Request timeout in seconds. Default: 30
- `auth` (Optional[tuple]): (username, password) for basic auth

**Example:**
```python
from acton_agent.tools import RequestsTool

# Simple GET request
tool = RequestsTool(
    name="get_user",
    description="Fetch user by ID",
    method="GET",
    url_template="https://api.example.com/users/{user_id}",
    path_params_schema={
        "user_id": {"type": "integer", "description": "User ID"}
    }
)

# POST with body and query params
tool = RequestsTool(
    name="create_post",
    description="Create a new post",
    method="POST",
    url_template="https://api.example.com/posts",
    query_params_schema={
        "publish": {"type": "boolean", "description": "Publish immediately"}
    },
    body_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["title", "content"]
    },
    headers={"X-API-Key": "your-api-key"}
)
```

### create_api_tool

Helper function to quickly create a RequestsTool.

```python
def create_api_tool(
    name: str,
    description: str,
    endpoint: str,
    method: str = "GET",
    **kwargs
) -> RequestsTool
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `endpoint` (str): Full endpoint URL (may include {params})
- `method` (str): HTTP method. Default: "GET"
- `**kwargs`: Additional RequestsTool parameters

**Returns:**
- `RequestsTool`: Configured tool

**Example:**
```python
from acton_agent.tools import create_api_tool

tool = create_api_tool(
    name="get_weather",
    description="Get weather for a city",
    endpoint="https://api.weather.com/v1/{city}",
    method="GET"
)
```

### ToolRegistry

Registry for managing tools and toolsets.

```python
class ToolRegistry:
    def __init__(self)
    
    def register(self, tool: Tool) -> None
    def register_toolset(self, toolset: ToolSet) -> None
    def unregister(self, tool_name: str) -> None
    def unregister_toolset(self, toolset_name: str) -> None
    def get(self, tool_name: str) -> Optional[Tool]
    def list_tools(self) -> List[Tool]
    def list_tool_names(self) -> List[str]
    def list_toolsets(self) -> List[str]
    def has_tool(self, tool_name: str) -> bool
    def clear(self) -> None
```

**Methods:**

- `register(tool)`: Register a single tool
- `register_toolset(toolset)`: Register a ToolSet and all its tools
- `unregister(tool_name)`: Remove a tool by name
- `unregister_toolset(toolset_name)`: Remove a toolset and all its tools
- `list_toolsets()`: Get names of all registered toolsets

**Example:**
```python
from acton_agent.agent import ToolRegistry, ToolSet, FunctionTool

registry = ToolRegistry()

# Register individual tool
registry.register(my_tool)

# Register a toolset
toolset = ToolSet(
    name="database_tools",
    description="Database query and management tools",
    tools=[query_tool, insert_tool]
)
registry.register_toolset(toolset)

# List toolsets
toolsets = registry.list_toolsets()  # Returns: ["database_tools"]

# Unregister toolset
registry.unregister_toolset("database_tools")
```

## Models

### Message

Represents a conversation message.

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
```

**Example:**
```python
from acton_agent.agent import Message

msg = Message(role="user", content="Hello!")
```

### ToolCall

Represents a tool call request.

```python
class ToolCall(BaseModel):
    id: str
    tool_name: str
    parameters: Dict[str, Any] = {}
```

**Example:**
```python
from acton_agent.agent import ToolCall

call = ToolCall(
    id="call_123",
    tool_name="calculator",
    parameters={"a": 5, "b": 3, "operation": "add"}
)
```

### ToolResult

Result from tool execution.

```python
class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    result: str
    error: Optional[str] = None
    
    @property
    def success(self) -> bool
```

**Example:**
```python
from acton_agent.agent import ToolResult

result = ToolResult(
    tool_call_id="call_123",
    tool_name="calculator",
    result="8",
    error=None
)
print(result.success)  # True
```

### ToolSet

Represents a collection of related tools with a shared description.

```python
class ToolSet(BaseModel):
    name: str
    description: str
    tools: List[Tool] = []
```

**Attributes:**
- `name` (str): Unique name for the toolset
- `description` (str): General description of what this group of tools can do
- `tools` (List[Tool]): List of Tool instances in this toolset

**Example:**
```python
from acton_agent.agent import ToolSet, FunctionTool

toolset = ToolSet(
    name="weather_tools",
    description="Tools for fetching and analyzing weather data",
    tools=[
        FunctionTool(name="get_weather", description="Get current weather", func=get_weather, schema={...}),
        FunctionTool(name="get_forecast", description="Get weather forecast", func=get_forecast, schema={...})
    ]
)
```

### AgentPlan

Agent's planning step.

```python
class AgentPlan(BaseModel):
    plan: str = ""
```

### AgentStep

Intermediate step with tool calls.

```python
class AgentStep(BaseModel):
    tool_thought: Optional[str] = None
    tool_calls: List[ToolCall] = []
```

### AgentFinalResponse

Agent's final answer.

```python
class AgentFinalResponse(BaseModel):
    final_answer: str
```

### Streaming Events

Event types for streaming:

```python
class AgentStreamStart(BaseModel):
    step_id: str

class AgentToken(BaseModel):
    step_id: str
    content: str

class AgentStreamEnd(BaseModel):
    step_id: str

class AgentPlanEvent(BaseModel):
    step_id: str
    plan: AgentPlan

class AgentStepEvent(BaseModel):
    step_id: str
    step: AgentStep

class AgentToolResultsEvent(BaseModel):
    step_id: str
    results: List[ToolResult]

class AgentFinalResponseEvent(BaseModel):
    step_id: str
    response: AgentFinalResponse
```

## Memory

### AgentMemory

Abstract base class for memory management.

```python
class AgentMemory(ABC):
    @abstractmethod
    def manage_history(self, history: List[Message]) -> List[Message]:
        pass
```

### SimpleAgentMemory

Token-based memory management.

```python
class SimpleAgentMemory(AgentMemory):
    def __init__(self, max_history_tokens: int = 8000)
```

**Parameters:**
- `max_history_tokens` (int): Maximum tokens to keep. Default: 8000

**Example:**
```python
from acton_agent.agent import SimpleAgentMemory

memory = SimpleAgentMemory(max_history_tokens=10000)
agent = Agent(llm_client=client, memory=memory)
```

## Exceptions

### AgentError

Base exception for all agent errors.

```python
class AgentError(Exception):
    pass
```

### ToolNotFoundError

Raised when a tool is not found.

```python
class ToolNotFoundError(AgentError):
    def __init__(self, tool_name: str)
```

**Attributes:**
- `tool_name` (str): Name of the missing tool

### ToolExecutionError

Raised when tool execution fails.

```python
class ToolExecutionError(AgentError):
    def __init__(self, tool_name: str, original_error: Exception)
```

**Attributes:**
- `tool_name` (str): Name of the failed tool
- `original_error` (Exception): Original exception

### LLMCallError

Raised when LLM call fails.

```python
class LLMCallError(AgentError):
    def __init__(self, original_error: Exception, retry_count: int = 0)
```

**Attributes:**
- `original_error` (Exception): Original exception
- `retry_count` (int): Number of retries attempted

### MaxIterationsError

Raised when agent reaches max iterations.

```python
class MaxIterationsError(AgentError):
    def __init__(self, max_iterations: int)
```

**Attributes:**
- `max_iterations` (int): Maximum iterations reached

### ResponseParseError

Raised when response cannot be parsed.

```python
class ResponseParseError(AgentError):
    def __init__(self, response_text: str, original_error: Exception)
```

**Attributes:**
- `response_text` (str): Raw response text
- `original_error` (Exception): Parsing error

### InvalidToolSchemaError

Raised when tool schema is invalid.

```python
class InvalidToolSchemaError(AgentError):
    def __init__(self, tool_name: str, reason: str)
```

**Attributes:**
- `tool_name` (str): Tool with invalid schema
- `reason` (str): Why schema is invalid

## Utilities

### RetryConfig

Configuration for retry logic.

```python
class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
        retry_on_exceptions: tuple = (Exception,)
    )
```

**Parameters:**
- `max_attempts` (int): Maximum retry attempts. Default: 3
- `min_wait` (float): Minimum wait time in seconds. Default: 1.0
- `max_wait` (float): Maximum wait time in seconds. Default: 60.0
- `multiplier` (float): Multiplier for exponential backoff. Default: 2.0
- `retry_on_exceptions` (tuple): Exception types to retry on. Default: (Exception,)

**Example:**
```python
from acton_agent.agent import RetryConfig

config = RetryConfig(
    max_attempts=5,
    min_wait=2.0,
    max_wait=120.0,
    multiplier=3.0
)
agent = Agent(llm_client=client, retry_config=config)
```

### OpenAPI Tools

#### create_tools_from_openapi

Generate tools from OpenAPI specification.

```python
def create_tools_from_openapi(
    spec: Union[str, Dict[str, Any]],
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
    operation_ids: Optional[List[str]] = None,
    max_tools: Optional[int] = None,
) -> List[RequestsTool]
```

**Parameters:**
- `spec`: OpenAPI spec (dict, URL, or file path)
- `base_url` (Optional[str]): Override base URL
- `headers` (Optional[Dict[str, str]]): Additional headers
- `tags` (Optional[List[str]]): Filter by tags
- `operation_ids` (Optional[List[str]]): Filter by operation IDs
- `max_tools` (Optional[int]): Maximum tools to generate

**Returns:**
- `List[RequestsTool]`: Generated tools

**Example:**
```python
from acton_agent.tools import create_tools_from_openapi

tools = create_tools_from_openapi(
    spec="https://api.example.com/openapi.json",
    headers={"X-API-Key": "your-key"},
    tags=["Users", "Posts"],
    max_tools=20
)

for tool in tools:
    agent.register_tool(tool)
```

## See Also

- [Getting Started](getting-started.md) - Quick start guide
- [Core Concepts](core-concepts.md) - Understanding the framework
- [Examples](examples/) - Practical examples
