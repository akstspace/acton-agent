# Acton Agent

> ⚠️ **Experimental Project**: This is a personal project currently in an experimental phase. The API may change without notice, and features may be incomplete or unstable. Use at your own discretion.

Acton Agent is a lightweight, flexible LLM Agent Framework with tool execution capabilities. It enables you to build AI agents that can interact with external APIs, execute custom Python functions, and maintain conversation context - all with minimal configuration.

## Installation

### Basic Installation

```bash
pip install acton-agent
```

### Installation with Optional Dependencies

For OpenAI integration:
```bash
pip install acton-agent[openai]
```

For streaming support:
```bash
pip install acton-agent[streaming]
```

For development (includes testing and linting tools):
```bash
pip install acton-agent[dev]
```

Install all optional dependencies:
```bash
pip install acton-agent[all]
```

### Requirements

- Python >= 3.8
- Core dependencies:
  - `pydantic >= 2.0.0`
  - `tenacity >= 8.0.0`
  - `loguru >= 0.7.0`
  - `requests >= 2.31.0`

## Usage Examples

### Example 1: Requests Tool Usage

The `RequestsTool` allows your agent to make HTTP API calls. Here's an example using the JSONPlaceholder API:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import RequestsTool

# Initialize the OpenAI client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4o"
)

# Create an agent
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant that can fetch data from APIs."
)

# Create a RequestsTool for fetching posts from JSONPlaceholder
posts_tool = RequestsTool(
    name="get_posts",
    description="Fetch posts from JSONPlaceholder API",
    method="GET",
    url_template="https://jsonplaceholder.typicode.com/posts",
    query_params_schema={
        "userId": {
            "type": "number",
            "description": "Filter posts by user ID",
            "required": False
        }
    }
)

# Register the tool with the agent
agent.register_tool(posts_tool)

# Run the agent with a query
result = agent.run("Get me the posts from user ID 1")
print(result)
```

You can also use the convenient `create_api_tool` helper:

```python
from acton_agent.tools import create_api_tool

# Create a tool for fetching a specific post
post_tool = create_api_tool(
    name="get_post",
    description="Fetch a specific post by ID",
    endpoint="https://jsonplaceholder.typicode.com/posts/{post_id}",
    method="GET"
)

# Note: Path parameters are automatically extracted from the URL template
agent.register_tool(post_tool)
result = agent.run("Get me post number 5")
```

### Example 2: Function Tool Agent

The `FunctionTool` allows you to wrap Python functions and expose them to your agent:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool

# Initialize the client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4o"
)

# Create an agent
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant with calculator capabilities."
)

# Define a Python function
def calculate(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Define the schema for the function
calculator_schema = {
    "type": "object",
    "properties": {
        "a": {
            "type": "number",
            "description": "First number"
        },
        "b": {
            "type": "number",
            "description": "Second number"
        },
        "operation": {
            "type": "string",
            "description": "Operation to perform",
            "enum": ["add", "subtract", "multiply", "divide"]
        }
    },
    "required": ["a", "b", "operation"]
}

# Create a FunctionTool
calculator_tool = FunctionTool(
    name="calculator",
    description="Perform basic arithmetic operations",
    func=calculate,
    schema=calculator_schema
)

# Register the tool with the agent
agent.register_tool(calculator_tool)

# Run the agent with queries
result = agent.run("What is 25 multiplied by 4?")
print(result)

result = agent.run("Calculate 100 divided by 5, then add 10 to the result")
print(result)
```

You can also create custom tools by subclassing the `Tool` class:

```python
from acton_agent.agent import Tool

class WeatherTool(Tool):
    """Custom tool for getting weather information."""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a city"
        )
    
    def execute(self, parameters: dict) -> str:
        """Execute the tool with the given parameters."""
        city = parameters.get("city", "Unknown")
        # In a real implementation, you would call a weather API here
        return f"The weather in {city} is sunny and 72°F"
    
    def get_schema(self) -> dict:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city"
                }
            },
            "required": ["city"]
        }

# Use the custom tool
weather_tool = WeatherTool()
agent.register_tool(weather_tool)
result = agent.run("What's the weather in San Francisco?")
```

## API Reference

### Core Classes

#### `Agent`

The main agent class that orchestrates LLM interactions and tool execution.

**Constructor Parameters:**
- `llm_client` (LLMClient): The LLM client for generating responses
- `system_prompt` (Optional[str]): Custom instructions for the agent
- `max_iterations` (int): Maximum reasoning iterations (default: 10)
- `retry_config` (Optional[RetryConfig]): Retry configuration for LLM/tool calls
- `stream` (bool): Enable streaming responses (default: False)

**Key Methods:**
- `register_tool(tool: Tool)`: Register a tool with the agent
- `unregister_tool(tool_name: str)`: Remove a tool by name
- `run(query: str, **kwargs)`: Execute the agent with a query
- `run_stream(query: str, **kwargs)`: Execute with streaming (yields events)
- `reset()`: Clear conversation history

#### `Tool` (Abstract Base Class)

Base class for all tools.

**Constructor Parameters:**
- `name` (str): Unique identifier for the tool
- `description` (str): Human-readable description

**Abstract Methods:**
- `execute(parameters: Dict[str, Any]) -> str`: Execute the tool
- `get_schema() -> Dict[str, Any]`: Return JSON schema for parameters

#### `FunctionTool`

Wrapper for Python functions as tools.

**Constructor Parameters:**
- `name` (str): Tool identifier
- `description` (str): Tool description
- `func` (Callable): Python function to wrap
- `schema` (Dict[str, Any]): JSON schema for function parameters

#### `RequestsTool`

Tool for making HTTP API calls.

**Constructor Parameters:**
- `name` (str): Tool identifier
- `description` (str): Tool description
- `method` (str): HTTP method (GET, POST, PUT, DELETE, PATCH)
- `url_template` (str): URL template with `{param}` placeholders
- `headers` (Optional[Dict[str, str]]): Default headers
- `query_params_schema` (Optional[Dict[str, Any]]): Query parameter schema
- `body_schema` (Optional[Dict[str, Any]]): Request body schema
- `path_params` (Optional[List[str]]): Path parameter names
- `timeout` (int): Request timeout in seconds (default: 30)
- `auth` (Optional[tuple]): Basic auth credentials (username, password)

#### `OpenAIClient`

Client for OpenAI-compatible APIs.

**Constructor Parameters:**
- `api_key` (Optional[str]): API key (or from OPENAI_API_KEY env var)
- `model` (str): Model identifier (default: "gpt-4o")
- `base_url` (str): API base URL (default: OpenAI's URL)
- `organization` (Optional[str]): Organization ID
- `default_headers` (Optional[dict]): Default request headers

**Methods:**
- `call(messages: List[Message], **kwargs) -> str`: Get a completion
- `call_stream(messages: List[Message], **kwargs) -> Generator`: Stream completion

#### `OpenRouterClient`

Client for OpenRouter API (supports multiple LLM providers).

**Constructor Parameters:**
- `api_key` (Optional[str]): API key (or from OPENROUTER_API_KEY env var)
- `model` (str): Model identifier (default: "openai/gpt-4o")
- `site_url` (Optional[str]): Your site URL for rankings
- `app_name` (Optional[str]): Your app name for rankings

### Models

#### `Message`

Represents a conversation message.

**Attributes:**
- `role` (str): Message role ("user", "assistant", "system")
- `content` (str): Message content
- `tool_calls` (Optional[List[ToolCall]]): Tool calls in the message

#### `ToolCall`

Represents a tool invocation.

**Attributes:**
- `id` (str): Unique call identifier
- `name` (str): Tool name
- `parameters` (Dict[str, Any]): Tool parameters

#### `ToolResult`

Represents a tool execution result.

**Attributes:**
- `tool_call_id` (str): ID of the tool call
- `tool_name` (str): Name of the tool
- `result` (str): Execution result

### Exceptions

- `AgentError`: Base exception for all agent errors
- `ToolNotFoundError`: Tool not found in registry
- `ToolExecutionError`: Tool execution failed
- `LLMCallError`: LLM API call failed
- `ResponseParseError`: Failed to parse LLM response
- `MaxIterationsError`: Agent exceeded maximum iterations
- `InvalidToolSchemaError`: Tool schema validation failed

### Retry Configuration

#### `RetryConfig`

Configure retry behavior for LLM and tool calls.

**Constructor Parameters:**
- `max_attempts` (int): Maximum retry attempts (default: 3)
- `wait_min` (float): Minimum wait between retries in seconds (default: 1.0)
- `wait_max` (float): Maximum wait between retries in seconds (default: 10.0)
- `multiplier` (float): Exponential backoff multiplier (default: 2.0)

## Additional Information

### Current Status

This project is in **experimental phase** and is primarily for personal use. The following should be considered:

- **API Stability**: The API may change between versions without notice
- **Production Readiness**: Not recommended for production use yet
- **Documentation**: Documentation is being actively developed
- **Testing**: Test coverage is being expanded

### Known Limitations

- Limited to text-based interactions (no multimodal support yet)
- Streaming support requires additional dependencies (`jiter`)
- Tool execution is synchronous (no async support yet)
- Limited error recovery strategies for complex tool chains
- No built-in conversation persistence

### Planned Features

- Asynchronous tool execution
- Multimodal support (images, audio)
- Built-in conversation persistence and memory
- More pre-built tools for common tasks
- Better error handling and recovery
- Support for more LLM providers
- Tool composition and chaining utilities
- Improved streaming capabilities
- Plugin system for extensions

### Contributing

As this is a personal experimental project, contributions are not actively sought at this time. However, if you find bugs or have suggestions, feel free to open an issue on GitHub.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Support

For questions or issues, please use the [GitHub Issues](https://github.com/akstspace/acton-agent/issues) page.

---

**Disclaimer**: This is an experimental personal project. Use it at your own risk. The author makes no guarantees about stability, security, or fitness for any particular purpose.
