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

### Example 3: Streaming Responses

You can stream responses from the agent in real-time:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import RequestsTool

# Initialize the client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4o"
)

# Create an agent with streaming enabled
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant.",
    stream=True
)

# Add a tool (optional)
posts_tool = RequestsTool(
    name="get_posts",
    description="Fetch posts from JSONPlaceholder API",
    method="GET",
    url_template="https://jsonplaceholder.typicode.com/posts"
)
agent.register_tool(posts_tool)

# Stream the response
for event in agent.run_stream("Tell me about post number 1"):
    if event.get("type") == "content":
        print(event.get("data"), end="", flush=True)
    elif event.get("type") == "tool_call":
        print(f"\n[Calling tool: {event.get('tool_name')}]\n")
    elif event.get("type") == "tool_result":
        print(f"\n[Tool result received]\n")
print()  # Final newline
```

## More Examples

For complete, runnable examples, check out the [examples](examples/) directory:

- [examples/requests_tool_example.py](examples/requests_tool_example.py) - API integration with RequestsTool
- [examples/function_tool_example.py](examples/function_tool_example.py) - Custom Python function tools
- [examples/streaming_example.py](examples/streaming_example.py) - Real-time streaming responses
- [examples/custom_tool_example.py](examples/custom_tool_example.py) - Building custom tool classes

## API Documentation

For detailed API documentation, please refer to the docstrings in the source code or visit our [GitHub repository](https://github.com/akstspace/acton-agent).

## Additional Information

### Current Status

This project is in **experimental phase** and is primarily for personal use. The following should be considered:

- **API Stability**: The API may change between versions without notice
- **Production Readiness**: Not recommended for production use yet
- **Documentation**: Documentation is being actively developed
- **Testing**: Test coverage is being expanded

### Contributing

As this is a personal experimental project, contributions are not actively sought at this time. However, if you find bugs or have suggestions, feel free to open an issue on GitHub.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Support

For questions or issues, please use the [GitHub Issues](https://github.com/akstspace/acton-agent/issues) page.

---

**Disclaimer**: This is an experimental personal project. Use it at your own risk. The author makes no guarantees about stability, security, or fitness for any particular purpose.
