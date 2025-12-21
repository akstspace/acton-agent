<p align="center">
  <img src="images/acton-icon.jpg" alt="Acton Agent" width="200"/>
</p>

# Acton Agent

**A lightweight, flexible Python framework for building LLM agents with tool execution capabilities**

> ⚠️ **Experimental Project**: This is a personal project currently in an experimental phase. The API may change without notice, and features may be incomplete or unstable. Use at your own discretion.

[![PyPI version](https://badge.fury.io/py/acton-agent.svg)](https://badge.fury.io/py/acton-agent)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Acton Agent enables you to build AI agents that can interact with external APIs, execute custom Python functions, and maintain conversation context. With minimal configuration, you can create agents that reason through complex tasks, call tools, and stream responses in real-time.

## Quick Start

```bash
pip install acton-agent[openai]
```

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool

# Create a simple calculator tool
def calculate(a: float, b: float, operation: str) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops.get(operation, 0)

# Initialize agent with OpenAI
client = OpenAIClient(api_key="your-key", model="gpt-4o")
agent = Agent(llm_client=client)

# Register the tool
agent.register_tool(FunctionTool(
    name="calculator",
    description="Perform basic arithmetic operations",
    func=calculate,
    schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}
        },
        "required": ["a", "b", "operation"]
    }
))

# Run the agent
result = agent.run("What is 25 multiplied by 4?")
print(result)  # "The result of 25 multiplied by 4 is 100."
```

## Key Features

### 🔧 **Flexible Tool System**
Create tools from Python functions, HTTP APIs, or custom classes. Organize related tools with ToolSets and shared configuration.
```python
from acton_agent import ToolSet, FunctionTool

# Define tools that use an API key
def get_weather(city: str, api_key: str) -> str:
    # api_key will be automatically injected from toolset_params
    return f"Weather in {city}: Sunny, 72°F"

def get_forecast(city: str, days: int, api_key: str) -> str:
    return f"{days}-day forecast for {city}"

# Group related tools with shared parameters
weather_tools = ToolSet(
    name="weather",
    description="Weather data tools",
    tools=[
        FunctionTool(
            name="current_weather",
            description="Get current weather for a city",
            func=get_weather,
            schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        ),
        FunctionTool(
            name="forecast",
            description="Get weather forecast",
            func=get_forecast,
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city", "days"]
            }
        )
    ],
    toolset_params={"api_key": "secret-api-key"}  # Hidden from LLM, auto-injected
)
agent.register_toolset(weather_tools)
```

### 🔄 **Automatic Retry & Error Handling**
Built-in retry logic with exponential backoff for both LLM calls and tool execution.
```python
from acton_agent.agent import RetryConfig

agent = Agent(llm_client=client, retry_config=RetryConfig(max_attempts=5))
```

### 💬 **Conversation Memory Management**
Automatic token-based history truncation to stay within context limits.
```python
from acton_agent import SimpleAgentMemory

agent = Agent(llm_client=client, memory=SimpleAgentMemory(max_history_tokens=8000))
```

### 🌊 **Streaming Support**
Stream agent responses token-by-token for real-time feedback.
```python
from acton_agent.agent import AgentToken

for event in agent.run_stream("Tell me a story"):
    if isinstance(event, AgentToken):
        print(event.content, end="", flush=True)
```

### 🔌 **Multi-Provider Support**
Works with OpenAI, OpenRouter, and any OpenAI-compatible API.
```python
from acton_agent import OpenRouterClient

client = OpenRouterClient(api_key="your-key", model="anthropic/claude-3-opus")
```

## Documentation

- **[Getting Started](docs/getting-started.md)** - Installation and first steps
- **[Core Concepts](docs/core-concepts.md)** - Understanding agents, tools, and memory
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Examples](docs/examples/)** - Practical examples from basic to advanced

## Examples

Explore complete examples in the [`examples/`](examples/) directory:

- **[Function Tools](examples/function_tool_example.py)** - Wrap Python functions as agent tools
- **[API Integration](examples/requests_tool_example.py)** - Connect to REST APIs
- **[Custom Tools](examples/custom_tool_example.py)** - Build custom tool classes
- **[ToolSet Parameters](examples/toolset_params_example.py)** - Use hidden parameters for API keys and configuration

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This is an experimental project. Use at your own risk. No guarantees about stability, security, or fitness for any particular purpose.
