# Getting Started with Acton Agent

Welcome to Acton Agent! This guide will help you install the framework and create your first AI agent with tool execution capabilities.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Your First Agent](#your-first-agent)
- [Adding Tools](#adding-tools)
- [Configuration](#configuration)
- [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9 or higher** installed on your system
- An **API key** from a supported LLM provider (OpenAI, OpenRouter, etc.)
- Basic familiarity with Python programming

### Verify Python Version

```bash
python --version  # Should show 3.9 or higher
```

If you need to install or upgrade Python, visit [python.org](https://www.python.org/downloads/).

## Installation

### Basic Installation

Install Acton Agent using pip:

```bash
pip install acton-agent
```

This installs the core framework with minimal dependencies.

### Installation with LLM Provider Support

For **OpenAI** support (GPT-4, GPT-3.5-turbo, etc.):

```bash
pip install acton-agent[openai]
```

For **Groq** support:

```bash
pip install acton-agent[groq]
```

### Installation for Development

If you plan to contribute or develop with Acton Agent:

```bash
pip install acton-agent[dev]
```

This includes testing tools (pytest), linting (ruff), type checking (mypy), and more.

### Install All Optional Dependencies

```bash
pip install acton-agent[all]
```

### Installation from Source

To install the latest development version:

```bash
git clone https://github.com/akstspace/acton-agent.git
cd acton-agent
pip install -e .
```

## Your First Agent

Let's create a simple agent that can answer questions. This example uses OpenAI's GPT-4.

### Step 1: Set Up Your API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or on Windows:

```cmd
set OPENAI_API_KEY=your-api-key-here
```

### Step 2: Create a Basic Agent

Create a file called `my_first_agent.py`:

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient

# Initialize the LLM client
client = OpenAIClient(
    api_key="your-api-key",  # Or omit to use OPENAI_API_KEY env var
    model="gpt-4o"
)

# Create the agent
agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant."
)

# Run the agent
response = agent.run("What is the capital of France?")
print(response)
```

### Step 3: Run Your Agent

```bash
python my_first_agent.py
```

**Expected Output:**
```
The capital of France is Paris.
```

🎉 Congratulations! You've created your first AI agent.

## Adding Tools

Agents become powerful when they can use tools. Let's add a calculator tool.

### Example: Calculator Tool

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool

# Define a calculator function
def calculate(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else 0
    }
    return operations.get(operation, 0)

# Initialize client and agent
client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)

# Create and register the tool
calculator_tool = FunctionTool(
    name="calculator",
    description="Perform basic arithmetic: add, subtract, multiply, divide",
    func=calculate,
    schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            }
        },
        "required": ["a", "b", "operation"]
    }
)

agent.register_tool(calculator_tool)

# Use the agent with the tool
result = agent.run("What is 156 multiplied by 23?")
print(result)
```

**Expected Output:**
```
The result of 156 multiplied by 23 is 3,588.
```

### How It Works

1. **Define a function** - Any Python function can become a tool
2. **Create a FunctionTool** - Wrap the function with a JSON schema
3. **Register the tool** - Add it to the agent's tool registry
4. **Use naturally** - The agent automatically decides when to call the tool

## Configuration

### Customizing Agent Behavior

You can configure various aspects of your agent:

```python
from acton_agent import Agent, SimpleAgentMemory
from acton_agent.client import OpenAIClient
from acton_agent.agent import RetryConfig

client = OpenAIClient(model="gpt-4o")

agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful coding assistant.",
    max_iterations=10,  # Maximum reasoning steps
    retry_config=RetryConfig(max_attempts=3, min_wait=1, max_wait=10),
    memory=SimpleAgentMemory(max_history_tokens=8000),
    timezone="America/New_York"  # For timestamp in system messages
)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm_client` | LLM client instance (required) | - |
| `system_prompt` | Custom instructions for the agent | None |
| `max_iterations` | Max reasoning steps before timeout | 10 |
| `retry_config` | Retry settings for LLM/tool calls | Default config |
| `memory` | Memory management instance | SimpleAgentMemory(8000) |
| `stream` | Enable streaming responses | False |
| `timezone` | Timezone for system timestamps | "UTC" |

### Retry Configuration

Control how the agent handles failures:

```python
from acton_agent.agent import RetryConfig

retry_config = RetryConfig(
    max_attempts=5,      # Try up to 5 times
    min_wait=1,          # Wait at least 1 second between retries
    max_wait=60,         # Wait at most 60 seconds
    multiplier=2,        # Double wait time each retry
    retry_on_exceptions=(Exception,)  # Which exceptions to retry
)

agent = Agent(llm_client=client, retry_config=retry_config)
```

## Common Setup Issues

### Issue: `ModuleNotFoundError: No module named 'openai'`

**Solution:** Install the OpenAI optional dependency:
```bash
pip install acton-agent[openai]
```

### Issue: `ValueError: OpenAI API key must be provided`

**Solution:** Set the `OPENAI_API_KEY` environment variable or pass it explicitly:
```python
client = OpenAIClient(api_key="your-key-here")
```

### Issue: Agent reaches max iterations without answer

**Solution:** Increase `max_iterations` or simplify your query:
```python
agent = Agent(llm_client=client, max_iterations=20)
```

### Issue: Rate limit errors from API

**Solution:** Configure retry settings with longer waits:
```python
retry_config = RetryConfig(max_wait=120, multiplier=3)
agent = Agent(llm_client=client, retry_config=retry_config)
```

## Next Steps

Now that you have a basic agent running, explore:

- **[Core Concepts](core-concepts.md)** - Deep dive into agents, tools, and memory
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples/)** - More practical examples

### Quick Links

- [Creating Custom Tools](core-concepts.md#creating-tools)
- [Memory Management](core-concepts.md#memory-management)
