# Acton Agent Examples

This directory contains complete, runnable examples demonstrating various features of the Acton Agent framework.

## Prerequisites

Before running these examples, make sure you have:

1. Installed acton-agent:
   ```bash
   pip install acton-agent
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Available Examples

### 1. RequestsTool Example (`requests_tool_example.py`)

Demonstrates how to use the `RequestsTool` to enable your agent to make HTTP API calls.

**Features shown:**
- Creating RequestsTool instances with query parameters
- Using the `create_api_tool` helper function
- Path parameters in URLs
- Multiple API tools working together

**Run it:**
```bash
python examples/requests_tool_example.py
```

### 2. FunctionTool Example (`function_tool_example.py`)

Shows how to wrap Python functions as tools that can be called by the agent.

**Features shown:**
- Creating FunctionTool with custom Python functions
- Defining JSON schemas for function parameters
- Calculator, time, word count, and string manipulation tools
- Multi-step reasoning with function calls

**Run it:**
```bash
python examples/function_tool_example.py
```

### 3. Streaming Example (`streaming_example.py`)

Demonstrates real-time streaming of agent responses.

**Features shown:**
- Enabling streaming mode
- Handling different event types (content, tool_call, tool_result)
- Real-time output display
- Streaming with and without tool execution

**Run it:**
```bash
python examples/streaming_example.py
```

### 4. Custom Tool Example (`custom_tool_example.py`)

Shows how to create custom tool classes by subclassing the `Tool` base class.

**Features shown:**
- Creating custom tool classes
- Implementing `execute()` and `get_schema()` methods
- Weather simulation tool
- Dice roller tool
- Text analyzer tool

**Run it:**
```bash
python examples/custom_tool_example.py
```

## Example Structure

Each example follows a similar structure:

1. **Import statements** - Required modules and classes
2. **Helper functions or custom classes** - Tool definitions or utilities
3. **main() function** - Sets up the agent and demonstrates usage
4. **Multiple queries** - Shows different use cases and capabilities

## Customization

Feel free to modify these examples:

- Change the OpenAI model (e.g., to `gpt-3.5-turbo` or `gpt-4-turbo`)
- Add your own tools
- Modify the system prompts
- Try different queries
- Combine features from multiple examples

## Tips

- **API Keys**: Never hardcode API keys in your scripts. Use environment variables instead.
- **Error Handling**: These examples include basic error handling. In production, you may want more robust error management.
- **Rate Limits**: Be mindful of API rate limits when making multiple calls.
- **Costs**: Using OpenAI APIs incurs costs. Monitor your usage to avoid unexpected charges.

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'acton_agent'`
- **Solution:** Install the package: `pip install acton-agent`

**Problem:** `Error: Please set OPENAI_API_KEY environment variable`
- **Solution:** Set your API key: `export OPENAI_API_KEY="your-key"`

**Problem:** Streaming example not working
- **Solution:** Streaming is now included by default, no additional installation needed

**Problem:** API calls failing
- **Solution:** Check your internet connection and verify the API endpoint is accessible

## More Resources

- [Main README](../README.md) - Full documentation
- [GitHub Repository](https://github.com/akstspace/acton-agent) - Source code and issues
- API documentation is available in the source code docstrings

## Contributing

If you create an interesting example and would like to share it, feel free to open a pull request or issue on GitHub!
