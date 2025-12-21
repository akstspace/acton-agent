# FAQ and Troubleshooting

Common questions and solutions for Acton Agent.

## Table of Contents

- [General Questions](#general-questions)
- [Installation Issues](#installation-issues)
- [Agent Behavior](#agent-behavior)
- [Tool Problems](#tool-problems)
- [Performance Issues](#performance-issues)
- [Error Messages](#error-messages)

## General Questions

### What is Acton Agent?

Acton Agent is a lightweight Python framework for building LLM-powered agents with tool execution capabilities. It enables AI agents to interact with external APIs, execute Python functions, and maintain conversation context.

### Is Acton Agent production-ready?

⚠️ **No**, Acton Agent is currently an experimental project. The API may change without notice. Use at your own discretion in production environments.

### Which LLM providers are supported?

- **OpenAI** (GPT-4, GPT-3.5, etc.) via `OpenAIClient`
- **OpenRouter** (multiple providers) via `OpenRouterClient`
- **Any OpenAI-compatible API** via `OpenAIClient` with custom `base_url`
- **Custom providers** by implementing the `LLMClient` protocol

### Can I use local models?

Yes! Use `OpenAIClient` with a custom `base_url`:

```python
from acton_agent.client import OpenAIClient

client = OpenAIClient(
    api_key="not-needed",
    model="llama-3",
    base_url="http://localhost:8000/v1"
)
```

Works with:
- Ollama (with OpenAI compatibility)
- LM Studio
- vLLM
- LocalAI
- Any OpenAI-compatible endpoint

### Can agents use multiple tools?

Yes! Register as many tools as you need:

```python
agent.register_tool(calculator_tool)
agent.register_tool(weather_tool)
agent.register_tool(database_tool)
# ... register more

# Agent automatically decides which tools to use
response = agent.run("What's the weather and what's 2+2?")
```

### Does it support async/await?

Not currently. All operations are synchronous. Async support may be added in future versions.

## Installation Issues

### ModuleNotFoundError: No module named 'openai'

**Problem:** OpenAI dependency not installed

**Solution:**
```bash
pip install acton-agent[openai]
```

### ImportError: cannot import name 'Agent'

**Problem:** Old version installed or installation incomplete

**Solution:**
```bash
pip uninstall acton-agent
pip install --upgrade acton-agent
```

### Python version incompatibility

**Problem:** Python version too old

**Solution:** Acton Agent requires Python 3.9+
```bash
python --version  # Check version
# Upgrade Python if needed
pip install acton-agent
```

### SSL Certificate errors

**Problem:** SSL verification failing

**Solution:** Update certificates or use environment variable:
```bash
# macOS
/Applications/Python*/Install\ Certificates.command

# Or temporarily disable (not recommended)
export PYTHONHTTPSVERIFY=0
```

## Agent Behavior

### Agent doesn't use my tools

**Possible causes and solutions:**

1. **Tool schema is unclear**
   ```python
   # Bad: Vague description
   tool = FunctionTool(
       name="thing",
       description="Does stuff",
       ...
   )
   
   # Good: Clear description
   tool = FunctionTool(
       name="calculator",
       description="Perform arithmetic operations: add, subtract, multiply, divide",
       ...
   )
   ```

2. **System prompt conflicts**
   ```python
   # Avoid telling agent NOT to use tools
   agent = Agent(
       llm_client=client,
       system_prompt="Answer directly without using tools"  # ❌
   )
   ```

3. **Query doesn't require tools**
   ```python
   # This doesn't need a calculator
   agent.run("What is two plus two?")  # Agent knows: 4
   
   # This makes tool use more likely
   agent.run("Calculate 1523 * 847")  # Agent uses calculator
   ```

### Agent reaches max iterations

**Problem:** Agent doesn't produce final answer in time

**Solutions:**

1. **Increase max_iterations:**
   ```python
   agent = Agent(llm_client=client, max_iterations=20)
   ```

2. **Simplify the query:**
   ```python
   # Instead of:
   "Research X, analyze Y, compare Z, and write a report"
   
   # Break into steps:
   agent.run("First, research X")
   agent.run("Now analyze Y")
   # ...
   ```

3. **Check tool reliability:**
   Tools that frequently fail can cause iteration loops.

4. **Improve system prompt:**
   ```python
   agent = Agent(
       llm_client=client,
       system_prompt="Answer efficiently. Use tools only when necessary."
   )
   ```

### Agent gives wrong answers

**Debugging steps:**

1. **Check conversation history:**
   ```python
   history = agent.get_conversation_history()
   for msg in history:
       print(f"{msg.role}: {msg.content}")
   ```

2. **Verify tool outputs:**
   Test tools independently:
   ```python
   result = my_tool.execute({"param": "value"})
   print(result)  # Is this correct?
   ```

3. **Try different model:**
   ```python
   # Try a more capable model
   client = OpenAIClient(model="gpt-4o")
   ```

4. **Reset conversation:**
   ```python
   agent.reset()  # Clear history
   ```

### Conversation loses context

**Problem:** Agent forgets previous messages

**Cause:** Memory management truncating history

**Solutions:**

1. **Increase memory limit:**
   ```python
   from acton_agent.agent import SimpleAgentMemory
   
   memory = SimpleAgentMemory(max_history_tokens=15000)
   agent = Agent(llm_client=client, memory=memory)
   ```

2. **Disable memory management:**
   ```python
   agent = Agent(llm_client=client, memory=None)
   ```

3. **Use custom memory strategy:**
   Implement `AgentMemory` to preserve important messages.

## Tool Problems

### Tool not found error

**Problem:** `ToolNotFoundError: Tool 'xyz' not found`

**Solutions:**

1. **Check tool name:**
   ```python
   # Tool registered as "calculator"
   agent.register_tool(calculator_tool)
   
   # LLM tries to call "calc" - won't work
   # Make description clear about name
   ```

2. **Verify registration:**
   ```python
   print(agent.list_tools())  # Check registered tools
   ```

3. **Re-register if needed:**
   ```python
   agent.register_tool(my_tool)
   ```

### Tool execution fails

**Problem:** `ToolExecutionError`

**Debugging:**

1. **Test tool directly:**
   ```python
   try:
       result = my_tool.execute({"param": "value"})
       print(result)
   except Exception as e:
       print(f"Tool error: {e}")
   ```

2. **Check parameters:**
   ```python
   # Log what parameters agent is sending
   class DebugTool(Tool):
       def execute(self, parameters):
           print(f"Received: {parameters}")
           # ... rest of implementation
   ```

3. **Add error handling:**
   ```python
   def safe_function(**kwargs):
       try:
           # Your logic
           return result
       except Exception as e:
           return f"Error: {str(e)}"
   ```

### RequestsTool timeouts

**Problem:** HTTP requests timing out

**Solutions:**

1. **Increase timeout:**
   ```python
   tool = RequestsTool(
       name="slow_api",
       description="...",
       url_template="...",
       timeout=120  # 2 minutes
   )
   ```

2. **Check API availability:**
   Test the endpoint directly with `curl` or Postman

3. **Add retries:**
   ```python
   agent = Agent(
       llm_client=client,
       retry_config=RetryConfig(max_attempts=5)
   )
   ```

### Invalid tool schema error

**Problem:** `InvalidToolSchemaError`

**Cause:** Schema doesn't follow JSON Schema format

**Solution:**
```python
# ❌ Invalid schema
schema = {
    "param": "string"  # Wrong format
}

# ✅ Valid schema
schema = {
    "type": "object",
    "properties": {
        "param": {
            "type": "string",
            "description": "Parameter description"
        }
    },
    "required": ["param"]
}
```

## Performance Issues

### Agent is slow

**Possible causes:**

1. **Large conversation history:**
   - Solution: Use memory management
   - Or: Call `agent.reset()` periodically

2. **Slow LLM model:**
   - Solution: Use faster model (e.g., gpt-3.5-turbo)

3. **Tool execution delays:**
   - Solution: Optimize tool implementations
   - Solution: Add caching

4. **Network latency:**
   - Solution: Use LLM provider closer to your location
   - Solution: Use streaming for perceived speed

### High token usage / costs

**Solutions:**

1. **Use smaller context window:**
   ```python
   memory = SimpleAgentMemory(max_history_tokens=2000)
   ```

2. **Shorter system prompt:**
   ```python
   agent = Agent(
       llm_client=client,
       system_prompt="You are a helpful assistant."  # Concise
   )
   ```

3. **Limit iterations:**
   ```python
   agent = Agent(llm_client=client, max_iterations=5)
   ```

4. **Use cheaper model:**
   ```python
   client = OpenAIClient(model="gpt-3.5-turbo")
   ```

5. **Reset conversation history:**
   ```python
   agent.reset()  # Don't carry unnecessary history
   ```

### Memory leaks

**Problem:** Memory usage growing over time

**Cause:** Long-running agent keeping history

**Solution:**
```python
# Periodically reset
agent.reset()

# Or: Use aggressive memory management
memory = SimpleAgentMemory(max_history_tokens=1000)
```

## Error Messages

### ValueError: OpenAI API key must be provided

**Problem:** API key not set

**Solutions:**

1. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Pass explicitly:**
   ```python
   client = OpenAIClient(api_key="sk-...")
   ```

### LLMCallError: LLM call failed after N retries

**Possible causes:**

1. **Invalid API key**
2. **Rate limiting**
3. **Network issues**
4. **Service outage**

**Solutions:**

1. **Verify API key:**
   Test with curl or OpenAI playground

2. **Increase retry wait:**
   ```python
   retry_config = RetryConfig(
       max_attempts=5,
       max_wait=120
   )
   ```

3. **Check API status:**
   Visit status.openai.com

### ResponseParseError: Failed to parse agent response

**Problem:** LLM response doesn't match expected format

**Debugging:**

1. **Check raw response:**
   The error includes the raw response text

2. **Model compatibility:**
   Some models may not follow instructions well
   - Try a more capable model

3. **Custom format instructions:**
   Don't override unless necessary

## Common Patterns

### How do I make the agent always use a specific tool?

**In system prompt:**
```python
agent = Agent(
    llm_client=client,
    system_prompt="Always use the calculator tool for any math operations."
)
```

### How do I prevent infinite loops?

**Set low max_iterations:**
```python
agent = Agent(llm_client=client, max_iterations=5)
```

### How do I handle rate limits?

**Increase retry delays:**
```python
retry_config = RetryConfig(
    max_wait=300,  # 5 minutes
    multiplier=3.0
)
agent = Agent(llm_client=client, retry_config=retry_config)
```

### How do I debug what the agent is doing?

**Enable detailed logging:**
```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Now agent operations will be logged in detail
```

### How do I make responses more concise?

**In system prompt:**
```python
agent = Agent(
    llm_client=client,
    system_prompt="Be concise. Answer in 2-3 sentences maximum."
)
```

## Still Need Help?

1. **Check examples:** Review the [examples directory](examples/README.md)
2. **Read docs:** See [documentation](getting-started.md)
3. **GitHub Issues:** Search or create an issue at [github.com/akstspace/acton-agent/issues](https://github.com/akstspace/acton-agent/issues)
4. **Enable debug logging:** See what's happening internally

## See Also

- [Getting Started](getting-started.md) - Installation and setup
- [Core Concepts](core-concepts.md) - Understanding the framework
- [Examples](examples/README.md) - Working code examples
