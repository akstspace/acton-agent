# Advanced Topics

This guide covers advanced usage patterns, performance optimization, production deployment, and best practices for Acton Agent.

## Table of Contents

- [Streaming Responses](#streaming-responses)
- [Custom Tool Development](#custom-tool-development)
- [Error Handling Strategies](#error-handling-strategies)
- [Performance Optimization](#performance-optimization)
- [Production Deployment](#production-deployment)
- [Security Considerations](#security-considerations)
- [Testing Agents](#testing-agents)
- [Debugging and Logging](#debugging-and-logging)
- [Multi-Agent Systems](#multi-agent-systems)

## Streaming Responses

Streaming provides real-time feedback as the agent processes requests.

### Basic Streaming

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import AgentToken, AgentFinalResponseEvent

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client, stream=True)

# Stream token-by-token
for event in agent.run_stream("Write a short poem about AI"):
    if isinstance(event, AgentToken):
        print(event.content, end="", flush=True)
    elif isinstance(event, AgentFinalResponseEvent):
        print("\n\nComplete!")
```

### Handling All Event Types

```python
from acton_agent.agent import (
    AgentStreamStart,
    AgentStreamEnd,
    AgentToken,
    AgentPlanEvent,
    AgentStepEvent,
    AgentToolResultsEvent,
    AgentFinalResponseEvent
)

for event in agent.run_stream("Complex query requiring tools"):
    if isinstance(event, AgentStreamStart):
        print("[Agent thinking...]")
    
    elif isinstance(event, AgentToken):
        print(event.content, end="", flush=True)
    
    elif isinstance(event, AgentStreamEnd):
        print("\n[Thinking complete]")
    
    elif isinstance(event, AgentPlanEvent):
        print(f"\n[Plan: {event.plan.plan}]")
    
    elif isinstance(event, AgentStepEvent):
        tools = [call.tool_name for call in event.step.tool_calls]
        print(f"\n[Executing tools: {', '.join(tools)}]")
    
    elif isinstance(event, AgentToolResultsEvent):
        print(f"\n[Tools complete: {len(event.results)} results]")
    
    elif isinstance(event, AgentFinalResponseEvent):
        print(f"\n\n=== Final Answer ===\n{event.response.final_answer}")
```

### Building Interactive UIs

```python
import sys
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import AgentToken

def stream_to_console(agent: Agent, query: str):
    """Stream agent response with visual indicators."""
    print("\n🤖 Assistant: ", end="", flush=True)
    
    buffer = ""
    for event in agent.run_stream(query):
        if isinstance(event, AgentToken):
            buffer += event.content
            print(event.content, end="", flush=True)
            sys.stdout.flush()
    
    print("\n")
    return buffer

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client, stream=True)

# Interactive loop
while True:
    query = input("\n💬 You: ")
    if query.lower() in ['exit', 'quit']:
        break
    
    response = stream_to_console(agent, query)
```

## Custom Tool Development

### Advanced Custom Tool

```python
from acton_agent.agent import Tool
from typing import Dict, Any
import sqlite3

class DatabaseQueryTool(Tool):
    """Tool for querying a SQLite database with safety checks."""
    
    def __init__(self, db_path: str, allowed_tables: list):
        super().__init__(
            name="query_database",
            description=f"Query database tables: {', '.join(allowed_tables)}"
        )
        self.db_path = db_path
        self.allowed_tables = set(allowed_tables)
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        query = parameters.get("query", "")
        
        # Security: Validate query
        if not self._is_safe_query(query):
            return "Error: Only SELECT queries on allowed tables are permitted"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return "No results found"
                
                # Format results
                columns = [desc[0] for desc in cursor.description]
                formatted = f"Columns: {', '.join(columns)}\n\n"
                
                for row in results[:10]:  # Limit to 10 rows
                    formatted += str(row) + "\n"
                
                if len(results) > 10:
                    formatted += f"\n... ({len(results) - 10} more rows)"
                
                return formatted
        
        except sqlite3.Error as e:
            return f"Error: Database error - {str(e)}"
    
    def _is_safe_query(self, query: str) -> bool:
        """Validate that query is a safe SELECT statement."""
        query_upper = query.upper().strip()
        
        # Only allow SELECT
        if not query_upper.startswith("SELECT"):
            return False
        
        # No write operations
        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
        if any(keyword in query_upper for keyword in dangerous_keywords):
            return False
        
        # Check table names
        for table in self.allowed_tables:
            if table.upper() in query_upper:
                return True
        
        return False
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"SQL SELECT query for tables: {', '.join(self.allowed_tables)}"
                }
            },
            "required": ["query"]
        }

# Usage
tool = DatabaseQueryTool(
    db_path="./mydata.db",
    allowed_tables=["users", "products", "orders"]
)
agent.register_tool(tool)
```

### Tool with Caching

```python
from acton_agent.agent import Tool
from typing import Dict, Any
from functools import lru_cache
import time

class CachedAPITool(Tool):
    """Tool with result caching for expensive operations."""
    
    def __init__(self, name: str, description: str, cache_ttl: int = 300):
        super().__init__(name, description)
        self.cache_ttl = cache_ttl
        self._cache = {}
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        # Create cache key from parameters
        cache_key = self._make_cache_key(parameters)
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return f"[Cached] {cached_data}"
        
        # Execute expensive operation
        result = self._do_expensive_operation(parameters)
        
        # Store in cache
        self._cache[cache_key] = (result, time.time())
        
        return result
    
    def _make_cache_key(self, parameters: Dict[str, Any]) -> str:
        # Simple cache key from sorted params
        items = sorted(parameters.items())
        return str(items)
    
    def _do_expensive_operation(self, parameters: Dict[str, Any]) -> str:
        # Override in subclass
        raise NotImplementedError
    
    def get_schema(self) -> Dict[str, Any]:
        raise NotImplementedError
```

### Tool with Progress Callbacks

```python
from acton_agent.agent import Tool
from typing import Dict, Any, Callable, Optional

class ProgressiveTool(Tool):
    """Tool that reports progress during long operations."""
    
    def __init__(
        self,
        name: str,
        description: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__(name, description)
        self.progress_callback = progress_callback
    
    def _report_progress(self, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        items = parameters.get("items", [])
        total = len(items)
        
        results = []
        for i, item in enumerate(items):
            # Report progress
            self._report_progress(f"Processing {i+1}/{total}: {item}")
            
            # Do work
            result = self._process_item(item)
            results.append(result)
        
        return f"Processed {total} items: {results}"
    
    def _process_item(self, item):
        # Override in subclass
        return item.upper()
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["items"]
        }

# Usage with progress callback
def print_progress(message: str):
    print(f"[PROGRESS] {message}")

tool = ProgressiveTool(
    name="batch_processor",
    description="Process items in batch",
    progress_callback=print_progress
)
```

## Error Handling Strategies

### Graceful Degradation

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import MaxIterationsError, LLMCallError

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client)

def safe_agent_run(agent: Agent, query: str, fallback: str = None) -> str:
    """Run agent with graceful error handling."""
    try:
        return agent.run(query)
    
    except MaxIterationsError:
        if fallback:
            return fallback
        return "I apologize, but I couldn't complete this request in the allotted time."
    
    except LLMCallError as e:
        if fallback:
            return fallback
        return f"I'm experiencing technical difficulties: {str(e.original_error)}"
    
    except Exception as e:
        if fallback:
            return fallback
        return f"An unexpected error occurred: {str(e)}"

# Usage
response = safe_agent_run(
    agent,
    "Complex query",
    fallback="I couldn't answer that, but here's what I know: ..."
)
```

### Retry with Backoff

```python
from acton_agent.agent import RetryConfig, Agent
from acton_agent.client import OpenAIClient
import time

# Configure aggressive retry for production
retry_config = RetryConfig(
    max_attempts=5,
    min_wait=2.0,
    max_wait=120.0,
    multiplier=3.0
)

client = OpenAIClient(model="gpt-4o")
agent = Agent(llm_client=client, retry_config=retry_config)

# The agent will automatically retry on failures with exponential backoff
response = agent.run("Query that might fail transiently")
```

### Circuit Breaker Pattern

```python
import time
from typing import Optional

class CircuitBreaker:
    """Prevent cascading failures in agent systems."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
            
            raise e

# Usage
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)

def make_agent_call(query: str) -> str:
    return circuit_breaker.call(agent.run, query)
```

## Performance Optimization

### Optimize Token Usage

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import SimpleAgentMemory

# Use smaller context window
client = OpenAIClient(model="gpt-3.5-turbo")  # Faster, cheaper

# Aggressive memory management
memory = SimpleAgentMemory(max_history_tokens=2000)

# Limit iterations
agent = Agent(
    llm_client=client,
    memory=memory,
    max_iterations=5,  # Prevent runaway loops
    system_prompt="Be concise."  # Encourage shorter responses
)
```

### Parallel Tool Execution

```python
from acton_agent.agent import Tool
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelToolExecutor:
    """Execute multiple independent tools in parallel."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
    
    def execute_tools(self, tool_calls: List[tuple]) -> List[str]:
        """
        Execute tools in parallel.
        
        Args:
            tool_calls: List of (tool, parameters) tuples
        
        Returns:
            List of results in same order as input
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(tool.execute, params): i
                for i, (tool, params) in enumerate(tool_calls)
            }
            
            # Collect results in order
            results = [None] * len(tool_calls)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = f"Error: {str(e)}"
            
            return results
```

### Caching LLM Responses

```python
from functools import lru_cache
import hashlib
import json

class CachedAgent:
    """Wrapper that caches agent responses."""
    
    def __init__(self, agent: Agent, cache_size: int = 128):
        self.agent = agent
        self.cache_size = cache_size
    
    @lru_cache(maxsize=128)
    def _cached_run(self, query_hash: str) -> str:
        # This won't work directly due to agent state
        # Use external cache like Redis in production
        pass
    
    def run(self, query: str) -> str:
        # Create cache key
        key = hashlib.md5(query.encode()).hexdigest()
        
        # Try cache (implement with Redis/memcached in production)
        # cached = redis_client.get(key)
        # if cached:
        #     return cached
        
        # Run agent
        result = self.agent.run(query)
        
        # Store in cache
        # redis_client.setex(key, 3600, result)  # 1 hour TTL
        
        return result
```

## Production Deployment

### Environment Configuration

```python
import os
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import RetryConfig, SimpleAgentMemory

def create_production_agent() -> Agent:
    """Create agent with production configuration."""
    
    # Load from environment
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("LLM_MODEL", "gpt-4o")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "15"))
    max_history = int(os.environ.get("MAX_HISTORY_TOKENS", "10000"))
    
    # Configure client
    client = OpenAIClient(
        api_key=api_key,
        model=model
    )
    
    # Production retry config
    retry_config = RetryConfig(
        max_attempts=5,
        min_wait=2.0,
        max_wait=120.0,
        multiplier=2.5
    )
    
    # Create agent
    agent = Agent(
        llm_client=client,
        max_iterations=max_iterations,
        retry_config=retry_config,
        memory=SimpleAgentMemory(max_history_tokens=max_history),
        timezone=os.environ.get("TIMEZONE", "UTC")
    )
    
    return agent
```

### Health Checks

```python
from acton_agent import Agent

def health_check(agent: Agent) -> dict:
    """Check agent health status."""
    try:
        # Simple test query
        response = agent.run("Respond with 'OK'")
        
        return {
            "status": "healthy",
            "tools": len(agent.list_tools()),
            "history_size": len(agent.get_conversation_history())
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    """Simple rate limiter for agent calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def acquire(self):
        """Block until rate limit allows."""
        now = time.time()
        
        # Remove old calls
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        # Wait if at limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.time_window - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls.popleft()
        
        self.calls.append(time.time())

# Usage
rate_limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute

def rate_limited_run(agent: Agent, query: str) -> str:
    rate_limiter.acquire()
    return agent.run(query)
```

## Security Considerations

### Input Validation

```python
import re

def validate_user_input(user_input: str) -> bool:
    """Validate user input for safety."""
    
    # Length check
    if len(user_input) > 10000:
        raise ValueError("Input too long")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"eval\(",
        r"exec\("
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError(f"Suspicious pattern detected: {pattern}")
    
    return True

# Usage
def safe_run(agent: Agent, user_input: str) -> str:
    validate_user_input(user_input)
    return agent.run(user_input)
```

### Sandboxed Tool Execution

```python
from acton_agent.agent import Tool, FunctionTool
from typing import Dict, Any
import subprocess

class SandboxedShellTool(Tool):
    """Execute shell commands in restricted environment."""
    
    ALLOWED_COMMANDS = {"ls", "cat", "grep", "echo"}
    
    def __init__(self):
        super().__init__(
            name="safe_shell",
            description="Execute safe shell commands"
        )
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        command = parameters.get("command", "")
        
        # Parse command
        parts = command.split()
        if not parts:
            return "Error: Empty command"
        
        # Validate command
        if parts[0] not in self.ALLOWED_COMMANDS:
            return f"Error: Command '{parts[0]}' not allowed"
        
        # Execute with timeout
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=5,
                cwd="/tmp"  # Restricted directory
            )
            return result.stdout or result.stderr
        
        except subprocess.TimeoutExpired:
            return "Error: Command timeout"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": f"Shell command (allowed: {', '.join(self.ALLOWED_COMMANDS)})"
                }
            },
            "required": ["command"]
        }
```

### API Key Management

```python
import os
from typing import Optional

class SecureConfig:
    """Secure configuration management."""
    
    @staticmethod
    def get_api_key(provider: str) -> Optional[str]:
        """Get API key from secure source."""
        
        # Try environment variable
        env_var = f"{provider.upper()}_API_KEY"
        key = os.environ.get(env_var)
        
        if key:
            return key
        
        # Try secrets file (production)
        secrets_path = os.environ.get("SECRETS_PATH", "/run/secrets")
        secret_file = os.path.join(secrets_path, f"{provider}_api_key")
        
        if os.path.exists(secret_file):
            with open(secret_file) as f:
                return f.read().strip()
        
        return None

# Usage
api_key = SecureConfig.get_api_key("openai")
client = OpenAIClient(api_key=api_key)
```

## Testing Agents

### Unit Testing Tools

```python
import unittest
from acton_agent.agent import FunctionTool

class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        def calc(a: float, b: float, op: str) -> float:
            ops = {"add": a+b, "sub": a-b, "mul": a*b, "div": a/b}
            return ops[op]
        
        self.tool = FunctionTool(
            name="calc",
            description="Calculator",
            func=calc,
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string"}
                },
                "required": ["a", "b", "op"]
            }
        )
    
    def test_addition(self):
        result = self.tool.execute({"a": 2, "b": 3, "op": "add"})
        self.assertEqual(result, "5")
    
    def test_schema(self):
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("a", schema["properties"])
```

### Mocking LLM Clients

```python
from typing import List
from acton_agent.agent import Message

class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
    
    def call(self, messages: List[Message], **kwargs) -> str:
        if self.call_count >= len(self.responses):
            return "Mock response"
        
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

# Usage in tests
def test_agent():
    mock_client = MockLLMClient([
        "FINAL_ANSWER: Test response"
    ])
    
    agent = Agent(llm_client=mock_client)
    result = agent.run("Test query")
    
    assert "Test response" in result
```

## Debugging and Logging

### Custom Logging

```python
from loguru import logger
import sys

# Configure detailed logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Add file logging
logger.add(
    "agent_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="INFO"
)

# Agent will use these logger settings
from acton_agent import Agent
agent = Agent(llm_client=client)
```

### Debug Mode

```python
from acton_agent import Agent
from acton_agent.client import OpenAIClient

def create_debug_agent() -> Agent:
    """Create agent with verbose debugging."""
    
    client = OpenAIClient(model="gpt-4o")
    
    agent = Agent(
        llm_client=client,
        max_iterations=3  # Limit for debugging
    )
    
    # Wrap run method for debugging
    original_run = agent.run
    
    def debug_run(query: str) -> str:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = original_run(query)
        
        print(f"\n{'='*60}")
        print(f"Result: {result}")
        print(f"History length: {len(agent.get_conversation_history())}")
        print(f"{'='*60}\n")
        
        return result
    
    agent.run = debug_run
    return agent
```

## Multi-Agent Systems

### Agent Orchestration

```python
from typing import List, Dict
from acton_agent import Agent

class MultiAgentOrchestrator:
    """Coordinate multiple specialized agents."""
    
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.router_agent = None  # Could add a router agent
    
    def route_query(self, query: str) -> str:
        """Route query to appropriate agent."""
        
        # Simple keyword-based routing
        if "weather" in query.lower():
            return self.agents["weather"].run(query)
        elif "calculate" in query.lower() or "math" in query.lower():
            return self.agents["calculator"].run(query)
        elif "code" in query.lower() or "program" in query.lower():
            return self.agents["coder"].run(query)
        else:
            # Default to general agent
            return self.agents["general"].run(query)
    
    def collaborative_solve(self, query: str) -> Dict[str, str]:
        """Get responses from multiple agents."""
        
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.run(query)
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        
        return results

# Usage
orchestrator = MultiAgentOrchestrator({
    "weather": weather_agent,
    "calculator": calc_agent,
    "coder": code_agent,
    "general": general_agent
})

response = orchestrator.route_query("What's the weather in Paris?")
```

## See Also

- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples/) - Practical code examples
- [Core Concepts](core-concepts.md) - Understanding the framework
