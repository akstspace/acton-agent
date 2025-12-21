# Architecture Overview

This document explains the architecture and design decisions behind Acton Agent.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Application                       │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │          Agent                │
         │                               │
         │  • Orchestration              │
         │  • Conversation Management    │
         │  • Iteration Control          │
         └──────┬──────────────┬─────────┘
                │              │
       ┌────────▼──────┐  ┌───▼──────────┐
       │  LLM Client   │  │ Tool Registry │
       │               │  │               │
       │  • OpenAI     │  │  • Function   │
       │  • OpenRouter │  │  • Requests   │
       │  • Custom     │  │  • Custom     │
       └───────┬───────┘  └───┬──────────┘
               │              │
       ┌───────▼──────┐  ┌───▼──────────┐
       │  Provider    │  │ Tools        │
       │  API         │  │              │
       └──────────────┘  └──────────────┘
```

## Core Components

### 1. Agent

The **Agent** is the central orchestrator that manages:
- **LLM interactions** - Calling the language model
- **Tool execution** - Running registered tools
- **Conversation state** - Maintaining message history
- **Iteration control** - Managing reasoning loops
- **Memory management** - Applying memory strategies

**Design:** The Agent class is the only component users typically instantiate directly.

### 2. LLM Client

Implements the `LLMClient` protocol to provide a consistent interface across different LLM providers.

**Protocol-based design** allows:
- Easy swapping of LLM providers
- Testing with mock clients
- Custom client implementations

**Built-in clients:**
- `OpenAIClient` - For OpenAI and compatible APIs
- `OpenRouterClient` - For OpenRouter's multi-provider access

### 3. Tools

Tools follow a simple interface:
- `execute(parameters)` - Run the tool
- `get_schema()` - Describe parameters

**Three implementation patterns:**
1. **FunctionTool** - Wrap existing functions
2. **RequestsTool** - HTTP API calls
3. **Custom Tool** - Inherit from Tool base class

### 4. Tool Registry

Manages the collection of available tools with:
- Registration and unregistration
- Lookup by name
- Schema formatting for prompts

**Design:** Single registry per agent instance, automatically managed.

### 5. Memory Management

Abstract `AgentMemory` protocol allows custom strategies:
- Token-based truncation (SimpleAgentMemory)
- Sliding window
- Summarization
- Semantic compression

**Design:** Pluggable memory management separated from agent logic.

## Data Flow

### Standard Agent Run

```
User Input
    │
    ▼
┌──────────────────────┐
│ Agent.run(input)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Build Messages       │◄──── System Prompt
│  • System            │      Tool Schemas
│  • History           │      Conversation History
│  • User Input        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ LLM Client.call()    │────► LLM Provider API
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Parse Response       │
│  • AgentPlan?        │
│  • AgentStep?        │
│  • FinalResponse?    │
└──────┬───────────────┘
       │
       ├─── AgentStep ──────────┐
       │                        ▼
       │              ┌──────────────────┐
       │              │ Execute Tools    │
       │              │  • Retry logic   │
       │              │  • Error handle  │
       │              └────┬─────────────┘
       │                   │
       │                   ▼
       │              ┌──────────────────┐
       │              │ Format Results   │
       │              │ Add to History   │
       │              └────┬─────────────┘
       │                   │
       │◄──────────────────┘
       │ (Continue iteration)
       │
       └─── FinalResponse ───► Return to User
```

### Streaming Flow

```
User Input
    │
    ▼
Agent.run_stream(input)
    │
    ├──► AgentStreamStart
    │
    ├──► AgentToken ──────────┐
    ├──► AgentToken            │ (Multiple)
    ├──► AgentToken ──────────┘
    │
    ├──► AgentStreamEnd
    │
    ├──► AgentPlanEvent (optional)
    │
    ├──► AgentStepEvent ──────┐
    │                          │
    ├──► ToolExecution Events  │ (If tools called)
    │                          │
    ├──► AgentToolResults ────┘
    │
    └──► AgentFinalResponse
```

## Design Patterns

### 1. Protocol-Oriented Design

Using Python protocols instead of abstract base classes for flexibility:

```python
class LLMClient(Protocol):
    def call(self, messages: List[Message], **kwargs) -> str:
        ...
```

**Benefits:**
- Duck typing support
- No forced inheritance
- Easier testing

### 2. Composition Over Inheritance

Agent composes smaller components rather than inheriting behavior:

```python
class Agent:
    def __init__(self, llm_client, memory, retry_config):
        self.llm_client = llm_client
        self.memory = memory
        self.retry_config = retry_config
        self.tool_registry = ToolRegistry()
```

**Benefits:**
- Flexible component swapping
- Better testability
- Clear responsibilities

### 3. Structured Data with Pydantic

All data models use Pydantic for validation:

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
```

**Benefits:**
- Type safety
- Automatic validation
- JSON serialization
- IDE autocomplete

### 4. Retry Pattern with Tenacity

Automatic retries with exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=60)
)
def call_llm():
    # ... implementation
```

**Benefits:**
- Handles transient failures
- Configurable behavior
- Production-ready

### 5. Event-Driven Streaming

Generator-based streaming with typed events:

```python
def run_stream() -> Generator[StreamingEvent, None, None]:
    yield AgentStreamStart(...)
    yield AgentToken(...)
    # ...
```

**Benefits:**
- Real-time feedback
- Memory efficient
- Type-safe events

## Extension Points

### Custom LLM Client

```python
class MyLLMClient:
    def call(self, messages: List[Message], **kwargs) -> str:
        # Your implementation
        pass
```

### Custom Tool

```python
class MyTool(Tool):
    def execute(self, parameters: Dict) -> str:
        # Your implementation
        pass
    
    def get_schema(self) -> Dict:
        # Return JSON Schema
        pass
```

### Custom Memory

```python
class MyMemory(AgentMemory):
    def manage_history(self, history: List[Message]) -> List[Message]:
        # Your strategy
        pass
```

## Error Handling Strategy

```
┌─────────────────┐
│  User Code      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Layer    │──► MaxIterationsError
│                 │──► AgentError (base)
└────────┬────────┘
         │
    ┌────┼────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│  LLM   │ │  Tools   │
│ Client │ │ Registry │
└────┬───┘ └────┬─────┘
     │          │
     ▼          ▼
  LLMCallError  ToolExecutionError
                ToolNotFoundError
```

**Error Hierarchy:**
- `AgentError` - Base for all agent errors
- `LLMCallError` - LLM communication failures
- `ToolExecutionError` - Tool execution failures
- `ToolNotFoundError` - Missing tool
- `MaxIterationsError` - Iteration limit reached
- `ResponseParseError` - Invalid LLM response
- `InvalidToolSchemaError` - Invalid tool schema

## Performance Considerations

### 1. Memory Management

- Automatic history truncation prevents token overflow
- Configurable limits based on model context size
- Lazy evaluation where possible

### 2. Retry Logic

- Exponential backoff prevents overwhelming services
- Configurable retry counts and delays
- Retry only on recoverable errors

### 3. Streaming

- Generator-based for memory efficiency
- Events emitted as they occur
- No buffering of complete response

### 4. Tool Execution

- Sequential by default for simplicity
- Could be parallelized for independent tools
- Retry logic per tool

## Thread Safety

**Current Status:** Not thread-safe

The Agent class maintains mutable state (conversation history) and is not designed for concurrent access.

**Recommendations:**
- Create one agent instance per thread/request
- Use a pool of agents for high concurrency
- Don't share agent instances across threads

## Future Extensibility

The architecture supports future enhancements:

1. **Parallel Tool Execution** - Execute independent tools concurrently
2. **Streaming Tool Results** - Stream tool outputs in real-time
3. **Multi-Agent Collaboration** - Agents working together
4. **Advanced Memory** - Semantic search, summarization
5. **Function Calling API** - Native LLM tool calling support
6. **Async Support** - Async/await throughout

## Design Trade-offs

### Simplicity vs. Features

**Choice:** Favor simplicity
- Clean, understandable API
- Fewer abstractions
- Easy to get started

### Flexibility vs. Opinionation

**Choice:** Flexible with good defaults
- Protocol-based extension points
- Sensible default configurations
- Override anything if needed

### Performance vs. Reliability

**Choice:** Reliability first
- Automatic retries
- Conservative defaults
- Graceful degradation

## Comparison to Other Frameworks

### vs. LangChain

- **Acton:** Simpler, more focused on tool execution
- **LangChain:** More features, steeper learning curve

### vs. AutoGen

- **Acton:** Single-agent focused, lightweight
- **AutoGen:** Multi-agent systems, heavier

### vs. Haystack

- **Acton:** General-purpose agents
- **Haystack:** Document-focused pipelines

## See Also

- [Core Concepts](core-concepts.md) - Detailed explanations
- [API Reference](api-reference.md) - Complete API
- [Advanced Topics](advanced-topics.md) - Production patterns
