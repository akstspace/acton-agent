"""
AI Agent subpackage - Production-ready LLM Agent Framework
"""

from .agent import Agent
from .client import LLMClient
from .exceptions import (
    AgentError,
    InvalidToolSchemaError,
    LLMCallError,
    MaxIterationsError,
    ResponseParseError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .models import (
    AgentFinalResponse,
    AgentFinalResponseEvent,
    AgentPlan,
    AgentPlanEvent,
    AgentResponse,
    AgentStep,
    AgentStepEvent,
    AgentStepUpdate,
    AgentStreamEnd,
    AgentStreamStart,
    AgentThought,
    AgentToken,
    AgentToolResultsEvent,
    Message,
    StreamingEvent,
    ToolCall,
    ToolResult,
)
from .parser import ResponseParser
from .retry import RetryConfig
from .tools import FunctionTool, Tool, ToolRegistry

__all__ = [
    "Agent",
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentResponse",
    "AgentThought",
    "AgentPlan",
    "AgentStep",
    "AgentFinalResponse",
    "Tool",
    "ToolRegistry",
    "FunctionTool",
    "RetryConfig",
    "LLMClient",
    "ResponseParser",
    "AgentError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "LLMCallError",
    "ResponseParseError",
    "MaxIterationsError",
    "InvalidToolSchemaError",
    # Streaming events
    "StreamingEvent",
    "AgentStreamStart",
    "AgentToken",
    "AgentStreamEnd",
    "AgentStepUpdate",
    "AgentToolResultsEvent",
    "AgentPlanEvent",
    "AgentStepEvent",
    "AgentFinalResponseEvent",
]
