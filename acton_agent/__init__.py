"""
Acton Agent - Experimental LLM Agent Framework

A flexible framework for building AI agents with:
- Flexible tool system for extending agent capabilities
- Built-in retry logic and error handling
- Conversation memory management
- Support for multiple LLM providers
- Streaming response support

Note: This is an experimental project. The API may change without notice.
"""

from .agent import Agent
from .client import LLMClient, OpenAIClient, OpenRouterClient
from .memory import AgentMemory, SimpleAgentMemory
from .parsers import ResponseParser, parse_streaming_events
from .tools import FunctionTool, RequestsTool, Tool, ToolRegistry, ToolSet, create_api_tool


__all__ = [
    "Agent",
    "AgentMemory",
    "FunctionTool",
    "LLMClient",
    "OpenAIClient",
    "OpenRouterClient",
    "RequestsTool",
    "ResponseParser",
    "SimpleAgentMemory",
    "Tool",
    "ToolRegistry",
    "ToolSet",
    "create_api_tool",
    "parse_streaming_events",
]
