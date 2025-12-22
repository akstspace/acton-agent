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
from .tools import ConfigSchema, FunctionTool, Tool, ToolInputSchema, ToolRegistry, ToolSet


__all__ = [
    "Agent",
    "AgentMemory",
    "ConfigSchema",
    "FunctionTool",
    "LLMClient",
    "OpenAIClient",
    "OpenRouterClient",
    "ResponseParser",
    "SimpleAgentMemory",
    "Tool",
    "ToolInputSchema",
    "ToolRegistry",
    "ToolSet",
    "parse_streaming_events",
]
