"""
Memory management for the AI Agent Framework.

This package provides memory management interfaces and implementations
for conversation history tracking.
"""

from .base import AgentMemory
from .simple import SimpleAgentMemory


__all__ = ["AgentMemory", "SimpleAgentMemory"]
