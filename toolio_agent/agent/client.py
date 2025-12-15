"""
Protocols for the AI Agent Framework.

This module defines protocols (interfaces) that external components
must implement to work with the agent framework.
"""

from typing import Protocol, List
from .models import Message


class LLMClient(Protocol):
    """
    Protocol for LLM client implementations.
    
    Any LLM client (OpenAI, Anthropic, local models, etc.) must implement
    this protocol to work with the agent framework.
    """
    
    def call(self, messages: List[Message], **kwargs) -> str:
        """
        Make a synchronous call to the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional LLM-specific parameters
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If the LLM call fails
        """
        ...
