"""
Protocols for the AI Agent Framework.

This module defines protocols (interfaces) that external components
must implement to work with the agent framework.
"""

from typing import Protocol

from ..agent.models import Message


class LLMClient(Protocol):
    """
    Protocol for LLM client implementations.

    Any LLM client (OpenAI, Anthropic, local models, etc.) must implement
    this protocol to work with the agent framework.
    """

    def call(self, messages: list[Message], **kwargs) -> str:
        """
        Invoke the language model with a sequence of conversation messages and return its reply.

        Parameters:
            messages (List[Message]): Conversation messages representing the prompt and context.
            **kwargs: Additional, implementation-specific parameters forwarded to the LLM.

        Returns:
            str: The LLM's response as a string.

        Raises:
            Exception: If the LLM invocation fails.
        """
        ...
