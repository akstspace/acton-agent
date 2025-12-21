"""
Abstract base class for memory management in the AI Agent Framework.
"""

from abc import ABC, abstractmethod

from ..agent.models import Message


class AgentMemory(ABC):
    """
    Abstract base class for agent memory management.

    Memory tools control how conversation history is managed, including
    truncation, summarization, or other strategies to stay within token limits.
    """

    @abstractmethod
    def manage_history(self, history: list[Message]) -> list[Message]:
        """
        Manage the agent's conversation history according to a memory strategy.
        
        This method returns a modified list of Message objects representing the managed conversation history (for example, truncated or summarized to satisfy token or storage constraints).
        
        Returns:
            list[Message]: Managed conversation history (may be truncated, summarized, or otherwise modified).
        """