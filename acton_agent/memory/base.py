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
        Process and potentially modify conversation history to manage memory.

        Parameters:
            history (List[Message]): Current conversation history to manage.

        Returns:
            List[Message]: Managed conversation history (may be truncated, summarized, etc.).
        """
