"""
Memory management system for the AI Agent Framework.

This module provides the abstract AgentMemory base class and built-in implementations
for managing conversation history and token limits.
"""

from abc import ABC, abstractmethod

from loguru import logger

from .models import Message


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


class SimpleAgentMemory(AgentMemory):
    """
    Simple token-based memory management that truncates old messages.

    This implementation uses a character-based heuristic (~4 chars per token)
    to estimate tokens and removes the oldest messages when the limit is exceeded,
    while preserving at least the 2 most recent messages.
    """

    def __init__(self, max_history_tokens: int = 8000):
        """
        Initialize SimpleAgentMemory with a token limit.

        Parameters:
            max_history_tokens (int): Maximum token limit for conversation history. Defaults to 8000.
        """
        self.max_history_tokens = max_history_tokens

    def _count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Uses a simple heuristic of ~4 characters per token, which is a reasonable
        approximation for most English text with common tokenizers.

        Parameters:
            text (str): The text to count tokens for.

        Returns:
            int: Estimated token count.
        """
        return len(text) // 4

    def manage_history(self, history: list[Message]) -> list[Message]:
        """
        Truncate conversation history to stay within max_history_tokens limit.

        Removes oldest messages while preserving recent conversation context.
        Always keeps at least the most recent user-assistant exchange.

        Parameters:
            history (List[Message]): Current conversation history.

        Returns:
            List[Message]: Truncated conversation history.
        """
        if not history:
            return history

        managed_history = history.copy()
        total_tokens = sum(self._count_tokens(msg.content) for msg in managed_history)
        if total_tokens <= self.max_history_tokens:
            return managed_history

        logger.info(f"Conversation history exceeds {self.max_history_tokens} tokens ({total_tokens}). Truncating...")

        # Remove oldest messages until we're under the limit
        # Keep at least the last 2 messages (most recent exchange)
        while len(managed_history) > 2 and total_tokens > self.max_history_tokens:
            removed_msg = managed_history.pop(0)
            total_tokens -= self._count_tokens(removed_msg.content)

        # If still over limit, truncate message content
        if total_tokens > self.max_history_tokens and len(managed_history) > 0:
            logger.info("Messages still exceed limit. Truncating message content...")

            # Calculate how many tokens to keep per message (divide limit by number of messages)
            tokens_per_message = self.max_history_tokens // len(managed_history)

            # Truncate each message content to fit within the limit
            for i, msg in enumerate(managed_history):
                msg_tokens = self._count_tokens(msg.content)
                if msg_tokens > tokens_per_message:
                    # Keep the most recent content (end of the message)
                    chars_to_keep = tokens_per_message * 4  # Convert tokens back to chars
                    if chars_to_keep > 0:
                        truncated_content = "..." + msg.content[-chars_to_keep:]
                        managed_history[i] = Message(role=msg.role, content=truncated_content)

            # Recalculate total tokens
            total_tokens = sum(self._count_tokens(msg.content) for msg in managed_history)

        logger.info(f"Truncated to {len(managed_history)} messages ({total_tokens} tokens)")

        return managed_history
