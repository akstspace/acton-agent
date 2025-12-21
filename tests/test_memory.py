"""
Test suite for the memory management system.
"""

import pytest

from acton_agent.agent import Message
from acton_agent.memory import AgentMemory, SimpleAgentMemory


class TestAgentMemory:
    """Test the abstract AgentMemory base class."""

    def test_agent_memory_is_abstract(self):
        """Verify AgentMemory cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentMemory()


class TestSimpleAgentMemory:
    """Test the SimpleAgentMemory implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        memory = SimpleAgentMemory()
        assert memory.max_history_tokens == 8000

    def test_initialization_custom(self):
        """Test custom token limit."""
        memory = SimpleAgentMemory(max_history_tokens=4000)
        assert memory.max_history_tokens == 4000

    def test_count_tokens(self):
        """Test token counting heuristic."""
        memory = SimpleAgentMemory()

        # 4 characters = 1 token
        assert memory._count_tokens("test") == 1
        assert memory._count_tokens("test test") == 2
        assert memory._count_tokens("a" * 100) == 25

    def test_manage_history_empty(self):
        """Test with empty history."""
        memory = SimpleAgentMemory()
        result = memory.manage_history([])
        assert result == []

    def test_manage_history_under_limit(self):
        """Test when history is under token limit."""
        memory = SimpleAgentMemory(max_history_tokens=1000)
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        result = memory.manage_history(history)
        assert len(result) == 2
        assert result == history

    def test_manage_history_truncates(self):
        """Test that old messages are truncated."""
        memory = SimpleAgentMemory(max_history_tokens=100)

        # Create messages that exceed token limit
        history = [
            Message(role="user", content="a" * 200),  # ~50 tokens
            Message(role="assistant", content="b" * 200),  # ~50 tokens
            Message(role="user", content="c" * 200),  # ~50 tokens
            Message(role="assistant", content="d" * 200),  # ~50 tokens
        ]

        result = memory.manage_history(history)

        # Should keep only the last 2 messages
        assert len(result) < len(history)
        assert len(result) >= 2  # Always keeps at least 2

    def test_manage_history_keeps_minimum(self):
        """Test that at least 2 messages are kept."""
        memory = SimpleAgentMemory(max_history_tokens=10)  # Very small limit

        history = [
            Message(role="user", content="a" * 1000),
            Message(role="assistant", content="b" * 1000),
            Message(role="user", content="c" * 1000),
        ]

        result = memory.manage_history(history)

        # Should keep at least 2 messages even if over limit
        assert len(result) >= 2

    def test_manage_history_does_not_modify_original(self):
        """Test that original history is not modified."""
        memory = SimpleAgentMemory(max_history_tokens=50)

        original_history = [
            Message(role="user", content="a" * 200),
            Message(role="assistant", content="b" * 200),
            Message(role="user", content="c" * 200),
        ]

        original_length = len(original_history)
        memory.manage_history(original_history)

        # Original should not be modified
        assert len(original_history) == original_length

    def test_manage_history_truncates_content_when_two_messages_exceed_limit(self):
        """Test that message content is truncated when even 2 messages exceed limit."""
        memory = SimpleAgentMemory(max_history_tokens=50)

        # Create 2 messages that together exceed the limit significantly
        history = [
            Message(role="user", content="a" * 400),  # ~100 tokens
            Message(role="assistant", content="b" * 400),  # ~100 tokens
        ]

        result = memory.manage_history(history)

        # Should still have 2 messages
        assert len(result) == 2

        # But their content should be truncated
        total_tokens = sum(memory._count_tokens(msg.content) for msg in result)
        assert total_tokens <= memory.max_history_tokens

        # Messages should have truncation indicator
        assert result[0].content.startswith("...")
        assert result[1].content.startswith("...")

    def test_manage_history_truncates_single_massive_message(self):
        """Test truncation of a single very large message."""
        memory = SimpleAgentMemory(max_history_tokens=100)

        history = [
            Message(role="user", content="x" * 2000),  # ~500 tokens - way over limit
        ]

        result = memory.manage_history(history)

        # Should keep the message but truncate it
        assert len(result) == 1
        assert result[0].content.startswith("...")
        assert len(result[0].content) < len(history[0].content)

        # Should be under limit
        total_tokens = memory._count_tokens(result[0].content)
        assert total_tokens <= memory.max_history_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
