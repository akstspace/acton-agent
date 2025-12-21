"""
Tests for the client module.
"""

from acton_agent.agent.client import LLMClient
from acton_agent.agent.models import Message


class TestLLMClientProtocol:
    """Tests for LLMClient protocol."""

    def test_protocol_exists(self):
        """Test that LLMClient protocol is defined."""
        assert LLMClient is not None

    def test_protocol_has_call_method(self):
        """Test that protocol requires call method."""
        # Check that the protocol has the call method defined
        assert hasattr(LLMClient, "call")

    def test_mock_client_satisfies_protocol(self):
        """Test that a simple implementation satisfies the protocol."""

        class SimpleClient:
            def call(self, messages: list[Message], **kwargs) -> str:
                """
                Provide a fixed response for the given messages.

                Parameters:
                        messages (List[Message]): Messages to be processed by the client.
                        **kwargs: Additional keyword arguments accepted but ignored.

                Returns:
                        response (str): The constant string "response".
                """
                return "response"

        client = SimpleClient()

        # Should be able to pass messages
        messages = [Message(role="user", content="test")]
        result = client.call(messages)
        assert result == "response"


class TestClientImplementation:
    """Tests for client implementations."""

    def test_client_with_custom_implementation(self):
        """Test using a custom client implementation."""

        class CustomClient:
            def __init__(self):
                """
                Initialize the client instance and set its internal call counter to zero.
                """
                self.call_count = 0

            def call(self, messages: list[Message], **kwargs) -> str:
                """
                Increment the client's internal call counter and produce a response including the updated count.

                Parameters:
                    messages (List[Message]): Messages passed to the client (not used by this implementation).
                    **kwargs: Additional keyword arguments (ignored).

                Returns:
                    str: Response formatted as "Response #<n>" where <n> is the updated call count.
                """
                self.call_count += 1
                return f"Response #{self.call_count}"

        client = CustomClient()

        messages = [Message(role="user", content="Hello")]

        result1 = client.call(messages)
        assert result1 == "Response #1"
        assert client.call_count == 1

        result2 = client.call(messages)
        assert result2 == "Response #2"
        assert client.call_count == 2
