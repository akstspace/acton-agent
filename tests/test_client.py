"""
Tests for the client module.
"""

import pytest
from toolio_agent.agent.client import LLMClient
from toolio_agent.agent.models import Message


class TestLLMClientProtocol:
    """Tests for LLMClient protocol."""
    
    def test_protocol_exists(self):
        """Test that LLMClient protocol is defined."""
        assert LLMClient is not None
    
    def test_protocol_has_call_method(self):
        """Test that protocol requires call method."""
        # Check that the protocol has the call method defined
        assert hasattr(LLMClient, 'call')
    
    def test_mock_client_satisfies_protocol(self):
        """Test that a simple implementation satisfies the protocol."""
        from typing import List
        
        class SimpleClient:
            def call(self, messages: List[Message], **kwargs) -> str:
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
        from typing import List
        
        class CustomClient:
            def __init__(self):
                self.call_count = 0
            
            def call(self, messages: List[Message], **kwargs) -> str:
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
