"""
Test configuration and fixtures for pytest.
"""

import pytest
from typing import List
from toolio_agent.agent.models import Message


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: List[str] = None):
        """
        Initialize mock client.
        
        Args:
            responses: List of responses to return in sequence
        """
        self.responses = responses or []
        self.call_count = 0
        self.calls = []  # Store all calls for inspection
    
    def call(self, messages: List[Message], **kwargs) -> str:
        """Mock call method."""
        self.calls.append({"messages": messages, "kwargs": kwargs})
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        
        # Default response if no more predefined responses
        return '```json\n{"final_answer": "Mock response"}\n```'
    
    def call_stream(self, messages: List[Message], **kwargs):
        """Mock streaming call method."""
        response = self.call(messages, **kwargs)
        # Yield character by character
        for char in response:
            yield char


@pytest.fixture
def mock_llm_client():
    """Fixture providing a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_client_with_responses():
    """Fixture factory for mock LLM client with custom responses."""
    def _create_client(responses: List[str]):
        return MockLLMClient(responses=responses)
    return _create_client


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages."""
    return [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]


@pytest.fixture
def tool_call_response():
    """Fixture providing a tool call response."""
    return '''```json
{
  "thought": "I need to calculate the sum",
  "tool_calls": [
    {
      "id": "call_1",
      "tool_name": "calculator",
      "parameters": {"a": 5, "b": 3}
    }
  ]
}
```'''


@pytest.fixture
def final_answer_response():
    """Fixture providing a final answer response."""
    return '''```json
{
  "thought": "I have completed the calculation",
  "final_answer": "The sum is 8"
}
```'''


@pytest.fixture
def plan_response():
    """Fixture providing a plan response."""
    return '''```json
{
  "thought": "Let me plan how to solve this",
  "plan": [
    "First, I will search for information",
    "Then, I will analyze the results",
    "Finally, I will provide the answer"
  ]
}
```'''
