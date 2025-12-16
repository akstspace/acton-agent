"""
Test configuration and fixtures for pytest.
"""

from typing import List

import pytest

from toolio_agent.agent.models import Message


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: List[str] = None):
        """
        Create a MockLLMClient configured with an optional sequence of preset responses.
        
        Parameters:
            responses (List[str], optional): Ordered list of response strings to return for successive `call`/`call_stream` invocations. If omitted or exhausted, the client will produce a default mock response when called.
        
        Notes:
            Initializes `call_count` to 0 and `calls` to an empty list to record invocation history for tests.
        """
        self.responses = responses or []
        self.call_count = 0
        self.calls = []  # Store all calls for inspection

    def call(self, messages: List[Message], **kwargs) -> str:
        """
        Record the invocation and return the next mock response.
        
        Appends a record with the provided messages and keyword arguments to the client's internal call log. If a predefined response is available it is returned (and the internal call counter is advanced); otherwise a default JSON-formatted final answer wrapped in a code fence is returned.
        
        Parameters:
            messages (List[Message]): Messages sent to the mock client; recorded for inspection.
            **kwargs: Additional call options; recorded for inspection.
        
        Returns:
            str: The next predefined response string if available, otherwise a default JSON code-fenced final answer.
        """
        self.calls.append({"messages": messages, "kwargs": kwargs})

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default response if no more predefined responses
        return '```json\n{"final_answer": "Mock response"}\n```'

    def call_stream(self, messages: List[Message], **kwargs):
        """
        Simulate streaming output from the LLM by yielding the response one character at a time.
        
        Parameters:
            messages (List[Message]): The message sequence sent to the client; used to produce the mock response.
        
        Returns:
            iterator: Yields each character of the resulting response string as a one-character string.
        """
        response = self.call(messages, **kwargs)
        # Yield character by character
        for char in response:
            yield char


@pytest.fixture
def mock_llm_client():
    """
    Provide a MockLLMClient instance for tests.
    
    Returns:
        MockLLMClient: A mock LLM client initialized with no preset responses; it records calls and can simulate streaming.
    """
    return MockLLMClient()


@pytest.fixture
def mock_llm_client_with_responses():
    """Fixture factory for mock LLM client with custom responses."""

    def _create_client(responses: List[str]):
        """
        Create a MockLLMClient preloaded with the given responses.
        
        Each invocation of the client's call() will return the next string from the provided list; when the list is exhausted the client falls back to its default mock response.
        
        Parameters:
            responses (List[str]): Ordered list of strings the mock client will return for successive calls.
        
        Returns:
            MockLLMClient: A mock LLM client configured to return the provided responses in sequence.
        """
        return MockLLMClient(responses=responses)

    return _create_client


@pytest.fixture
def sample_messages():
    """
    Provide a small sequence of Message objects representing a typical assistant interaction.
    
    Returns:
        List[Message]: Three messages in order — a system prompt, a user message, and an assistant reply.
    """
    return [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]


@pytest.fixture
def tool_call_response():
    """
    Provides a sample tool call response wrapped in a JSON code fence.
    
    Returns:
        str: A string containing a JSON-formatted tool call plan (includes a thought and a list of tool call entries) wrapped in triple-backtick ```json``` fencing.
    """
    return """```json
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
```"""


@pytest.fixture
def final_answer_response():
    """
    Sample LLM final answer response formatted as a JSON code block.
    
    Returns:
        str: A string containing a JSON object with keys `thought` and `final_answer`, wrapped in a ```json code fence.
    """
    return """```json
{
  "thought": "I have completed the calculation",
  "final_answer": "The sum is 8"
}
```"""


@pytest.fixture
def plan_response():
    """
    Provide a JSON-formatted plan containing a "thought" and a "plan" array wrapped in a Markdown ```json fenced code block.
    
    Returns:
        str: The plan response as a string containing a JSON object with a "thought" field and a "plan" array, enclosed in triple-backtick ```json fences.
    """
    return """```json
{
  "thought": "Let me plan how to solve this",
  "plan": [
    "First, I will search for information",
    "Then, I will analyze the results",
    "Finally, I will provide the answer"
  ]
}
```"""