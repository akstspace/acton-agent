"""
Tests for the core Agent class.
"""

import pytest

from acton_agent.agent.agent import Agent
from acton_agent.agent.exceptions import MaxIterationsError
from acton_agent.agent.models import Message, ToolCall
from acton_agent.agent.retry import RetryConfig
from acton_agent.agent.tools import FunctionTool, Tool


class SimpleCalculatorTool(Tool):
    """Simple calculator tool for testing."""

    def __init__(self):
        """
        Initialize the SimpleCalculatorTool with its registered name and description.

        Sets the tool's name to "calculator" and its description to "Perform basic arithmetic operations".
        """
        super().__init__(name="calculator", description="Perform basic arithmetic operations")

    def execute(self, parameters: dict) -> str:
        """
        Perform a basic arithmetic operation specified by the input parameters.

        Parameters:
            parameters (dict): Mapping with keys:
                - "a" (int|float): First operand (defaults to 0).
                - "b" (int|float): Second operand (defaults to 0).
                - "operation" (str): One of "add", "subtract", "multiply", or "divide" (defaults to "add").

        Returns:
            str: The numeric result converted to a string for successful operations, or an error message
            such as "Error: Division by zero" or "Error: Unknown operation {operation}".
        """
        a = parameters.get("a", 0)
        b = parameters.get("b", 0)
        operation = parameters.get("operation", "add")

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        else:
            return f"Error: Unknown operation {operation}"

        return str(result)

    def get_schema(self) -> dict:
        """
        JSON schema describing the tool's input parameters.

        Returns:
            dict: A JSON Schema object with properties:
                - a (number): First operand (required).
                - b (number): Second operand (required).
                - operation (string): Arithmetic operation to perform; one of "add", "subtract", "multiply", "divide".
        """
        return {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
            },
            "required": ["a", "b"],
        }


class TestAgentInitialization:
    """Tests for Agent initialization."""

    def test_agent_creation(self, mock_llm_client):
        """Test creating an agent."""
        agent = Agent(llm_client=mock_llm_client)
        assert agent is not None
        assert agent.llm_client == mock_llm_client
        assert agent.max_iterations == 10
        assert not agent.stream

    def test_agent_with_custom_config(self, mock_llm_client):
        """Test creating agent with custom configuration."""
        retry_config = RetryConfig(max_attempts=5, wait_min=0.5)
        agent = Agent(
            llm_client=mock_llm_client,
            system_prompt="Custom prompt",
            max_iterations=5,
            retry_config=retry_config,
            stream=True,
        )

        assert agent.max_iterations == 5
        assert agent.retry_config.max_attempts == 5
        assert agent.stream is True
        assert agent.custom_instructions == "Custom prompt"


class TestToolManagement:
    """Tests for tool registration and management."""

    def test_register_tool(self, mock_llm_client):
        """Test registering a tool."""
        agent = Agent(llm_client=mock_llm_client)
        tool = SimpleCalculatorTool()

        agent.register_tool(tool)
        assert "calculator" in agent.list_tools()

    def test_register_multiple_tools(self, mock_llm_client):
        """Test registering multiple tools."""
        agent = Agent(llm_client=mock_llm_client)

        tool1 = SimpleCalculatorTool()
        tool2 = FunctionTool(
            name="greeter",
            description="Greet someone",
            func=lambda name: f"Hello, {name}!",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Name to greet"}},
                "required": ["name"],
            },
        )

        agent.register_tool(tool1)
        agent.register_tool(tool2)

        tools = agent.list_tools()
        assert "calculator" in tools
        assert "greeter" in tools

    def test_unregister_tool(self, mock_llm_client):
        """Test unregistering a tool."""
        agent = Agent(llm_client=mock_llm_client)
        tool = SimpleCalculatorTool()

        agent.register_tool(tool)
        assert "calculator" in agent.list_tools()

        agent.unregister_tool("calculator")
        assert "calculator" not in agent.list_tools()


class TestAgentRun:
    """Tests for agent run functionality."""

    def test_run_with_final_answer(self, mock_llm_client_with_responses):
        """Test agent run that returns final answer directly."""
        response = """```json
{
  "final_answer": "The answer is 42"
}
```"""

        client = mock_llm_client_with_responses([response])
        agent = Agent(llm_client=client)

        result = agent.run("What is the answer?")
        assert result == "The answer is 42"

    def test_run_with_tool_call(self, mock_llm_client_with_responses):
        """Test agent run with tool execution."""
        tool_response = """```json
{
  "thought": "I need to calculate",
  "tool_calls": [
    {
      "id": "call_1",
      "tool_name": "calculator",
      "parameters": {"a": 5, "b": 3, "operation": "add"}
    }
  ]
}
```"""

        final_response = """```json
{
  "final_answer": "The sum of 5 and 3 is 8"
}
```"""

        client = mock_llm_client_with_responses([tool_response, final_response])
        agent = Agent(llm_client=client)
        agent.register_tool(SimpleCalculatorTool())

        result = agent.run("What is 5 + 3?")
        assert "8" in result
        assert client.call_count == 2

    def test_run_with_plan(self, mock_llm_client_with_responses):
        """Test agent run with planning step."""
        plan_response = """```json
{
  "plan": "Step 1: Calculate\\nStep 2: Return answer"
}
```"""

        tool_response = """```json
{
  "tool_thought": "Executing step 1",
  "tool_calls": [
    {
      "id": "call_1",
      "tool_name": "calculator",
      "parameters": {"a": 10, "b": 5, "operation": "subtract"}
    }
  ]
}
```"""

        final_response = """```json
{
  "final_answer": "The result is 5"
}
```"""

        client = mock_llm_client_with_responses([plan_response, tool_response, final_response])
        agent = Agent(llm_client=client)
        agent.register_tool(SimpleCalculatorTool())

        result = agent.run("What is 10 - 5?")
        assert "5" in result
        assert client.call_count == 3

    def test_run_max_iterations(self, mock_llm_client_with_responses):
        """Test that agent respects max_iterations."""
        # Create response that always requests tools (infinite loop)
        tool_response = """```json
{
  "thought": "Need more calculation",
  "tool_calls": [
    {
      "id": "call_1",
      "tool_name": "calculator",
      "parameters": {"a": 1, "b": 1, "operation": "add"}
    }
  ]
}
```"""

        # Provide enough responses to hit max_iterations
        client = mock_llm_client_with_responses([tool_response] * 20)
        agent = Agent(llm_client=client, max_iterations=3)
        agent.register_tool(SimpleCalculatorTool())

        with pytest.raises(MaxIterationsError):
            agent.run("Keep calculating")


class TestToolExecution:
    """Tests for tool execution logic."""

    def test_execute_tool_success(self, mock_llm_client):
        """Test successful tool execution."""
        agent = Agent(llm_client=mock_llm_client)
        tool = SimpleCalculatorTool()
        agent.register_tool(tool)

        tool_calls = [
            ToolCall(
                id="call_1",
                tool_name="calculator",
                parameters={"a": 7, "b": 3, "operation": "multiply"},
            )
        ]

        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert results[0].success
        assert results[0].result == "21"

    def test_execute_nonexistent_tool(self, mock_llm_client):
        """Test executing a tool that doesn't exist."""
        agent = Agent(llm_client=mock_llm_client)

        tool_calls = [ToolCall(id="call_1", tool_name="nonexistent", parameters={})]

        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert not results[0].success
        assert "not found" in results[0].error

    def test_execute_tool_with_error(self, mock_llm_client):
        """Test executing a tool that returns an error."""
        agent = Agent(llm_client=mock_llm_client)
        tool = SimpleCalculatorTool()
        agent.register_tool(tool)

        tool_calls = [
            ToolCall(
                id="call_1",
                tool_name="calculator",
                parameters={"a": 10, "b": 0, "operation": "divide"},
            )
        ]

        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert not results[0].success
        assert "Division by zero" in results[0].error


class TestConversationHistory:
    """Tests for conversation history management."""

    def test_conversation_history_tracking(self, mock_llm_client_with_responses):
        """Test that conversation history is tracked."""
        response = """```json
{
  "final_answer": "Done"
}
```"""

        client = mock_llm_client_with_responses([response])
        agent = Agent(llm_client=client)

        agent.run("Test query")

        # Check that user message was added
        assert len(agent.conversation_history) > 0
        assert any(msg.role == "user" for msg in agent.conversation_history)

    def test_clear_conversation_history(self, mock_llm_client):
        """
        Verifies that the agent can clear its conversation history.

        Adds messages to the agent's conversation_history, confirms it is non-empty, calls clear(), and asserts the history is empty.
        """
        agent = Agent(llm_client=mock_llm_client)

        # Manually add some history
        agent.conversation_history.append(Message(role="user", content="Test"))
        agent.conversation_history.append(Message(role="assistant", content="Response"))

        assert len(agent.conversation_history) > 0

        # The agent should have a way to clear history
        agent.conversation_history.clear()
        assert len(agent.conversation_history) == 0

    def test_add_message_user(self, mock_llm_client):
        """Test adding a user message to conversation history."""
        agent = Agent(llm_client=mock_llm_client)

        agent.add_message("user", "Hello, how are you?")

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].role == "user"
        assert agent.conversation_history[0].content == "Hello, how are you?"

    def test_add_message_assistant(self, mock_llm_client):
        """Test adding an assistant message to conversation history."""
        agent = Agent(llm_client=mock_llm_client)

        agent.add_message("assistant", "I'm doing well, thank you!")

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].role == "assistant"
        assert agent.conversation_history[0].content == "I'm doing well, thank you!"

    def test_add_message_system(self, mock_llm_client):
        """Test adding a system message to conversation history."""
        agent = Agent(llm_client=mock_llm_client)

        agent.add_message("system", "This is a system message")

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].role == "system"
        assert agent.conversation_history[0].content == "This is a system message"

    def test_add_multiple_messages(self, mock_llm_client):
        """Test adding multiple messages to conversation history."""
        agent = Agent(llm_client=mock_llm_client)

        agent.add_message("user", "First message")
        agent.add_message("assistant", "Second message")
        agent.add_message("user", "Third message")

        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[0].content == "First message"
        assert agent.conversation_history[1].content == "Second message"
        assert agent.conversation_history[2].content == "Third message"

    def test_add_message_preserves_order(self, mock_llm_client):
        """Test that messages are added in the correct order."""
        agent = Agent(llm_client=mock_llm_client)

        messages = [
            ("user", "Message 1"),
            ("assistant", "Message 2"),
            ("user", "Message 3"),
            ("assistant", "Message 4"),
        ]

        for role, content in messages:
            agent.add_message(role, content)

        assert len(agent.conversation_history) == 4
        for i, (role, content) in enumerate(messages):
            assert agent.conversation_history[i].role == role
            assert agent.conversation_history[i].content == content
