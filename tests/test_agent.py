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
        Initialize the SimpleCalculatorTool and set its registered name and description.

        Sets the tool's name to "calculator" and its description to "Perform basic arithmetic operations".
        """
        super().__init__(
            name="calculator", description="Perform basic arithmetic operations"
        )

    def execute(self, parameters: dict) -> str:
        """
        Perform a basic arithmetic operation described by the provided parameters.

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
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
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
  "thought": "Let me plan the solution",
  "plan": ["Step 1: Calculate", "Step 2: Return answer"]
}
```"""

        tool_response = """```json
{
  "thought": "Executing step 1",
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

        client = mock_llm_client_with_responses(
            [plan_response, tool_response, final_response]
        )
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
        """Test clearing conversation history."""
        agent = Agent(llm_client=mock_llm_client)

        # Manually add some history
        agent.conversation_history.append(Message(role="user", content="Test"))
        agent.conversation_history.append(Message(role="assistant", content="Response"))

        assert len(agent.conversation_history) > 0

        # The agent should have a way to clear history
        agent.conversation_history.clear()
        assert len(agent.conversation_history) == 0
