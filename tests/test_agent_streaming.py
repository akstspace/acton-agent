"""
Tests for agent streaming with structured events.
"""

from acton_agent.agent.agent import Agent
from acton_agent.agent.models import (
    AgentFinalResponseEvent,
    AgentPlanEvent,
    AgentStepEvent,
    AgentStepUpdate,
    AgentStreamEnd,
    AgentStreamStart,
    AgentToken,
    AgentToolResultsEvent,
)
from acton_agent.tools import Tool


class SimpleCalculatorTool(Tool):
    """Simple calculator tool for testing."""

    def __init__(self):
        """
        Create a SimpleCalculatorTool named "calculator" that performs basic arithmetic operations.
        """
        super().__init__(name="calculator", description="Perform basic arithmetic operations")

    def execute(self, parameters: dict, toolset_params: dict | None = None) -> str:
        """
        Perform a basic arithmetic operation using values from `parameters`.
        
        Parameters:
            parameters (dict): Input values for the operation. Expected keys:
                - "a" (number): first operand (default 0).
                - "b" (number): second operand (default 0).
                - "operation" (str): one of "add", "subtract", "multiply", "divide" (default "add").
            toolset_params (dict | None): Optional additional context; not used by this tool.
        
        Returns:
            str: The numeric result converted to a string, or an error message:
                 "Error: Division by zero" or "Error: Unknown operation {operation}".
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
        """Get the tool schema."""
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


class TestAgentStreamingEvents:
    """Tests for agent streaming with structured events."""

    def test_run_stream_yields_final_response_event(self, mock_llm_client_with_responses):
        """Test that run_stream yields AgentFinalResponseEvent."""
        response = """```json
{
  "final_answer": "The answer is 42"
}
```"""

        client = mock_llm_client_with_responses([response])
        agent = Agent(llm_client=client)

        events = list(agent.run_stream("What is the answer?"))

        # Should have at least one AgentFinalResponseEvent
        final_events = [e for e in events if isinstance(e, AgentFinalResponseEvent)]
        assert len(final_events) == 1
        assert final_events[0].response.final_answer == "The answer is 42"

    def test_run_stream_yields_plan_event(self, mock_llm_client_with_responses):
        """Test that run_stream yields AgentPlanEvent."""
        plan_response = """```json
{
  "plan": "Step 1: Calculate\\nStep 2: Return answer"
}
```"""

        final_response = """```json
{
  "final_answer": "Done"
}
```"""

        client = mock_llm_client_with_responses([plan_response, final_response])
        agent = Agent(llm_client=client)

        events = list(agent.run_stream("What is the plan?"))

        # Should have AgentPlanEvent
        plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
        assert len(plan_events) == 1
        assert "Step 1" in plan_events[0].plan.plan

    def test_run_stream_yields_step_event_and_tool_results(self, mock_llm_client_with_responses):
        """Test that run_stream yields AgentStepEvent and AgentToolResultsEvent."""
        step_response = """```json
{
  "thought": "Let me calculate",
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
  "final_answer": "The result is 8"
}
```"""

        client = mock_llm_client_with_responses([step_response, final_response])
        agent = Agent(llm_client=client)
        agent.register_tool(SimpleCalculatorTool())

        events = list(agent.run_stream("What is 5 + 3?"))

        # Should have AgentStepEvent
        step_events = [e for e in events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        assert len(step_events[0].step.tool_calls) == 1

        # Should have AgentToolResultsEvent
        tool_result_events = [e for e in events if isinstance(e, AgentToolResultsEvent)]
        assert len(tool_result_events) == 1
        assert len(tool_result_events[0].results) == 1
        assert tool_result_events[0].results[0].result == "8"

    def test_run_stream_with_streaming_enabled(self, mock_streaming_llm_client):
        """Test that run_stream yields streaming events when streaming is enabled."""
        response = """```json
{
  "final_answer": "Hello"
}
```"""

        client = mock_streaming_llm_client([response])
        agent = Agent(llm_client=client, stream=True)

        events = list(agent.run_stream("Say hello"))

        # Should have streaming events
        stream_start_events = [e for e in events if isinstance(e, AgentStreamStart)]
        assert len(stream_start_events) == 1

        token_events = [e for e in events if isinstance(e, AgentToken)]
        assert len(token_events) > 0

        stream_end_events = [e for e in events if isinstance(e, AgentStreamEnd)]
        assert len(stream_end_events) == 1

        # Should still have final response event
        final_events = [e for e in events if isinstance(e, AgentFinalResponseEvent)]
        assert len(final_events) == 1

    def test_run_stream_no_dict_events(self, mock_llm_client_with_responses):
        """Test that run_stream does not yield dict events anymore."""
        response = """```json
{
  "final_answer": "Done"
}
```"""

        client = mock_llm_client_with_responses([response])
        agent = Agent(llm_client=client)

        events = list(agent.run_stream("Test"))

        # Should not have any dict events
        dict_events = [e for e in events if isinstance(e, dict)]
        assert len(dict_events) == 0

        # All events should be instances of the streaming event models
        valid_event_types = (
            AgentStreamStart,
            AgentToken,
            AgentStreamEnd,
            AgentStepUpdate,
            AgentToolResultsEvent,
            AgentPlanEvent,
            AgentStepEvent,
            AgentFinalResponseEvent,
        )

        for event in events:
            assert isinstance(event, valid_event_types), f"Invalid event type: {type(event)}"