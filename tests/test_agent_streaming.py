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
from acton_agent.parsers.streaming_util import AgentAnswer
from acton_agent.tools import Tool


class SimpleCalculatorTool(Tool):
    """Simple calculator tool for testing."""

    def __init__(self):
        """
        Create a SimpleCalculatorTool named "calculator" that performs basic arithmetic operations.
        """
        super().__init__(name="calculator", description="Perform basic arithmetic operations")

    def execute(self, parameters: dict, config: dict | None = None) -> str:
        """
        Perform a basic arithmetic operation using values from `parameters`.

        Parameters:
            parameters (dict): Input values for the operation. Expected keys:
                - "a" (number): first operand (default 0).
                - "b" (number): second operand (default 0).
                - "operation" (str): one of "add", "subtract", "multiply", "divide" (default "add").
            config (dict | None): Optional additional context; not used by this tool.

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

    def test_stream_state_yields_agent_answer(self, mock_llm_client_with_responses):
        """Test that stream_state yields AgentAnswer objects."""
        response = """```json
{
  "final_answer": "The answer is 42"
}
```"""

        client = mock_llm_client_with_responses([response])
        agent = Agent(llm_client=client)

        states = list(agent.stream_state("What is the answer?"))

        # Should have at least one state
        assert len(states) > 0

        for state in states:
            assert isinstance(state, AgentAnswer)

        # Final state should be complete with final answer
        final_state = states[-1]
        assert final_state.is_complete
        assert final_state.final_answer == "The answer is 42"
        assert final_state.query == "What is the answer?"

    def test_stream_state_with_tools(self, mock_llm_client_with_responses):
        """Test that stream_state properly tracks tool executions."""
        step_response = """```json
{
  "tool_thought": "I need to calculate",
  "tool_calls": [
    {
      "id": "call_123",
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

        states = list(agent.stream_state("What is 5 + 3?"))

        # Find the state with tool executions
        states_with_tools = [s for s in states if s.steps and any(step.tool_executions for step in s.steps)]
        assert len(states_with_tools) > 0

        # Check tool execution details
        tool_state = states_with_tools[0]
        assert len(tool_state.steps) > 0

        # Find step with tool executions
        step_with_tools = next(step for step in tool_state.steps if step.tool_executions)
        assert len(step_with_tools.tool_executions) > 0

        tool_exec = step_with_tools.tool_executions[0]
        assert tool_exec.tool_name == "calculator"
        assert tool_exec.parameters == {"a": 5, "b": 3, "operation": "add"}

        # Final state should have the answer
        final_state = states[-1]
        assert final_state.is_complete
        assert final_state.final_answer == "The result is 8"

    def test_stream_state_with_plan(self, mock_llm_client_with_responses):
        """Test that stream_state properly tracks plan events."""
        plan_response = """```json
{
  "plan": "Step 1: Analyze the question\\nStep 2: Provide the answer"
}
```"""
        final_response = """```json
{
  "final_answer": "The answer is ready"
}
```"""

        client = mock_llm_client_with_responses([plan_response, final_response])
        agent = Agent(llm_client=client)

        states = list(agent.stream_state("What is your plan?"))

        # Should have at least one state
        assert len(states) > 0

        # All states should be AgentAnswer objects
        for state in states:
            assert isinstance(state, AgentAnswer)

        # Find states with plan
        states_with_plan = [s for s in states if s.steps and any(step.plan for step in s.steps)]
        assert len(states_with_plan) > 0

        # Check plan details
        plan_state = states_with_plan[0]
        step_with_plan = next(step for step in plan_state.steps if step.plan)
        assert "Step 1" in step_with_plan.plan
        assert "Step 2" in step_with_plan.plan
        assert step_with_plan.step_type == "plan"

        # Final state should be complete
        final_state = states[-1]
        assert final_state.is_complete
        assert final_state.final_answer == "The answer is ready"
        assert final_state.query == "What is your plan?"
