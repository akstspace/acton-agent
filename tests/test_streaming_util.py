"""
Tests for streaming_util module.
"""

from acton_agent.agent.models import (
    AgentFinalResponse,
    AgentFinalResponseEvent,
    AgentPlan,
    AgentPlanEvent,
    AgentStep,
    AgentStepEvent,
    AgentToolExecutionEvent,
)
from acton_agent.agent.streaming_util import (
    AgentAnswer,
    AgentStepState,
    ToolExecution,
    stream_agent_state,
)
from acton_agent.tools import ToolCall, ToolResult


class TestToolExecution:
    """Tests for ToolExecution model."""

    def test_create_tool_execution(self):
        """Test creating a tool execution."""
        tool_exec = ToolExecution(
            tool_id="test-id", tool_name="test_tool", parameters={"arg": "value"}, status="pending"
        )

        assert tool_exec.tool_id == "test-id"
        assert tool_exec.tool_name == "test_tool"
        assert tool_exec.parameters == {"arg": "value"}
        assert tool_exec.status == "pending"
        assert tool_exec.result is None
        assert tool_exec.error is None

    def test_tool_execution_with_result(self):
        """Test tool execution with result."""
        tool_exec = ToolExecution(tool_id="test-id", tool_name="test_tool", status="completed", result="Success!")

        assert tool_exec.status == "completed"
        assert tool_exec.result == "Success!"
        assert tool_exec.error is None

    def test_tool_execution_with_error(self):
        """Test tool execution with error."""
        tool_exec = ToolExecution(
            tool_id="test-id", tool_name="test_tool", status="failed", error="Something went wrong"
        )

        assert tool_exec.status == "failed"
        assert tool_exec.error == "Something went wrong"
        assert tool_exec.result is None


class TestAgentStepState:
    """Tests for AgentStepState model."""

    def test_create_step_state(self):
        """Test creating a step state."""
        step = AgentStepState(step_id="step-1", step_number=1, step_type="plan")

        assert step.step_id == "step-1"
        assert step.step_number == 1
        assert step.step_type == "plan"
        assert step.is_complete is False
        assert len(step.tool_executions) == 0

    def test_step_state_with_plan(self):
        """Test step state with plan."""
        step = AgentStepState(step_id="step-1", step_number=1, step_type="plan", plan="Here is my plan")

        assert step.plan == "Here is my plan"

    def test_step_state_with_tools(self):
        """Test step state with tool executions."""
        tool_exec = ToolExecution(tool_id="tool-1", tool_name="test_tool", status="completed")

        step = AgentStepState(step_id="step-1", step_number=1, step_type="execution", tool_executions=[tool_exec])

        assert len(step.tool_executions) == 1
        assert step.tool_executions[0].tool_name == "test_tool"


class TestAgentAnswer:
    """Tests for AgentAnswer model."""

    def test_create_agent_answer(self):
        """Test creating an agent answer."""
        answer = AgentAnswer(query="Test query")

        assert answer.query == "Test query"
        assert len(answer.steps) == 0
        assert answer.final_answer is None
        assert answer.is_complete is False

    def test_get_or_create_step_new(self):
        """Test creating a new step."""
        answer = AgentAnswer(query="Test")

        step = answer.get_or_create_step("step-1")

        assert step.step_id == "step-1"
        assert step.step_number == 1
        assert len(answer.steps) == 1
        assert answer.steps[0] == step

    def test_get_or_create_step_existing(self):
        """Test getting existing step."""
        answer = AgentAnswer(query="Test")

        step1 = answer.get_or_create_step("step-1")
        step1.plan = "My plan"

        step2 = answer.get_or_create_step("step-1")

        assert step1 is step2
        assert step2.plan == "My plan"
        assert len(answer.steps) == 1

    def test_get_or_create_step_different_id(self):
        """Test creating a new step when ID changes."""
        answer = AgentAnswer(query="Test")

        step1 = answer.get_or_create_step("step-1")
        step1.plan = "Plan 1"

        step2 = answer.get_or_create_step("step-2")

        assert step1 is not step2
        assert step2.step_id == "step-2"
        assert step2.step_number == 2
        assert len(answer.steps) == 2
        assert answer.steps[0] == step1
        assert answer.steps[1] == step2


class TestStreamAgentState:
    """Tests for stream_agent_state function."""

    def test_stream_with_plan_event(self):
        """Test streaming with plan event."""
        plan_event = AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="My plan"))

        events = [plan_event]
        states = list(stream_agent_state(iter(events), "Test query"))

        assert len(states) >= 1

        # Check the state after plan event
        state = states[0]
        assert state.query == "Test query"
        assert len(state.steps) == 1
        assert state.steps[0].step_type == "plan"
        assert state.steps[0].plan == "My plan"

    def test_stream_with_step_event(self):
        """Test streaming with step event."""
        tool_call = ToolCall(id="tool-1", tool_name="test_tool", parameters={"arg": "value"})

        step_event = AgentStepEvent(step_id="step-1", step=AgentStep(tool_thought="My thought", tool_calls=[tool_call]))

        events = [step_event]
        states = list(stream_agent_state(iter(events), "Test query"))

        state = states[0]
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.step_type == "execution"
        assert step.thought == "My thought"
        assert len(step.tool_executions) == 1
        assert step.tool_executions[0].tool_name == "test_tool"

    def test_stream_with_final_response(self):
        """Test streaming with final response."""
        final_event = AgentFinalResponseEvent(
            step_id="step-1", response=AgentFinalResponse(final_answer="The answer is 42")
        )

        events = [final_event]
        states = list(stream_agent_state(iter(events), "Test query"))

        # Final state should exist
        assert len(states) >= 1
        final_state = states[-1]

        assert final_state.final_answer == "The answer is 42"
        assert final_state.is_complete is True
        assert len(final_state.steps) == 1
        assert final_state.steps[0].answer == "The answer is 42"

    def test_stream_with_tool_execution(self):
        """Test streaming with tool execution event."""
        result = ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Tool result")

        exec_event = AgentToolExecutionEvent(
            step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="completed", result=result
        )

        events = [exec_event]
        states = list(stream_agent_state(iter(events), "Test query"))

        state = states[0]
        assert len(state.steps) == 1
        assert len(state.steps[0].tool_executions) == 1
        tool_exec = state.steps[0].tool_executions[0]
        assert tool_exec.status == "completed"
        assert tool_exec.result == "Tool result"

    def test_stream_with_error(self):
        """Test streaming with error."""

        def error_generator():
            yield AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="Plan"))
            raise ValueError("Test error")

        states = list(stream_agent_state(error_generator(), "Test query"))

        final_state = states[-1]
        assert final_state.is_complete is True
        assert len(final_state.steps) >= 1
        # Error should be in the last step
        assert final_state.steps[-1].error is not None
        assert "Test error" in final_state.steps[-1].error

    def test_stream_complete_flow(self):
        """Test complete streaming flow."""
        tool_call = ToolCall(id="tool-1", tool_name="test_tool", parameters={})

        result = ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Success")

        events = [
            AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="First, I'll use the tool")),
            AgentStepEvent(step_id="step-2", step=AgentStep(tool_thought="Using the tool", tool_calls=[tool_call])),
            AgentToolExecutionEvent(
                step_id="step-2", tool_call_id="tool-1", tool_name="test_tool", status="completed", result=result
            ),
            AgentFinalResponseEvent(step_id="step-3", response=AgentFinalResponse(final_answer="Done!")),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Should have multiple states
        assert len(states) >= 4

        # Final state should be complete
        final_state = states[-1]
        assert final_state.is_complete is True
        assert final_state.final_answer == "Done!"
        assert len(final_state.steps) == 3  # Should have 3 steps
