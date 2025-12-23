"""
Tests to ensure no duplicate ToolExecution objects are created in streaming_util.

This test module specifically validates that the stream_agent_state function
correctly uses tool_id and step_id to search for and update existing
ToolExecution objects rather than creating duplicates.
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
from acton_agent.agent.streaming_util import stream_agent_state
from acton_agent.tools import ToolCall, ToolResult


class TestNoDuplicateToolExecutions:
    """Tests to ensure no duplicate ToolExecution objects are created."""

    def test_multiple_execution_events_same_tool_no_duplicates(self):
        """Test that multiple execution events for the same tool don't create duplicates."""
        # Scenario: Same tool goes through started -> completed states
        events = [
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Done"),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Verify no duplicates in any state
        for state in states:
            assert len(state.steps) == 1
            step = state.steps[0]

            # Check for duplicate tool_ids
            tool_ids = [te.tool_id for te in step.tool_executions]
            assert len(tool_ids) == len(set(tool_ids)), f"Duplicate tool_ids found: {tool_ids}"

            # Check for duplicate objects
            obj_ids = [id(te) for te in step.tool_executions]
            assert len(obj_ids) == len(set(obj_ids)), "Duplicate ToolExecution objects found"

            # Should have exactly 1 tool execution
            assert len(step.tool_executions) == 1

        # Final state should have completed status
        final_state = states[-1]
        assert final_state.steps[0].tool_executions[0].status == "completed"
        assert final_state.steps[0].tool_executions[0].result == "Done"

    def test_step_event_then_execution_events_no_duplicates(self):
        """Test that AgentStepEvent followed by execution events doesn't create duplicates."""
        tool_call = ToolCall(id="tool-1", tool_name="test_tool", parameters={"arg": "value"})

        events = [
            AgentStepEvent(step_id="step-1", step=AgentStep(tool_thought="My thought", tool_calls=[tool_call])),
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Success"),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Verify no duplicates in any state
        for state in states:
            assert len(state.steps) == 1
            step = state.steps[0]

            tool_ids = [te.tool_id for te in step.tool_executions]
            assert len(tool_ids) == len(set(tool_ids)), f"Duplicate tool_ids found: {tool_ids}"
            assert len(step.tool_executions) == 1, f"Expected 1 tool execution, got {len(step.tool_executions)}"

        # Verify parameters were set from StepEvent
        final_state = states[-1]
        tool_exec = final_state.steps[0].tool_executions[0]
        assert tool_exec.parameters == {"arg": "value"}
        assert tool_exec.status == "completed"

    def test_execution_event_before_step_event_no_duplicates(self):
        """Test that execution event before step event doesn't create duplicates."""
        tool_call = ToolCall(id="tool-1", tool_name="test_tool", parameters={"arg": "value"})

        events = [
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentStepEvent(step_id="step-1", step=AgentStep(tool_thought="My thought", tool_calls=[tool_call])),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Done"),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Verify no duplicates
        for state in states:
            assert len(state.steps) == 1
            step = state.steps[0]
            assert len(step.tool_executions) == 1, f"Expected 1 tool execution, got {len(step.tool_executions)}"

        # Verify final state has both the parameters (from step event) and result (from execution event)
        final_state = states[-1]
        tool_exec = final_state.steps[0].tool_executions[0]
        assert tool_exec.parameters == {"arg": "value"}
        assert tool_exec.status == "completed"
        assert tool_exec.result == "Done"

    def test_multiple_tools_no_cross_contamination(self):
        """Test that multiple different tools don't interfere with each other."""
        tool_call_1 = ToolCall(id="tool-1", tool_name="tool_a", parameters={"arg": "a"})
        tool_call_2 = ToolCall(id="tool-2", tool_name="tool_b", parameters={"arg": "b"})

        events = [
            AgentStepEvent(
                step_id="step-1", step=AgentStep(tool_thought="Using tools", tool_calls=[tool_call_1, tool_call_2])
            ),
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="tool_a", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-2", tool_name="tool_b", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="tool_a",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="tool_a", success=True, result="Result A"),
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-2",
                tool_name="tool_b",
                status="completed",
                result=ToolResult(tool_call_id="tool-2", tool_name="tool_b", success=True, result="Result B"),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Verify no duplicates in any state
        for state in states:
            assert len(state.steps) == 1
            step = state.steps[0]

            # Should have exactly 2 tool executions
            assert len(step.tool_executions) == 2, f"Expected 2 tool executions, got {len(step.tool_executions)}"

            # No duplicate tool_ids
            tool_ids = [te.tool_id for te in step.tool_executions]
            assert len(tool_ids) == len(set(tool_ids)), f"Duplicate tool_ids found: {tool_ids}"

        # Verify final state
        final_state = states[-1]
        tool_execs = final_state.steps[0].tool_executions

        tool_a = next(te for te in tool_execs if te.tool_id == "tool-1")
        tool_b = next(te for te in tool_execs if te.tool_id == "tool-2")

        assert tool_a.result == "Result A"
        assert tool_b.result == "Result B"
        assert tool_a.parameters == {"arg": "a"}
        assert tool_b.parameters == {"arg": "b"}

    def test_multiple_steps_with_same_tool_id_no_cross_contamination(self):
        """Test that different steps can have tools with same IDs without interference."""
        tool_call_1 = ToolCall(id="tool-1", tool_name="test_tool", parameters={"step": 1})
        tool_call_2 = ToolCall(id="tool-1", tool_name="test_tool", parameters={"step": 2})

        events = [
            AgentStepEvent(step_id="step-1", step=AgentStep(tool_thought="Step 1", tool_calls=[tool_call_1])),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Result 1"),
            ),
            AgentStepEvent(step_id="step-2", step=AgentStep(tool_thought="Step 2", tool_calls=[tool_call_2])),
            AgentToolExecutionEvent(
                step_id="step-2",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Result 2"),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))
        final_state = states[-1]

        # Should have 2 steps
        assert len(final_state.steps) == 2

        # Each step should have exactly 1 tool execution
        assert len(final_state.steps[0].tool_executions) == 1
        assert len(final_state.steps[1].tool_executions) == 1

        # Verify results are correct for each step
        assert final_state.steps[0].tool_executions[0].result == "Result 1"
        assert final_state.steps[1].tool_executions[0].result == "Result 2"

        assert final_state.steps[0].tool_executions[0].parameters == {"step": 1}
        assert final_state.steps[1].tool_executions[0].parameters == {"step": 2}

    def test_tool_result_updates_correctly(self):
        """Test that ToolResult properly updates existing ToolExecution."""
        events = [
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(
                    tool_call_id="tool-1", tool_name="test_tool", success=True, result="Final result here"
                ),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))
        final_state = states[-1]

        # Should have exactly 1 tool execution
        assert len(final_state.steps[0].tool_executions) == 1

        tool_exec = final_state.steps[0].tool_executions[0]
        assert tool_exec.status == "completed"
        assert tool_exec.result == "Final result here"
        assert tool_exec.error is None

    def test_tool_error_updates_correctly(self):
        """Test that ToolResult with error properly updates existing ToolExecution."""
        events = [
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="failed",
                result=ToolResult(
                    tool_call_id="tool-1", tool_name="test_tool", result="", error="Something went wrong"
                ),
            ),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))
        final_state = states[-1]

        # Should have exactly 1 tool execution
        assert len(final_state.steps[0].tool_executions) == 1

        tool_exec = final_state.steps[0].tool_executions[0]
        assert tool_exec.status == "failed"
        assert tool_exec.error == "Something went wrong"

    def test_same_state_object_yielded(self):
        """Test that the same AgentAnswer object is yielded for all events."""
        events = [
            AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="Plan")),
            AgentToolExecutionEvent(
                step_id="step-1", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentFinalResponseEvent(step_id="step-2", response=AgentFinalResponse(final_answer="Done")),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # All yielded states should be the same object
        state_ids = [id(state) for state in states]
        assert len(set(state_ids)) == 1, "Different state objects were yielded"

        # The state should have been mutated with updates
        assert len(states) >= 3  # At least one yield per event

    def test_complete_flow_with_all_event_types_no_duplicates(self):
        """Test complete flow with plan, step, execution, and final response events."""
        tool_call = ToolCall(id="tool-1", tool_name="test_tool", parameters={"query": "test"})

        events = [
            AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="First, use the tool")),
            AgentStepEvent(step_id="step-2", step=AgentStep(tool_thought="Using tool", tool_calls=[tool_call])),
            AgentToolExecutionEvent(
                step_id="step-2", tool_call_id="tool-1", tool_name="test_tool", status="started", result=None
            ),
            AgentToolExecutionEvent(
                step_id="step-2",
                tool_call_id="tool-1",
                tool_name="test_tool",
                status="completed",
                result=ToolResult(tool_call_id="tool-1", tool_name="test_tool", success=True, result="Success"),
            ),
            AgentFinalResponseEvent(step_id="step-3", response=AgentFinalResponse(final_answer="All done!")),
        ]

        states = list(stream_agent_state(iter(events), "Test query"))

        # Verify no duplicates in any state
        for state in states:
            for step in state.steps:
                tool_ids = [te.tool_id for te in step.tool_executions]
                assert len(tool_ids) == len(set(tool_ids)), "Duplicate tool_ids found"

        # Final state should have 3 steps
        final_state = states[-1]
        assert len(final_state.steps) == 3

        # Step 2 should have exactly 1 tool execution
        assert len(final_state.steps[1].tool_executions) == 1

        # Verify completion
        assert final_state.is_complete
        assert final_state.final_answer == "All done!"
