"""
Tests for the models module.
"""

import pytest
from pydantic import ValidationError

from acton_agent.agent.models import (
    AgentFinalResponse,
    AgentPlan,
    AgentStep,
    AgentStepUpdate,
    AgentToken,
    Message,
    ToolCall,
    ToolResult,
)


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a valid message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test all valid message roles."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="Test")
            assert msg.role == role

    def test_invalid_role(self):
        """Test that invalid role raises error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Test")


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            id="call_123", tool_name="calculator", parameters={"a": 1, "b": 2}
        )
        assert tool_call.id == "call_123"
        assert tool_call.tool_name == "calculator"
        assert tool_call.parameters == {"a": 1, "b": 2}

    def test_tool_call_with_empty_parameters(self):
        """Test tool call with no parameters."""
        tool_call = ToolCall(id="call_123", tool_name="get_time")
        assert tool_call.parameters == {}


class TestToolResult:
    """Tests for ToolResult model."""

    def test_successful_tool_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            tool_call_id="call_123", tool_name="calculator", result="42", error=None
        )
        assert result.success
        assert result.result == "42"
        assert result.error is None

    def test_failed_tool_result(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            tool_call_id="call_123",
            tool_name="calculator",
            result="",
            error="Division by zero",
        )
        assert not result.success
        assert result.error == "Division by zero"


class TestAgentPlan:
    """Tests for AgentPlan model."""

    def test_create_plan(self):
        """Test creating an agent plan."""
        plan = AgentPlan(
            thought="Let me plan the solution", plan=["Step 1", "Step 2", "Step 3"]
        )
        assert plan.thought == "Let me plan the solution"
        assert len(plan.plan) == 3

    def test_plan_validation(self):
        """Test that plan requires fields."""
        with pytest.raises(ValidationError):
            AgentPlan(thought="Test")  # Missing plan field


class TestAgentStep:
    """Tests for AgentStep model."""

    def test_create_step(self):
        """Test creating an agent step."""
        step = AgentStep(
            thought="I need to call a tool",
            tool_calls=[ToolCall(id="call_1", tool_name="test", parameters={})],
        )
        assert step.thought == "I need to call a tool"
        assert step.has_tool_calls
        assert len(step.tool_calls) == 1

    def test_step_with_no_tool_calls(self):
        """Test step with empty tool calls."""
        step = AgentStep(thought="Thinking", tool_calls=[])
        assert not step.has_tool_calls


class TestAgentFinalResponse:
    """Tests for AgentFinalResponse model."""

    def test_create_final_response(self):
        """Test creating a final response."""
        response = AgentFinalResponse(
            thought="I have the answer", final_answer="The answer is 42"
        )
        assert response.thought == "I have the answer"
        assert response.final_answer == "The answer is 42"

    def test_final_response_without_thought(self):
        """Test final response without thought."""
        response = AgentFinalResponse(final_answer="The answer is 42")
        assert response.thought is None
        assert response.final_answer == "The answer is 42"


class TestStreamingModels:
    """Tests for streaming event models."""

    def test_agent_token(self):
        """Test AgentToken model."""
        token = AgentToken(content="test")
        assert token.type == "token"
        assert token.content == "test"

    def test_agent_step_update(self):
        """Test AgentStepUpdate model."""
        update = AgentStepUpdate(data={"thought": "partial"}, complete=False)
        assert update.type == "step_update"
        assert update.data == {"thought": "partial"}
        assert not update.complete

    def test_streaming_event_types(self):
        """Test that all streaming events have correct type."""
        from acton_agent.agent.models import (
            AgentFinalResponseEvent,
            AgentPlanEvent,
            AgentStepEvent,
            AgentStreamEnd,
            AgentStreamStart,
        )

        assert AgentStreamStart().type == "stream_start"
        assert AgentStreamEnd().type == "stream_end"

        plan_event = AgentPlanEvent(plan=AgentPlan(thought="test", plan=["step1"]))
        assert plan_event.type == "agent_plan"

        step_event = AgentStepEvent(
            step=AgentStep(
                thought="test", tool_calls=[ToolCall(id="1", tool_name="test")]
            )
        )
        assert step_event.type == "agent_step"

        final_event = AgentFinalResponseEvent(
            response=AgentFinalResponse(final_answer="done")
        )
        assert final_event.type == "final_response"
