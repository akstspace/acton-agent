"""
Core models for the AI Agent Framework.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    Represents a message in the conversation.

    Attributes:
        role: The role of the message sender (user, assistant, or system)
        content: The actual message content
    """

    role: Literal["user", "assistant", "system"]
    content: str


class AgentPlan(BaseModel):
    """
    Initial planning step from the agent.

    This represents the agent's high-level plan for solving the user's request.
    It should outline what the agent intends to do step by step.

    Attributes:
        plan: The planned steps to accomplish the task
    """

    plan: str = Field(default="", description="The planned steps to accomplish the task")


class AgentStep(BaseModel):
    """
    Intermediate step with tool calls.

    This represents an intermediate reasoning step where the agent decides
    to call one or more tools to gather information.

    Attributes:
        tool_thought: Agent's reasoning for this step
        tool_calls: List of tools to call
    """

    tool_thought: str | None = Field(None, description="Agent's reasoning for this step")
    tool_calls: list[Any] = Field(default_factory=list, description="Tools to call in this step")

    @property
    def has_tool_calls(self) -> bool:
        """
        Check whether the agent step contains one or more tool calls.
        
        Returns:
            `true` if the step contains one or more tool calls, `false` otherwise.
        """
        return len(self.tool_calls) > 0


class AgentFinalResponse(BaseModel):
    """
    Final response to the user.

    This represents the agent's final answer after completing all necessary steps.

    Attributes:
        final_answer: The complete answer to the user's request
    """

    final_answer: str = Field(
        default="",
        description="The complete answer to the user's request",
    )


# Streaming Event Models


class AgentStreamStart(BaseModel):
    """Event indicating the start of LLM streaming."""

    type: Literal["stream_start"] = "stream_start"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")


class AgentToken(BaseModel):
    """Event containing a single token from the LLM stream."""

    type: Literal["token"] = "token"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    content: str = Field(..., description="Token content")


class AgentStreamEnd(BaseModel):
    """Event indicating the end of LLM streaming."""

    type: Literal["stream_end"] = "stream_end"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")


class AgentStepUpdate(BaseModel):
    """Event containing partial parsed data during streaming."""

    type: Literal["step_update"] = "step_update"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    data: dict[str, Any] = Field(..., description="Partially parsed JSON data")
    complete: bool = Field(..., description="Whether this step is complete")
    tokens: list[str] | None = Field(None, description="Accumulated tokens for this step")


class AgentToolResultsEvent(BaseModel):
    """Event containing tool execution results."""

    type: Literal["tool_results"] = "tool_results"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    results: list[Any] = Field(..., description="Tool execution results")


class AgentToolExecutionEvent(BaseModel):
    """Event for individual tool execution progress."""

    type: Literal["tool_execution"] = "tool_execution"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    tool_call_id: str = Field(..., description="ID of the tool call being executed")
    tool_name: str = Field(..., description="Name of the tool being executed")
    status: Literal["started", "completed", "failed"] = Field(..., description="Execution status")
    result: Any | None = Field(
        None,
        description="Tool result (only present when status is completed or failed)",
    )


class AgentPlanEvent(BaseModel):
    """Event containing a complete agent plan."""

    type: Literal["agent_plan"] = "agent_plan"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    plan: AgentPlan = Field(..., description="The agent's plan")
    complete: bool = True


class AgentStepEvent(BaseModel):
    """Event containing a complete agent step."""

    type: Literal["agent_step"] = "agent_step"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    step: AgentStep = Field(..., description="The agent's step")
    complete: bool = True


class AgentFinalResponseEvent(BaseModel):
    """Event containing the final agent response."""

    type: Literal["final_response"] = "final_response"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    response: AgentFinalResponse = Field(..., description="The agent's final response")
    complete: bool = True


# Union type for all streaming events
StreamingEvent = (
    AgentStreamStart
    | AgentToken
    | AgentStreamEnd
    | AgentStepUpdate
    | AgentToolResultsEvent
    | AgentToolExecutionEvent
    | AgentPlanEvent
    | AgentStepEvent
    | AgentFinalResponseEvent
)