"""
Core models for the AI Agent Framework.

This module contains Pydantic models representing messages, tool calls,
tool results, and agent responses.
"""

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    pass


class Message(BaseModel):
    """
    Represents a message in the conversation.

    Attributes:
        role: The role of the message sender (user, assistant, or system)
        content: The actual message content
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ToolCall(BaseModel):
    """
    Represents a single tool call request.

    Attributes:
        id: Unique identifier for this tool call
        tool_name: Name of the tool to invoke
        parameters: Dictionary of parameters to pass to the tool
    """

    id: str = Field(..., description="Unique identifier for this tool call")
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolResult(BaseModel):
    """
    Result from executing a tool.

    Attributes:
        tool_call_id: ID of the tool call this result is for
        tool_name: Name of the tool that was executed
        result: Result string from the tool execution
        error: Error message if execution failed
    """

    tool_call_id: str = Field(..., description="ID of the tool call this result is for")
    tool_name: str = Field(..., description="Name of the tool that was called")
    result: str = Field(..., description="Result from the tool execution")
    error: Optional[str] = Field(None, description="Error message if tool execution failed")

    @property
    def success(self) -> bool:
        """
        Determine whether the tool execution succeeded.

        Returns:
            `true` if the tool produced no error, `false` otherwise.
        """
        return self.error is None


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

    tool_thought: Optional[str] = Field(None, description="Agent's reasoning for this step")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tools to call in this step")

    @property
    def has_tool_calls(self) -> bool:
        """
        Determine whether the agent step contains any tool calls.

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


class ToolSet(BaseModel):
    """
    Represents a collection of related tools with a shared description.

    ToolSets allow grouping related tools together and providing a general
    description that applies to the entire group. This is useful for organizing
    tools by domain or functionality.

    Attributes:
        name: Unique name for the toolset
        description: General description of what this group of tools can do
        tools: List of Tool instances in this toolset
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name for the toolset")
    description: str = Field(..., description="General description of what this group of tools can do")
    tools: list[Any] = Field(default_factory=list, description="List of Tool instances in this toolset")


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
    tokens: Optional[list[str]] = Field(None, description="Accumulated tokens for this step")


class AgentToolResultsEvent(BaseModel):
    """Event containing tool execution results."""

    type: Literal["tool_results"] = "tool_results"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    results: list[ToolResult] = Field(..., description="Tool execution results")


class AgentToolExecutionEvent(BaseModel):
    """Event for individual tool execution progress."""

    type: Literal["tool_execution"] = "tool_execution"
    step_id: str = Field(..., description="Unique identifier for this agent step/iteration")
    tool_call_id: str = Field(..., description="ID of the tool call being executed")
    tool_name: str = Field(..., description="Name of the tool being executed")
    status: Literal["started", "completed", "failed"] = Field(..., description="Execution status")
    result: Optional[ToolResult] = Field(
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
StreamingEvent = Union[
    AgentStreamStart,
    AgentToken,
    AgentStreamEnd,
    AgentStepUpdate,
    AgentToolResultsEvent,
    AgentToolExecutionEvent,
    AgentPlanEvent,
    AgentStepEvent,
    AgentFinalResponseEvent,
]
