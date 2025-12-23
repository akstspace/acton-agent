"""
Streaming utility for maintaining and yielding agent state during execution.

This module provides a structured way to track and yield the current state
of the agent during streaming execution, making it easy to build UIs that
display real-time progress.
"""

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field

from ..agent.models import (
    AgentFinalResponseEvent,
    AgentPlanEvent,
    AgentStepEvent,
    AgentToolExecutionEvent,
)


class ToolExecution(BaseModel):
    """
    Represents the execution state of a single tool.

    Attributes:
        tool_id: Unique identifier for the tool call
        tool_name: Name of the tool being executed
        parameters: Parameters passed to the tool
        status: Current status of the tool execution
        result: Result of the tool execution (if completed)
        error: Error message (if failed)
    """

    tool_id: str = Field(..., description="Unique identifier for the tool call")
    tool_name: str = Field(..., description="Name of the tool being executed")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    status: str = Field(default="pending", description="Execution status: pending, started, completed, failed")
    result: str | None = Field(None, description="Tool execution result")
    error: str | None = Field(None, description="Error message if execution failed")


class AgentStepState(BaseModel):
    """
    Represents the state of a single agent step.

    Attributes:
        step_id: Unique identifier for this step
        step_number: Sequential number of this step
        step_type: Type of step (plan, execution, final)
        plan: The agent's plan for this iteration
        thought: The agent's reasoning for this step
        tool_executions: List of tool executions in this step
        answer: Final answer (only for final step)
        is_complete: Whether this step is complete
        error: Error message if this step failed
    """

    step_id: str = Field(..., description="Unique identifier for this step")
    step_number: int = Field(..., description="Sequential step number")
    step_type: str = Field(default="unknown", description="Type: plan, execution, final")
    plan: str | None = Field(None, description="Agent's plan")
    thought: str | None = Field(None, description="Agent's reasoning")
    tool_executions: list[ToolExecution] = Field(default_factory=list, description="Tool executions")
    answer: str | None = Field(None, description="Final answer")
    is_complete: bool = Field(default=False, description="Whether this step is complete")
    error: str | None = Field(None, description="Error message if step failed")

    def get_or_create_tool_execution(self, tool_id: str, tool_name: str = "") -> ToolExecution:
        """
        Get an existing tool execution by ID or create a new one.

        Searches for a tool execution with matching tool_id in the tool_executions list.
        If found, returns it. If not found, creates a new tool execution,
        appends it to tool_executions, and returns it.

        This method ensures that duplicate ToolExecution objects are never created
        for the same tool_id. When processing multiple events for the same tool
        (e.g., started -> completed), this method returns the same ToolExecution
        object which is then updated in place.

        Args:
            tool_id: The tool call ID to find or create
            tool_name: Name of the tool (used only when creating new execution)

        Returns:
            The found or newly created tool execution
        """
        # Search for existing tool execution by tool_id to prevent duplicates
        for tool_exec in self.tool_executions:
            if tool_exec.tool_id == tool_id:
                return tool_exec

        # Create new tool execution only if not found
        new_tool_exec = ToolExecution(tool_id=tool_id, tool_name=tool_name, status="pending")
        self.tool_executions.append(new_tool_exec)
        return new_tool_exec


class AgentAnswer(BaseModel):
    """
    Simplified state model representing the current state of agent execution.

    This model contains all information needed to render a real-time UI showing
    the agent's progress, including all steps, tool executions, and final answer.

    Attributes:
        query: The original user query
        steps: List of all agent steps in chronological order
        final_answer: The final answer to the user's query
        is_complete: Whether the agent has finished execution
    """

    query: str = Field(..., description="The original user query")
    steps: list[AgentStepState] = Field(default_factory=list, description="All agent steps")
    final_answer: str | None = Field(None, description="Final answer")
    is_complete: bool = Field(default=False, description="Whether execution is complete")

    def get_or_create_step(self, step_id: str) -> AgentStepState:
        """
        Get an existing step by ID or create a new one.

        Searches for a step with matching step_id in the steps list.
        If found, returns it. If not found, creates a new step with
        step_number = len(steps) + 1, appends it to steps, and returns it.

        Args:
            step_id: The step ID to find or create

        Returns:
            The found or newly created step
        """
        # Search for existing step
        for step in self.steps:
            if step.step_id == step_id:
                return step

        # Create new step
        step_number = len(self.steps) + 1
        new_step = AgentStepState(step_id=step_id, step_number=step_number)
        self.steps.append(new_step)
        return new_step


def stream_agent_state(agent_stream: Generator, query: str) -> Generator[AgentAnswer, None, None]:
    """
    Process agent streaming events and yield AgentAnswer states.

    This function takes the raw streaming events from the agent and maintains
    a state object that is yielded after each event. The same AgentAnswer object
    is yielded repeatedly with progressive updates, making it easy to build UIs
    that show real-time progress.

    Note: This is a low-level utility function. Most users should use the
    Agent.run_with_state() method instead, which provides the same functionality
    with a simpler API.

    Args:
        agent_stream: Generator yielding streaming events from agent.run_stream()
        query: The original user query

    Yields:
        The same AgentAnswer object with progressively updated state

    Example:
        ```python
        # Recommended: Use Agent.stream_state() instead
        agent = Agent(...)
        for state in agent.stream_state("What's the weather?"):
            print(f"Steps: {len(state.steps)}")
            print(f"Complete: {state.is_complete}")
            if state.final_answer:
                print(f"Answer: {state.final_answer}")

        # Low-level usage (not recommended for most use cases)
        from acton_agent.parsers.streaming import parse_streaming_events
        raw_stream = agent.run_stream("What's the weather?")
        for state in stream_agent_state(
            parse_streaming_events(raw_stream),
            "What's the weather?"
        ):
            print(f"Steps: {len(state.steps)}")
        ```
    """
    # Initialize single state object that will be yielded repeatedly with updates
    state = AgentAnswer(query=query)

    try:
        for event in agent_stream:
            step_id = getattr(event, "step_id", "unknown")

            # Get or create step for this event
            step = state.get_or_create_step(step_id)

            # Process different event types
            if isinstance(event, AgentPlanEvent):
                step.step_type = "plan"
                step.plan = event.plan.plan

            elif isinstance(event, AgentStepEvent):
                step.step_type = "execution"
                if event.step.tool_thought:
                    step.thought = event.step.tool_thought

                # Add or update tool calls
                # Using get_or_create_tool_execution ensures no duplicates are created
                if event.step.tool_calls:
                    for tool_call in event.step.tool_calls:
                        tool_exec = step.get_or_create_tool_execution(tool_call.id, tool_call.tool_name)
                        tool_exec.tool_name = tool_call.tool_name
                        tool_exec.parameters = tool_call.parameters
                        tool_exec.status = "pending"

            elif isinstance(event, AgentToolExecutionEvent):
                if step.step_type != "execution":
                    step.step_type = "execution"

                # Get or create tool execution (prevents duplicates by searching for existing tool_id)
                tool_exec = step.get_or_create_tool_execution(event.tool_call_id, event.tool_name)
                tool_exec.status = event.status

                # Update result or error
                if event.status in ["completed", "failed"] and event.result:
                    if event.result.success:
                        tool_exec.result = str(event.result.result)
                    else:
                        tool_exec.error = str(event.result.error)

            elif isinstance(event, AgentFinalResponseEvent):
                step.step_type = "final"
                step.answer = event.response.final_answer
                step.is_complete = True
                state.final_answer = event.response.final_answer
                state.is_complete = True

            # Always yield the same state object
            yield state

        # Final yield to ensure completion
        if not state.is_complete:
            state.is_complete = True
            yield state

    except Exception as e:
        # Store error in the current step if available
        if state.steps:
            state.steps[-1].error = str(e)
        state.is_complete = True
        yield state
