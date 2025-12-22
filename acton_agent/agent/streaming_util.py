"""
Streaming utility for maintaining and yielding agent state during execution.

This module provides a structured way to track and yield the current state
of the agent during streaming execution, making it easy to build UIs that
display real-time progress.
"""

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field

from .models import (
    AgentFinalResponseEvent,
    AgentPlanEvent,
    AgentStepEvent,
    AgentToolExecutionEvent,
    AgentToolResultsEvent,
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
    """

    step_id: str = Field(..., description="Unique identifier for this step")
    step_number: int = Field(..., description="Sequential step number")
    step_type: str = Field(default="unknown", description="Type: plan, execution, final")
    plan: str | None = Field(None, description="Agent's plan")
    thought: str | None = Field(None, description="Agent's reasoning")
    tool_executions: list[ToolExecution] = Field(default_factory=list, description="Tool executions")
    answer: str | None = Field(None, description="Final answer")
    is_complete: bool = Field(default=False, description="Whether this step is complete")


class AgentAnswer(BaseModel):
    """
    Comprehensive state model representing the current state of agent execution.

    This model contains all information needed to render a real-time UI showing
    the agent's progress, including all steps, tool executions, and final answer.

    Attributes:
        query: The original user query
        steps: List of all agent steps in chronological order
        current_step: The currently active step (if any)
        final_answer: The final answer to the user's query
        is_complete: Whether the agent has finished execution
        error: Error message if execution failed
        metadata: Additional metadata about the execution
    """

    query: str = Field(..., description="The original user query")
    steps: list[AgentStepState] = Field(default_factory=list, description="All agent steps")
    current_step: AgentStepState | None = Field(None, description="Currently active step")
    final_answer: str | None = Field(None, description="Final answer")
    is_complete: bool = Field(default=False, description="Whether execution is complete")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_step(self, step: AgentStepState) -> None:
        """
        Add a completed step to the history and update current step.

        Args:
            step: The step to add
        """
        if self.current_step and self.current_step.step_id != step.step_id and self.current_step not in self.steps:
            # Save the previous current step if it's different
            self.steps.append(self.current_step)
        self.current_step = step

    def complete_current_step(self) -> None:
        """Mark the current step as complete and move it to steps history."""
        if self.current_step:
            self.current_step.is_complete = True
            if self.current_step not in self.steps:
                self.steps.append(self.current_step)
            self.current_step = None

    def get_or_create_current_step(self, step_id: str) -> AgentStepState:
        """
        Get the current step or create a new one if step_id doesn't match.

        Args:
            step_id: The step ID to check or create

        Returns:
            The current step
        """
        if not self.current_step or self.current_step.step_id != step_id:
            # Complete previous step
            if self.current_step:
                self.complete_current_step()

            # Create new step
            step_number = len(self.steps) + 1
            self.current_step = AgentStepState(step_id=step_id, step_number=step_number)

        return self.current_step


def stream_agent_state(
    agent_stream: Generator, query: str, metadata: dict[str, Any] | None = None
) -> Generator[AgentAnswer, None, None]:
    """
    Process agent streaming events and yield AgentAnswer states.

    This function takes the raw streaming events from the agent and maintains
    a comprehensive state object that is yielded after each event. This makes
    it easy to build UIs that show real-time progress.

    Args:
        agent_stream: Generator yielding streaming events from agent.run_stream()
        query: The original user query
        metadata: Optional metadata to include in the state

    Yields:
        AgentAnswer objects representing the current state after each event

    Example:
        ```python
        from acton_agent.agent import parse_streaming_events

        agent = Agent(...)
        raw_stream = agent.run_stream("What's the weather?")

        for state in stream_agent_state(
            parse_streaming_events(raw_stream),
            "What's the weather?"
        ):
            print(f"Step: {state.current_step.step_number if state.current_step else 'N/A'}")
            print(f"Complete: {state.is_complete}")
            if state.final_answer:
                print(f"Answer: {state.final_answer}")
        ```
    """
    # Initialize state
    state = AgentAnswer(query=query, metadata=metadata or {})

    # Track tool call ID mapping
    tool_call_id_map: dict[str, str] = {}

    try:
        for event in agent_stream:
            step_id = getattr(event, "step_id", "unknown")

            # Get or create current step
            current = state.get_or_create_current_step(step_id)

            # Process different event types
            if isinstance(event, AgentPlanEvent):
                current.step_type = "plan"
                current.plan = event.plan.plan

            elif isinstance(event, AgentStepEvent):
                current.step_type = "execution"
                if event.step.tool_thought:
                    current.thought = event.step.tool_thought

                # Add tool calls
                if event.step.tool_calls:
                    for tool_call in event.step.tool_calls:
                        tool_exec = ToolExecution(
                            tool_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            parameters=tool_call.parameters,
                            status="pending",
                        )
                        current.tool_executions.append(tool_exec)
                        tool_call_id_map[tool_call.id] = tool_call.id

            elif isinstance(event, AgentToolExecutionEvent):
                if current.step_type != "execution":
                    current.step_type = "execution"

                # Find or create tool execution
                tool_id = tool_call_id_map.get(event.tool_call_id, event.tool_call_id)
                tool_exec = next((t for t in current.tool_executions if t.tool_id == tool_id), None)

                if not tool_exec:
                    tool_exec = ToolExecution(tool_id=tool_id, tool_name=event.tool_name, status=event.status)
                    current.tool_executions.append(tool_exec)
                else:
                    tool_exec.status = event.status

                # Update result or error
                if event.status in ["completed", "failed"] and event.result:
                    if event.result.success:
                        tool_exec.result = str(event.result.result)
                    else:
                        tool_exec.error = str(event.result.error)

            elif isinstance(event, AgentToolResultsEvent):
                if current.step_type != "execution":
                    current.step_type = "execution"

                # Update tool executions with results
                for result in event.results:
                    tool_id = result.tool_call_id
                    tool_exec = next((t for t in current.tool_executions if t.tool_id == tool_id), None)

                    if tool_exec:
                        tool_exec.status = "completed" if result.success else "failed"
                        if result.success:
                            tool_exec.result = str(result.result)
                        else:
                            tool_exec.error = str(result.error)

            elif isinstance(event, AgentFinalResponseEvent):
                current.step_type = "final"
                current.answer = event.response.final_answer
                state.final_answer = event.response.final_answer
                state.complete_current_step()
                state.is_complete = True

            # Yield current state
            yield state

        # Final yield to ensure completion
        if not state.is_complete:
            state.complete_current_step()
            state.is_complete = True
            yield state

    except Exception as e:
        state.error = str(e)
        state.is_complete = True
        state.complete_current_step()
        yield state
