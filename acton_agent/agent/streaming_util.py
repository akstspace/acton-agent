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

    Args:
        agent_stream: Generator yielding streaming events from agent.run_stream()
        query: The original user query

    Yields:
        The same AgentAnswer object with progressively updated state

    Example:
        ```python
        from acton_agent.agent import parse_streaming_events

        agent = Agent(...)
        raw_stream = agent.run_stream("What's the weather?")

        for state in stream_agent_state(
            parse_streaming_events(raw_stream),
            "What's the weather?"
        ):
            print(f"Steps: {len(state.steps)}")
            print(f"Complete: {state.is_complete}")
            if state.final_answer:
                print(f"Answer: {state.final_answer}")
        ```
    """
    # Initialize state once
    state = AgentAnswer(query=query)

    # Track tool call ID mapping
    tool_call_id_map: dict[str, str] = {}

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

                # Add tool calls
                if event.step.tool_calls:
                    for tool_call in event.step.tool_calls:
                        tool_exec = ToolExecution(
                            tool_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            parameters=tool_call.parameters,
                            status="pending",
                        )
                        step.tool_executions.append(tool_exec)
                        tool_call_id_map[tool_call.id] = tool_call.id

            elif isinstance(event, AgentToolExecutionEvent):
                if step.step_type != "execution":
                    step.step_type = "execution"

                # Find or create tool execution
                tool_id = tool_call_id_map.get(event.tool_call_id, event.tool_call_id)
                tool_exec = next((t for t in step.tool_executions if t.tool_id == tool_id), None)

                if not tool_exec:
                    tool_exec = ToolExecution(tool_id=tool_id, tool_name=event.tool_name, status=event.status)
                    step.tool_executions.append(tool_exec)
                else:
                    tool_exec.status = event.status

                # Update result or error
                if event.status in ["completed", "failed"] and event.result:
                    if event.result.success:
                        tool_exec.result = str(event.result.result)
                    else:
                        tool_exec.error = str(event.result.error)

            elif isinstance(event, AgentToolResultsEvent):
                if step.step_type != "execution":
                    step.step_type = "execution"

                # Update tool executions with results
                for result in event.results:
                    tool_id = result.tool_call_id
                    tool_exec = next((t for t in step.tool_executions if t.tool_id == tool_id), None)

                    if tool_exec:
                        tool_exec.status = "completed" if result.success else "failed"
                        if result.success:
                            tool_exec.result = str(result.result)
                        else:
                            tool_exec.error = str(result.error)

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
