"""
Streaming parser for agent events.
"""

from collections.abc import Generator
from typing import Any, Literal

import jiter
from loguru import logger

from ..agent.models import (
    AgentFinalResponse,
    AgentFinalResponseEvent,
    AgentPlan,
    AgentPlanEvent,
    AgentStep,
    AgentStepEvent,
    AgentStreamEnd,
    AgentStreamStart,
    AgentToken,
    AgentToolExecutionEvent,
    AgentToolResultsEvent,
    StreamingEvent,
)
from ..tools.models import ToolCall


EventType = Literal["plan", "step", "final_response", "unknown"]

# Pre-compiled constants for faster checks
MARKDOWN_START = b"```"
MARKDOWN_JSON_START = b"```json"
MARKDOWN_END = b"```"
OPEN_BRACE = ord("{")
CLOSE_BRACE = ord("}")
OPEN_BRACKET = ord("[")
CLOSE_BRACKET = ord("]")
QUOTE = ord('"')
BACKSLASH = ord("\\")
COLON = ord(":")


class StreamingTokenParser:
    """Parser for accumulating and progressively parsing streaming tokens with early event detection."""

    __slots__ = ("detected_types", "step_buffers")

    def __init__(self):
        """
        Create a StreamingTokenParser and initialize internal state.

        Initializes:
        - step_buffers (Dict[str, bytearray]): per-step byte buffers used to accumulate incoming tokens efficiently.
        - detected_types (Dict[str, EventType]): map of step_id to a heuristically detected event type ("plan", "step", "final_response", or "unknown").
        """
        self.step_buffers: dict[str, bytearray] = {}
        self.detected_types: dict[str, EventType] = {}

    def add_token(self, step_id: str, token: str) -> None:
        """Add a token to the buffer for a specific step."""
        if step_id not in self.step_buffers:
            self.step_buffers[step_id] = bytearray()
        self.step_buffers[step_id].extend(token.encode("utf-8"))

    def get_buffer(self, step_id: str) -> bytes:
        """
        Get the accumulated bytes buffer for the specified step identifier.

        Returns:
            bytes: The accumulated bytes for the step, or b"" if no buffer exists.
        """
        buf = self.step_buffers.get(step_id)
        return bytes(buf) if buf else b""

    def clear_buffer(self, step_id: str) -> None:
        """
        Remove and discard any accumulated token buffer and detected event type for the given step.

        Parameters:
            step_id (str): Identifier of the step whose buffer and detected event type should be removed.
        """
        self.step_buffers.pop(step_id, None)
        self.detected_types.pop(step_id, None)

    def _extract_json_from_markdown(self, data: bytes) -> bytes:
        """
        Extract the JSON payload from a fenced markdown code block, if present.

        This method trims surrounding whitespace, detects an opening fence of either ``` or ```json,
        and returns the bytes inside the code fence (with surrounding whitespace removed). If a
        closing fence is not found, returns the bytes after the opening fence trimmed. If the input
        does not start with a markdown fence, returns the trimmed input unchanged.

        Parameters:
                data (bytes): Bytes that may contain a fenced markdown code block.

        Returns:
                bytes: The extracted and trimmed JSON bytes from inside the markdown fence, or the
                trimmed original data if no fence is present or no closing fence is found.
        """
        data = data.strip()

        if not data.startswith(MARKDOWN_START):
            return data

        # Find start of actual JSON (after ```json or ```)
        start = 3  # len(b'```')
        if data.startswith(MARKDOWN_JSON_START):
            start = 7  # len(b'```json')

        # Skip whitespace after opening fence
        while start < len(data) and data[start] in (
            ord(" "),
            ord("\n"),
            ord("\r"),
            ord("\t"),
        ):
            start += 1

        # Find end marker
        end = data.find(MARKDOWN_END, start)
        if end != -1:
            return data[start:end].strip()

        # No closing fence yet - return everything after opening
        return data[start:].strip()

    def _detect_event_type_from_partial(self, data: dict[str, Any]) -> EventType:
        """
        Detects the agent event type from a partially parsed JSON payload.
        
        Parameters:
            data (dict): Partially parsed JSON object whose top-level keys are inspected for indicative fields.
        
        Returns:
            EventType: "plan" if `data` contains the key "plan", "step" if it contains "tool_calls" or "tool_thought", "final_response" if it contains "final_answer", and "unknown" otherwise.
        """
        # Optimized: single pass through keys
        if "plan" in data:
            return "plan"
        if "tool_calls" in data or "tool_thought" in data:
            return "step"
        if "final_answer" in data:
            return "final_response"
        return "unknown"

    def try_parse_partial(self, step_id: str) -> StreamingEvent | None:
        """
        Try to parse the buffered tokens for a step into a structured streaming event.
        
        Attempts to extract JSON (including from markdown code fences) from the step's buffer and, when sufficient fields are present, constructs and returns an AgentPlanEvent, AgentStepEvent, or AgentFinalResponseEvent with complete=False. Returns `None` when the buffer is empty, contains incomplete or unsupported data, or cannot yet be converted into a structured event.
        
        Returns:
            StreamingEvent | None: A `StreamingEvent` instance when a recognizable event is produced, `None` otherwise.
        """
        buffer = self.get_buffer(step_id)
        if not buffer:
            return None

        json_bytes = self._extract_json_from_markdown(buffer)

        try:
            data = jiter.from_json(json_bytes, partial_mode="trailing-strings")

            if not isinstance(data, dict):
                return None

            detected_type = self.detected_types.get(step_id)
            if detected_type is None:
                detected_type = self._detect_event_type_from_partial(data)
                if detected_type != "unknown":
                    self.detected_types[step_id] = detected_type
                    logger.debug(f"🎯 Early detection: {detected_type} (step_id={step_id})")

            if detected_type == "plan" and "plan" in data:
                plan_str = str(data["plan"]) if data["plan"] else ""
                return AgentPlanEvent(step_id=step_id, plan=AgentPlan(plan=plan_str), complete=False)

            if detected_type == "step" and ("tool_thought" in data or "tool_calls" in data):
                tool_calls = []
                tool_calls_data = data.get("tool_calls")

                if isinstance(tool_calls_data, list):
                    # Batch process tool calls
                    for tc in tool_calls_data:
                        if isinstance(tc, dict) and "id" in tc and "tool_name" in tc:
                            tool_calls.append(
                                ToolCall(
                                    id=tc["id"],
                                    tool_name=tc["tool_name"],
                                    parameters=tc.get("parameters", {}),
                                )
                            )
                return AgentStepEvent(
                    step_id=step_id,
                    step=AgentStep(tool_thought=data.get("tool_thought"), tool_calls=tool_calls),
                    complete=False,
                )

            if detected_type == "final_response" and "final_answer" in data:
                final_answer = data.get("final_answer", "")
                return AgentFinalResponseEvent(
                    step_id=step_id,
                    response=AgentFinalResponse(final_answer=final_answer),
                    complete=False,
                )

        except Exception:
            # Expected for incomplete JSON
            pass

        return None


def parse_streaming_events(
    agent_stream: Generator[StreamingEvent, None, None],
) -> Generator[StreamingEvent, None, None]:
    """
    Produce structured streaming events from an agent's raw streaming-event generator.

    This function consumes an upstream generator of StreamingEvent objects, accumulates token payloads per step, attempts incremental parsing of partial JSON payloads into structured events (AgentPlanEvent, AgentStepEvent, AgentFinalResponseEvent), and yields either parsed events or pass-through events from the original stream.

    Parameters:
        agent_stream (Generator[StreamingEvent, None, None]): Source generator producing streaming events from the agent.

    Returns:
        Generator[StreamingEvent, None, None]: A generator that yields structured StreamingEvent instances as they become available (parsed incremental events or original pass-through events).
    """
    parser = StreamingTokenParser()
    current_step_id: str | None = None
    stream_active = False
    last_complete = False

    for event in agent_stream:
        if isinstance(event, AgentStreamStart):
            current_step_id = event.step_id
            stream_active = True
            last_complete = False
            logger.debug(f"🚀 Stream started (step_id={current_step_id})")

        elif isinstance(event, AgentToken):
            if current_step_id:
                parser.add_token(current_step_id, event.content)

                parsed_event = parser.try_parse_partial(current_step_id)
                if parsed_event:
                    yield parsed_event
                    last_complete = getattr(parsed_event, "complete", False)

                    if last_complete:
                        parser.clear_buffer(current_step_id)
                        stream_active = False

        elif isinstance(event, AgentStreamEnd):
            if current_step_id and stream_active and not last_complete:
                parsed_event = parser.try_parse_partial(current_step_id)
                if parsed_event:
                    yield parsed_event

                parser.clear_buffer(current_step_id)

            stream_active = False
            current_step_id = None
            last_complete = False

        elif isinstance(
            event,
            (
                AgentPlanEvent,
                AgentStepEvent,
                AgentFinalResponseEvent,
                AgentToolResultsEvent,
                AgentToolExecutionEvent,
            ),
        ):
            yield event

        else:
            logger.debug(f"➡️ Passing through: {type(event).__name__}")
            yield event