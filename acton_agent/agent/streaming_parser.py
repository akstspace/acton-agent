"""
Streaming parser for agent events.
"""

from typing import Generator, Optional, Dict, Any, Literal

from loguru import logger
import jiter

from .models import (
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
    ToolCall,
)


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

    __slots__ = ("step_buffers", "detected_types")

    def __init__(self):
        """
        Create a StreamingTokenParser and initialize internal state.
        
        Initializes:
        - step_buffers (Dict[str, bytearray]): per-step byte buffers used to accumulate incoming tokens efficiently.
        - detected_types (Dict[str, EventType]): map of step_id to a heuristically detected event type ("plan", "step", "final_response", or "unknown").
        """
        self.step_buffers: Dict[
            str, bytearray
        ] = {}  # Use bytearray for faster concatenation
        self.detected_types: Dict[str, EventType] = {}

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

    def _complete_partial_json(self, json_bytes: bytes) -> bytes:
        """
        Complete a possibly truncated JSON byte sequence by appending minimal tokens to produce a valid JSON value.
        
        This function makes targeted fixes for common truncation cases: it returns b"{}" for empty input, appends a closing quote if a string is left open, closes any unmatched arrays or objects, and inserts a simple empty value for trailing keys or colons when a key/value pattern is incomplete. If no completion is required the original bytes are returned unchanged.
        
        Parameters:
            json_bytes (bytes): Partial JSON data as UTF-8 bytes.
        
        Returns:
            bytes: The original bytes augmented with the minimal suffix needed to form a complete JSON value (or the original bytes if no changes were necessary).
        """
        if not json_bytes:
            return b"{}"

        # Strip whitespace
        json_bytes = json_bytes.strip()
        if json_bytes == b"{":
            return b"{}"

        # Fast byte-level counting
        open_braces = open_brackets = quote_count = 0
        has_colon = False
        escape_next = False
        ends_with_colon = False

        for i, byte in enumerate(json_bytes):
            if escape_next:
                escape_next = False
                continue

            if byte == BACKSLASH:
                escape_next = True
            elif byte == QUOTE:
                quote_count += 1
            elif byte == OPEN_BRACE:
                open_braces += 1
            elif byte == CLOSE_BRACE:
                open_braces -= 1
            elif byte == OPEN_BRACKET:
                open_brackets += 1
            elif byte == CLOSE_BRACKET:
                open_brackets -= 1
            elif byte == COLON:
                has_colon = True
                ends_with_colon = i == len(json_bytes) - 1

        # Build completion suffix
        suffix = bytearray()

        # Handle incomplete key-value patterns
        if not has_colon and quote_count > 0:
            suffix.extend(b':"" ')
        elif ends_with_colon:
            suffix.extend(b'""')

        # Close unclosed string
        if quote_count % 2 == 1:
            suffix.append(QUOTE)

        # Close arrays and objects
        if open_brackets > 0:
            suffix.extend(b"]" * open_brackets)
        if open_braces > 0:
            suffix.extend(b"}" * open_braces)

        if suffix:
            return json_bytes + bytes(suffix)
        return json_bytes

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

    def _detect_event_type_from_partial(self, data: Dict[str, Any]) -> EventType:
        """
        Determine the agent event type based on keys present in a partially parsed JSON payload.
        
        Parameters:
            data (Dict[str, Any]): Partially parsed JSON object to inspect for indicative keys.
        
        Returns:
            EventType: `'plan'` if `data` contains the key `"plan"`, `'step'` if it contains `"tool_calls"` or `"tool_thought"`, `'final_response'` if it contains `"final_answer"`, and `'unknown'` otherwise.
        """
        # Optimized: single pass through keys
        if "plan" in data:
            return "plan"
        if "tool_calls" in data or "tool_thought" in data:
            return "step"
        if "final_answer" in data:
            return "final_response"
        return "unknown"

    def try_parse_partial(self, step_id: str) -> Optional[StreamingEvent]:
        """
        Attempt to parse the accumulated token buffer for a step into a structured streaming event.
        
        Given the current buffered tokens for `step_id`, this method extracts JSON-like content (including from markdown code fences), attempts to complete any partial JSON, and parses it into one of the structured streaming event types (AgentPlanEvent, AgentStepEvent, or AgentFinalResponseEvent) when enough information is present.
        
        Parameters:
            step_id (str): Identifier of the step whose buffer will be parsed.
        
        Returns:
            Optional[StreamingEvent]: A fully formed streaming event (AgentPlanEvent, AgentStepEvent, or AgentFinalResponseEvent) when the buffer contains sufficient parseable information; `None` if the buffer is empty, incomplete, contains unsupported data, or cannot yet be converted into a structured event.
        """
        buffer = self.get_buffer(step_id)
        if not buffer:
            return None

        json_bytes = self._extract_json_from_markdown(buffer)
        completed_json = self._complete_partial_json(json_bytes)

        try:
            data = jiter.from_json(completed_json)

            if not isinstance(data, dict):
                return None

            detected_type = self.detected_types.get(step_id)
            if detected_type is None:
                detected_type = self._detect_event_type_from_partial(data)
                if detected_type != "unknown":
                    self.detected_types[step_id] = detected_type
                    logger.debug(
                        f"🎯 Early detection: {detected_type} (step_id={step_id})"
                    )

            if detected_type == "plan" and "plan" in data:
                plan_str = str(data["plan"]) if data["plan"] else ""
                is_complete = bool(plan_str.strip())
                return AgentPlanEvent(
                    step_id=step_id, plan=AgentPlan(plan=plan_str), complete=is_complete
                )

            elif detected_type == "step" and (
                "tool_thought" in data or "tool_calls" in data
            ):
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
                    step=AgentStep(
                        tool_thought=data.get("tool_thought"), tool_calls=tool_calls
                    ),
                    complete=bool(tool_calls),
                )

            elif detected_type == "final_response" and (
                "thought" in data or "final_answer" in data
            ):
                final_answer = data.get("final_answer", "")
                return AgentFinalResponseEvent(
                    step_id=step_id,
                    response=AgentFinalResponse(
                        thought=data.get("thought"), final_answer=final_answer
                    ),
                    complete=bool(final_answer),
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
    current_step_id: Optional[str] = None
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