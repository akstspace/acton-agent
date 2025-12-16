"""
Streaming helper utilities for parsing partial JSON responses.

This module provides utilities to handle partial JSON streaming from the agent,
parsing incomplete JSON chunks and providing meaningful updates to the UI.
"""

import json
import re
from io import StringIO
from typing import Any, Dict, Generator, Optional

from .models import (
    AgentFinalResponseEvent,
    AgentPlanEvent,
    AgentStepEvent,
    AgentStepUpdate,
    AgentStreamEnd,
    AgentStreamStart,
    AgentToken,
    AgentToolResultsEvent,
    StreamingEvent,
)

try:
    import jiter

    HAS_JITER = True
except ImportError:
    HAS_JITER = False


# Pattern to detect incomplete unicode escapes
PARTIAL_UNICODE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{0,3}$")


def parse_partial_json(json_str: str) -> Any:
    """
    Parse a possibly truncated JSON string and return a best-effort Python object.
    
    This function recovers partial/incomplete JSON commonly seen in streaming outputs by attempting a series of safe, observable recoveries: trimming a trailing unescaped quote or single backslash, removing a trailing incomplete Unicode escape, attempting a partial-parse fast path when available, and finally trying to balance and close unclosed braces or brackets before a final parse attempt. If parsing cannot be recovered, an empty dict is returned.
    
    Parameters:
        json_str (str): The JSON text to parse, which may be incomplete or truncated.
    
    Returns:
        The parsed Python object (e.g., dict, list, str) when parsing succeeds, or `{}` if the input cannot be parsed or recovered.
    """
    if not json_str.strip():
        return {}

    # Clean up trailing issues
    if (
        json_str.endswith('"')
        and not json_str.endswith('\\"')
        and not json_str.endswith(':"')
    ):
        json_str = json_str[:-1]
    elif json_str.endswith("\\") and not json_str.endswith("\\\\"):
        json_str = json_str[:-1]
    else:
        # Workaround for incomplete unicode escapes
        m = PARTIAL_UNICODE_PATTERN.search(json_str)
        if m:
            json_str = json_str[: -len(m.group(0))]

    # Try jiter first if available (faster and better at partial JSON)
    if HAS_JITER:
        try:
            return jiter.from_json(
                json_str.encode("utf-8"),
                cache_mode="keys",
                partial_mode="trailing-strings",
            )
        except Exception:
            pass

    # Fallback to standard json with error recovery
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to parse what we can
        try:
            # Add closing braces/brackets to make it valid
            if json_str.strip().startswith("{"):
                # Count opening and closing braces
                open_braces = json_str.count("{")
                close_braces = json_str.count("}")
                if open_braces > close_braces:
                    json_str += "}" * (open_braces - close_braces)
            elif json_str.strip().startswith("["):
                open_brackets = json_str.count("[")
                close_brackets = json_str.count("]")
                if open_brackets > close_brackets:
                    json_str += "]" * (open_brackets - close_brackets)

            return json.loads(json_str)
        except Exception:
            return {}


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract a JSON string embedded in a Markdown fenced code block.
    
    Searches for the first fenced code block that begins with ```json or ``` and returns its inner content trimmed of surrounding whitespace. If no fenced code block is found, returns None.
    
    Parameters:
        text: Text that may contain a Markdown fenced code block.
    
    Returns:
        The extracted JSON string without the surrounding fences and trimmed of whitespace, or None if no code block is present.
    """
    # Look for ```json or ```
    pattern = r"```(?:json)?\s*\n?(.*?)(?:```|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class StreamingResponseParser:
    """
    Parser for streaming agent responses that handles partial JSON.

    This accumulates tokens and continuously parses partial JSON,
    yielding updates as more data becomes available.
    """

    def __init__(self):
        """
        Initialize parser state for incrementally parsing partial JSON from streaming text.
        
        Attributes:
            buffer (StringIO): Accumulates incoming text chunks.
            in_code_block (bool): Whether the parser is currently inside a Markdown code block.
            json_content (str): Most recently extracted JSON string from the buffer.
            last_parsed (dict | list | Any): Most recent successfully parsed JSON object (empty if none).
        """
        self.buffer = StringIO()
        self.in_code_block = False
        self.json_content = ""
        self.last_parsed = {}

    def push(self, chunk: str) -> Optional[Dict[str, Any]]:
        """
        Ingest a streaming text chunk and return a newly parsed JSON object when the accumulated content produces a different parse.
        
        Parameters:
            chunk (str): Text chunk to append to the internal buffer.
        
        Returns:
            dict | list | Any | None: The parsed JSON value (e.g., dict or list) if parsing succeeds and differs from the previous result, `None` otherwise.
        """
        self.buffer.write(chunk)
        full_text = self.buffer.getvalue()

        # Try to extract JSON from markdown block
        json_str = extract_json_from_markdown(full_text)

        if json_str:
            self.json_content = json_str
            parsed = parse_partial_json(json_str)
            if parsed and parsed != self.last_parsed:
                self.last_parsed = parsed
                return parsed

        return None

    def get_current(self) -> Dict[str, Any]:
        """
        Return the most recent successfully parsed JSON-compatible value extracted from received chunks.
        
        Returns:
            last_parsed (Dict[str, Any]): The last parsed object (JSON-compatible dict/list/etc.); an empty dict if no successful parse exists.
        """
        return self.last_parsed

    def reset(self):
        """
        Reset the parser to its initial empty state.
        
        Clears the internal text buffer and resets the stored JSON content and last parsed result.
        """
        self.buffer = StringIO()
        self.json_content = ""
        self.last_parsed = {}


def stream_with_partial_json(
    agent_run_generator: Generator,
) -> Generator[StreamingEvent, None, None]:
    """
    Wrap an agent.run() generator to emit structured streaming events and partial JSON updates.
    
    Accumulates streaming tokens, extracts JSON embedded in streamed Markdown, and yields high-level events as the agent produces tokens and structured responses.
    
    Parameters:
        agent_run_generator (Generator): Generator produced by agent.run() that yields streaming steps and structured response objects.
    
    Yields:
        StreamingEvent: One of the streaming event models:
            - AgentStreamStart: emitted when a stream starts.
            - AgentToken: emitted for each raw token.
            - AgentStreamEnd: emitted when a stream ends.
            - AgentStepUpdate: emitted when partial or complete JSON is parsed from the current stream; `complete` indicates whether parsing finished.
            - AgentToolResultsEvent: emitted when tool results are produced.
            - AgentPlanEvent: emitted for a complete AgentPlan.
            - AgentStepEvent: emitted for a complete AgentStep.
            - AgentFinalResponseEvent: emitted for a complete AgentFinalResponse.
    """
    parser = StreamingResponseParser()
    steps_log = []
    current_step_tokens = []
    in_llm_response = False

    for step in agent_run_generator:
        # Track streaming state
        if isinstance(step, dict):
            if step.get("type") == "stream_start":
                in_llm_response = True
                parser.reset()
                current_step_tokens = []
                yield AgentStreamStart()
                continue

            elif step.get("type") == "token":
                if in_llm_response:
                    token = step["content"]
                    current_step_tokens.append(token)

                    # Try to parse partial JSON
                    parsed = parser.push(token)

                    if parsed:
                        # We have something parseable - yield updates for any type
                        # The parsed dict can contain fields from AgentPlan, AgentStep, or AgentFinalResponse
                        yield AgentStepUpdate(
                            data=parsed,
                            complete=False,
                            tokens=current_step_tokens.copy(),
                        )

                # Also pass through the raw token
                yield AgentToken(content=step["content"])
                continue

            elif step.get("type") == "stream_end":
                in_llm_response = False
                # Get final parsed state
                final_parsed = parser.get_current()
                if final_parsed:
                    yield AgentStepUpdate(
                        data=final_parsed, complete=True, tokens=current_step_tokens
                    )

                yield AgentStreamEnd()
                continue

            elif step.get("type") == "tool_results":
                steps_log.append({"type": "tool_results", "data": step["results"]})
                yield AgentToolResultsEvent(results=step["results"])
                continue

        # Handle structured responses
        from .models import AgentFinalResponse, AgentPlan, AgentStep

        if isinstance(step, AgentPlan):
            steps_log.append({"type": "plan", "data": step})
            yield AgentPlanEvent(plan=step)

        elif isinstance(step, AgentStep):
            steps_log.append({"type": "step", "data": step})
            yield AgentStepEvent(step=step)

        elif isinstance(step, AgentFinalResponse):
            steps_log.append({"type": "final", "data": step})
            yield AgentFinalResponseEvent(response=step)