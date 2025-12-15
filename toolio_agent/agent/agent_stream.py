"""
Streaming helper utilities for parsing partial JSON responses.

This module provides utilities to handle partial JSON streaming from the agent,
parsing incomplete JSON chunks and providing meaningful updates to the UI.
"""

import re
import json
from typing import Generator, Dict, Any, Optional, List
from io import StringIO

from .models import (
    AgentPlan, AgentStep, AgentFinalResponse,
    AgentStreamStart, AgentToken, AgentStreamEnd,
    AgentStepUpdate, AgentToolResultsEvent,
    AgentPlanEvent, AgentStepEvent, AgentFinalResponseEvent,
    StreamingEvent
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
    Parse partial/incomplete JSON string.
    
    This handles JSON strings that may be truncated mid-way, which is common
    during streaming. It tries to parse as much as possible.
    
    Args:
        json_str: Potentially incomplete JSON string
        
    Returns:
        Parsed object (dict, list, str, etc.) or empty dict if unparseable
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
                partial_mode="trailing-strings"
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
            if json_str.strip().startswith('{'):
                # Count opening and closing braces
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)
            elif json_str.strip().startswith('['):
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')
                if open_brackets > close_brackets:
                    json_str += ']' * (open_brackets - close_brackets)
            
            return json.loads(json_str)
        except Exception:
            return {}


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code block.
    
    Args:
        text: Text potentially containing ```json ... ``` block
        
    Returns:
        Extracted JSON string or None
    """
    # Look for ```json or ```
    pattern = r'```(?:json)?\s*\n?(.*?)(?:```|$)'
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
        self.buffer = StringIO()
        self.in_code_block = False
        self.json_content = ""
        self.last_parsed = {}
    
    def push(self, chunk: str) -> Optional[Dict[str, Any]]:
        """
        Push a new chunk of text and attempt to parse.
        
        Args:
            chunk: New text chunk from stream
            
        Returns:
            Parsed object if parsing succeeds, None otherwise
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
        """Get the current parsed state."""
        return self.last_parsed
    
    def reset(self):
        """Reset the parser state."""
        self.buffer = StringIO()
        self.json_content = ""
        self.last_parsed = {}


def stream_with_partial_json(
    agent_run_generator: Generator,
) -> Generator[StreamingEvent, None, None]:
    """
    Wrap agent.run() generator to provide partial JSON parsing with Pydantic models.
    
    This helper function intercepts streaming tokens, accumulates them,
    and parses partial JSON objects (AgentPlan, AgentStep, AgentFinalResponse)
    as they're being streamed. It provides structured Pydantic events even when the
    JSON is incomplete.
    
    Yields Pydantic models:
    - AgentStreamStart: Stream started
    - AgentToken: Individual token
    - AgentStreamEnd: Stream ended
    - AgentStepUpdate: Partial JSON parsing update (for any type)
    - AgentPlanEvent: Complete plan
    - AgentStepEvent: Complete step with tool calls
    - AgentToolResultsEvent: Tool execution results
    - AgentFinalResponseEvent: Final response
    
    Args:
        agent_run_generator: The generator from agent.run()
        
    Yields:
        Pydantic StreamingEvent models
        
    Example:
        ```python
        agent = Agent(llm_client=client, stream=True)
        
        for event in stream_with_partial_json(agent.run("query")):
            if isinstance(event, AgentStepUpdate):
                if "thought" in event.data:
                    print(f"Thinking: {event.data['thought']}")
                if "final_answer" in event.data:
                    print(f"Answer: {event.data['final_answer']}")
            
            elif isinstance(event, AgentPlanEvent):
                print(f"Plan: {event.plan.plan}")
            
            elif isinstance(event, AgentFinalResponseEvent):
                print(f"Final: {event.response.final_answer}")
        ```
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
                            tokens=current_step_tokens.copy()
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
                        data=final_parsed,
                        complete=True,
                        tokens=current_step_tokens
                    )
                
                yield AgentStreamEnd()
                continue
            
            elif step.get("type") == "tool_results":
                steps_log.append({"type": "tool_results", "data": step["results"]})
                yield AgentToolResultsEvent(results=step["results"])
                continue
        
        # Handle structured responses
        from .models import AgentPlan, AgentStep, AgentFinalResponse
        
        if isinstance(step, AgentPlan):
            steps_log.append({"type": "plan", "data": step})
            yield AgentPlanEvent(plan=step)
        
        elif isinstance(step, AgentStep):
            steps_log.append({"type": "step", "data": step})
            yield AgentStepEvent(step=step)
        
        elif isinstance(step, AgentFinalResponse):
            steps_log.append({"type": "final", "data": step})
            yield AgentFinalResponseEvent(response=step)
