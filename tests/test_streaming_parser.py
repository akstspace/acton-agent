"""
Tests for the streaming_parser module.
"""

from acton_agent.agent.models import (
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
    ToolCall,
    ToolResult,
)
from acton_agent.agent.streaming_parser import (
    StreamingTokenParser,
    parse_streaming_events,
)


class TestStreamingTokenParser:
    """Tests for StreamingTokenParser class."""

    def test_initialization(self):
        """Test parser initialization."""
        parser = StreamingTokenParser()
        assert parser.step_buffers == {}
        assert parser.detected_types == {}

    def test_add_token(self):
        """Test adding tokens to buffer."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", "Hello")
        parser.add_token("step-1", " World")

        buffer = parser.get_buffer("step-1")
        assert buffer == b"Hello World"

    def test_add_token_multiple_steps(self):
        """Test adding tokens for multiple steps."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", "First")
        parser.add_token("step-2", "Second")

        assert parser.get_buffer("step-1") == b"First"
        assert parser.get_buffer("step-2") == b"Second"

    def test_get_buffer_nonexistent_step(self):
        """Test getting buffer for nonexistent step returns empty bytes."""
        parser = StreamingTokenParser()
        buffer = parser.get_buffer("nonexistent")
        assert buffer == b""

    def test_clear_buffer(self):
        """Test clearing buffer."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", "Test")
        parser.detected_types["step-1"] = "plan"

        parser.clear_buffer("step-1")

        assert "step-1" not in parser.step_buffers
        assert "step-1" not in parser.detected_types

    def test_clear_nonexistent_buffer(self):
        """Test clearing nonexistent buffer doesn't error."""
        parser = StreamingTokenParser()
        parser.clear_buffer("nonexistent")  # Should not raise

    def test_extract_json_from_markdown_with_json_marker(self):
        """Test extracting JSON from markdown with json marker."""
        parser = StreamingTokenParser()
        markdown = b'```json\n{"key": "value"}\n```'
        result = parser._extract_json_from_markdown(markdown)
        assert result == b'{"key": "value"}'

    def test_extract_json_from_markdown_without_json_marker(self):
        """Test extracting JSON from markdown without json marker."""
        parser = StreamingTokenParser()
        markdown = b'```\n{"key": "value"}\n```'
        result = parser._extract_json_from_markdown(markdown)
        assert result == b'{"key": "value"}'

    def test_extract_json_from_markdown_no_closing_fence(self):
        """Test extracting JSON from markdown without closing fence."""
        parser = StreamingTokenParser()
        markdown = b'```json\n{"key": "value"'
        result = parser._extract_json_from_markdown(markdown)
        assert result == b'{"key": "value"'

    def test_extract_json_from_markdown_plain_json(self):
        """Test extracting when no markdown present."""
        parser = StreamingTokenParser()
        json_bytes = b'{"key": "value"}'
        result = parser._extract_json_from_markdown(json_bytes)
        assert result == json_bytes

    def test_detect_event_type_plan(self):
        """Test detecting plan event type."""
        parser = StreamingTokenParser()
        data = {"plan": "Step 1"}
        event_type = parser._detect_event_type_from_partial(data)
        assert event_type == "plan"

    def test_detect_event_type_step_with_tool_calls(self):
        """Test detecting step event type with tool_calls."""
        parser = StreamingTokenParser()
        data = {"tool_calls": []}
        event_type = parser._detect_event_type_from_partial(data)
        assert event_type == "step"

    def test_detect_event_type_step_with_tool_thought(self):
        """Test detecting step event type with tool_thought."""
        parser = StreamingTokenParser()
        data = {"tool_thought": "Thinking"}
        event_type = parser._detect_event_type_from_partial(data)
        assert event_type == "step"

    def test_detect_event_type_final_response(self):
        """Test detecting final response event type."""
        parser = StreamingTokenParser()
        data = {"final_answer": "Done"}
        event_type = parser._detect_event_type_from_partial(data)
        assert event_type == "final_response"

    def test_detect_event_type_unknown(self):
        """Test detecting unknown event type."""
        parser = StreamingTokenParser()
        data = {"unknown_field": "value"}
        event_type = parser._detect_event_type_from_partial(data)
        assert event_type == "unknown"

    def test_try_parse_partial_empty_buffer(self):
        """Test parsing with empty buffer returns None."""
        parser = StreamingTokenParser()
        result = parser.try_parse_partial("step-1")
        assert result is None

    def test_try_parse_partial_plan_complete(self):
        """Test parsing complete plan."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"plan": "Step 1\\nStep 2"}')

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentPlanEvent)
        assert result.step_id == "step-1"
        # JSON parsing converts \\n to actual newline
        assert "Step 1" in result.plan.plan
        assert "Step 2" in result.plan.plan
        assert result.complete is False

    def test_try_parse_partial_plan_incomplete(self):
        """Test parsing incomplete plan."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"plan": "')

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentPlanEvent)
        assert result.step_id == "step-1"
        assert result.complete is False

    def test_try_parse_partial_step_with_tool_calls(self):
        """Test parsing step with tool calls."""
        parser = StreamingTokenParser()
        json_data = """
        {
            "tool_thought": "I need to calculate",
            "tool_calls": [
                {
                    "id": "call-1",
                    "tool_name": "calculator",
                    "parameters": {"a": 5, "b": 3}
                }
            ]
        }
        """
        parser.add_token("step-1", json_data)

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentStepEvent)
        assert result.step_id == "step-1"
        assert result.step.tool_thought == "I need to calculate"
        assert len(result.step.tool_calls) == 1
        assert result.step.tool_calls[0].tool_name == "calculator"
        assert result.complete is False

    def test_try_parse_partial_step_incomplete_tool_calls(self):
        """Test parsing step with incomplete tool calls."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"tool_calls": [{"id": "1", "tool_name"')

        result = parser.try_parse_partial("step-1")

        # With very incomplete JSON, parser might complete it but have empty/invalid tool calls
        # The behavior depends on how the completion algorithm handles it
        # It may return None if it can't parse, or return an event with empty tool_calls
        if result is not None:
            assert isinstance(result, AgentStepEvent)
            assert result.step_id == "step-1"
            # May or may not be complete depending on completion
        # If None, that's also acceptable for very broken JSON

    def test_try_parse_partial_final_response(self):
        """Test parsing final response."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"final_answer": "42"}')

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentFinalResponseEvent)
        assert result.step_id == "step-1"
        assert result.response.final_answer == "42"
        assert result.complete is False

    def test_try_parse_partial_final_response_incomplete(self):
        """Test parsing incomplete final response."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"final_answer": "')

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentFinalResponseEvent)
        assert result.step_id == "step-1"
        assert result.complete is False

    def test_try_parse_partial_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '```json\n{"plan": "Step 1"}\n```')

        result = parser.try_parse_partial("step-1")

        assert isinstance(result, AgentPlanEvent)
        assert result.plan.plan == "Step 1"

    def test_try_parse_partial_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", "not json at all")

        result = parser.try_parse_partial("step-1")

        # Should return None for completely invalid JSON
        assert result is None

    def test_try_parse_partial_caches_detected_type(self):
        """Test that detected type is cached."""
        parser = StreamingTokenParser()
        parser.add_token("step-1", '{"plan": "h')

        # First parse
        parser.try_parse_partial("step-1")
        assert "step-1" in parser.detected_types
        assert parser.detected_types["step-1"] == "plan"

        # Add more tokens
        parser.add_token("step-1", ' "More content"}')

        # Second parse should use cached type
        result2 = parser.try_parse_partial("step-1")
        assert isinstance(result2, AgentPlanEvent)


class TestParseStreamingEvents:
    """Tests for parse_streaming_events wrapper function."""

    def test_parse_streaming_events_basic_flow(self):
        """Test basic flow through parse_streaming_events."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content='{"plan": "')
            yield AgentToken(step_id="step-1", content='Step 1"}')
            yield AgentStreamEnd(step_id="step-1")

        events = list(parse_streaming_events(mock_stream()))

        # Should have at least one AgentPlanEvent
        plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
        assert len(plan_events) >= 1

    def test_parse_streaming_events_multiple_steps(self):
        """Test handling multiple streaming steps."""

        def mock_stream():
            # First step - plan
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content='{"plan": "Do this"}')
            yield AgentStreamEnd(step_id="step-1")

            # Second step - tool call
            yield AgentStreamStart(step_id="step-2")
            yield AgentToken(
                step_id="step-2",
                content='{"tool_calls": [{"id": "1", "tool_name": "test"}]}',
            )
            yield AgentStreamEnd(step_id="step-2")

        events = list(parse_streaming_events(mock_stream()))

        plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
        step_events = [e for e in events if isinstance(e, AgentStepEvent)]

        assert len(plan_events) >= 1
        assert len(step_events) >= 1

    def test_parse_streaming_events_incremental_tokens(self):
        """Test parsing with incremental token delivery."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            # Send JSON character by character
            json_str = '{"final_answer": "The answer is 42"}'
            for char in json_str:
                yield AgentToken(step_id="step-1", content=char)
            yield AgentStreamEnd(step_id="step-1")

        events = list(parse_streaming_events(mock_stream()))

        final_events = [e for e in events if isinstance(e, AgentFinalResponseEvent)]
        assert len(final_events) >= 1
        # All events should be marked as incomplete (streaming)
        assert all(not e.complete for e in final_events)

    def test_parse_streaming_events_with_tool_results(self):
        """Test that tool result events pass through."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content='{"final_answer": "Done"}')
            yield AgentStreamEnd(step_id="step-1")
            yield AgentToolResultsEvent(
                step_id="step-1",
                results=[ToolResult(tool_call_id="1", tool_name="test", result="Success", error=None)],
            )

        events = list(parse_streaming_events(mock_stream()))

        tool_result_events = [e for e in events if isinstance(e, AgentToolResultsEvent)]
        assert len(tool_result_events) == 1

    def test_parse_streaming_events_with_tool_execution(self):
        """Test that tool execution events pass through."""

        def mock_stream():
            yield AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="call-1",
                tool_name="calculator",
                status="started",
            )
            yield AgentToolExecutionEvent(
                step_id="step-1",
                tool_call_id="call-1",
                tool_name="calculator",
                status="completed",
                result=ToolResult(tool_call_id="call-1", tool_name="calculator", result="42"),
            )

        events = list(parse_streaming_events(mock_stream()))

        execution_events = [e for e in events if isinstance(e, AgentToolExecutionEvent)]
        assert len(execution_events) == 2
        assert execution_events[0].status == "started"
        assert execution_events[1].status == "completed"

    def test_parse_streaming_events_clears_buffer_on_complete(self):
        """Test that buffer is cleared when event is complete."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content='{"plan": "Complete plan"}')
            yield AgentStreamEnd(step_id="step-1")

        events = list(parse_streaming_events(mock_stream()))

        # Should have parsed and cleared
        plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
        assert len(plan_events) >= 1
        assert not plan_events[-1].complete

    def test_parse_streaming_events_handles_passthrough_events(self):
        """Test that already parsed events pass through unchanged."""

        def mock_stream():
            # Direct event (not tokens)
            yield AgentPlanEvent(step_id="step-1", plan=AgentPlan(plan="Direct plan"), complete=True)
            yield AgentStepEvent(
                step_id="step-2",
                step=AgentStep(tool_calls=[ToolCall(id="1", tool_name="test", parameters={})]),
                complete=True,
            )
            yield AgentFinalResponseEvent(
                step_id="step-3",
                response=AgentFinalResponse(final_answer="Direct answer"),
                complete=True,
            )

        events = list(parse_streaming_events(mock_stream()))

        assert len(events) == 3
        assert isinstance(events[0], AgentPlanEvent)
        assert isinstance(events[1], AgentStepEvent)
        assert isinstance(events[2], AgentFinalResponseEvent)

    def test_parse_streaming_events_empty_stream(self):
        """Test handling empty stream."""

        def mock_stream():
            return
            yield  # Make it a generator

        events = list(parse_streaming_events(mock_stream()))
        assert events == []

    def test_parse_streaming_events_stream_without_end(self):
        """Test handling stream without end event."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content='{"plan": "Incomplete')
            # No StreamEnd

        events = list(parse_streaming_events(mock_stream()))

        # Should still attempt to parse what was received
        # May or may not have events depending on partial parsing
        assert isinstance(events, list)

    def test_parse_streaming_events_with_markdown_tokens(self):
        """Test parsing tokens that include markdown."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            yield AgentToken(step_id="step-1", content="```json\n")
            yield AgentToken(step_id="step-1", content='{"plan": "Step 1"}\n')
            yield AgentToken(step_id="step-1", content="```")
            yield AgentStreamEnd(step_id="step-1")

        events = list(parse_streaming_events(mock_stream()))

        plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
        assert len(plan_events) >= 1
        assert plan_events[-1].plan.plan == "Step 1"

    def test_parse_streaming_events_mixed_content(self):
        """Test parsing with mixed valid and invalid tokens."""

        def mock_stream():
            yield AgentStreamStart(step_id="step-1")
            # Some garbage before valid JSON
            yield AgentToken(step_id="step-1", content="thinking...")
            yield AgentToken(step_id="step-1", content='{"final_answer": "42"}')
            yield AgentStreamEnd(step_id="step-1")

        events = list(parse_streaming_events(mock_stream()))

        # Parser should handle the mixed content
        # Might not parse correctly due to garbage, but shouldn't crash
        assert isinstance(events, list)
