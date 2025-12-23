"""
Tests for the parser module.
"""

import uuid

from acton_agent.agent.models import (
    AgentFinalResponse,
    AgentPlan,
    AgentStep,
)
from acton_agent.parsers import ResponseParser
from acton_agent.tools.models import ToolCall


class TestResponseParser:
    """Tests for ResponseParser."""

    def test_parse_final_answer(self):
        """Test parsing a final answer response."""
        response_text = """```json
{
  "final_answer": "The answer is 42"
}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentFinalResponse)
        assert result.final_answer == "The answer is 42"

    def test_parse_plan(self):
        """Test parsing a plan response."""
        response_text = """```json
{
  "plan": "Step 1: Do this\\nStep 2: Do that\\nStep 3: Complete"
}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentPlan)
        assert "Step 1" in result.plan

    def test_parse_step_with_tool_calls(self):
        """Test parsing a step with tool calls."""
        response_text = """```json
{
  "tool_thought": "I need to call a tool",
  "tool_calls": [
    {
      "id": "call_1",
      "tool_name": "calculator",
      "parameters": {"a": 5, "b": 3}
    }
  ]
}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentStep)
        assert result.tool_thought == "I need to call a tool"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"

    def test_parse_without_code_block(self):
        """Test parsing JSON without code block."""
        response_text = """{"final_answer": "Direct JSON"}"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentFinalResponse)
        assert result.final_answer == "Direct JSON"

    def test_parse_with_json_marker(self):
        """Test parsing with ```json marker."""
        response_text = """```json
{"final_answer": "Answer with marker"}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentFinalResponse)
        assert result.final_answer == "Answer with marker"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON falls back to final answer."""
        response_text = "This is not JSON at all"

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentFinalResponse)
        assert result.final_answer == response_text

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        response_text = ""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentFinalResponse)

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown."""
        text = """Some text before
```json
{"key": "value"}
```
Some text after"""

        extracted = ResponseParser._extract_json_from_markdown(text)
        assert extracted == '{"key": "value"}'

    def test_extract_json_no_marker(self):
        """Test extraction when no code block present."""
        text = '{"key": "value"}'

        extracted = ResponseParser._extract_json_from_markdown(text)
        assert extracted == text


class TestResponseValidation:
    """Tests for response validation."""

    def test_validate_valid_plan(self):
        """Test validating a valid plan."""
        plan = AgentPlan(plan="step1\nstep2")
        assert ResponseParser.validate_response(plan)

    def test_validate_invalid_plan(self):
        """Test validating invalid plan (empty)."""
        plan = AgentPlan(plan="")
        assert not ResponseParser.validate_response(plan)

    def test_validate_valid_step(self):
        """Test validating a valid step."""
        step = AgentStep(
            tool_thought="test",
            tool_calls=[ToolCall(id="1", tool_name="test", parameters={})],
        )
        assert ResponseParser.validate_response(step)

    def test_validate_invalid_step(self):
        """Test validating invalid step (no tool calls)."""
        step = AgentStep(tool_thought="test", tool_calls=[])
        assert not ResponseParser.validate_response(step)

    def test_validate_valid_final_response(self):
        """Test validating a valid final response."""
        response = AgentFinalResponse(final_answer="The answer")
        assert ResponseParser.validate_response(response)

    def test_validate_invalid_final_response(self):
        """Test validating invalid final response (empty answer)."""
        response = AgentFinalResponse(final_answer="")
        assert not ResponseParser.validate_response(response)


class TestExtractThought:
    """Tests for extracting thought from responses."""

    def test_extract_thought_from_plan(self):
        """Test extracting thought from plan - plans don't have thought field."""
        plan = AgentPlan(plan="planning thought")
        thought = ResponseParser.extract_thought(plan)
        assert thought is None

    def test_extract_thought_from_step(self):
        """Test extracting thought from step."""
        step = AgentStep(tool_thought="step thought", tool_calls=[ToolCall(id="1", tool_name="test")])
        thought = ResponseParser.extract_thought(step)
        assert thought == "step thought"

    def test_extract_thought_from_final_response(self):
        """Test extracting thought from final response (should return None since thought field removed)."""
        response = AgentFinalResponse(final_answer="answer")
        thought = ResponseParser.extract_thought(response)
        assert thought is None

    def test_extract_thought_when_none(self):
        """Test extracting thought when it's None."""
        response = AgentFinalResponse(final_answer="answer")
        thought = ResponseParser.extract_thought(response)
        assert thought is None


class TestUUIDValidation:
    """Tests for UUID validation in ResponseParser."""

    def test_is_valid_uuid_with_valid_uuid4(self):
        """Test _is_valid_uuid with a valid UUID4."""
        valid_uuid = str(uuid.uuid4())
        assert ResponseParser._is_valid_uuid(valid_uuid)

    def test_is_valid_uuid_with_invalid_string(self):
        """Test _is_valid_uuid with an invalid string."""
        assert not ResponseParser._is_valid_uuid("not-a-uuid")

    def test_is_valid_uuid_with_empty_string(self):
        """Test _is_valid_uuid with an empty string."""
        assert not ResponseParser._is_valid_uuid("")

    def test_is_valid_uuid_with_partial_uuid(self):
        """Test _is_valid_uuid with a partial UUID."""
        assert not ResponseParser._is_valid_uuid("123e4567-e89b-12d3")

    def test_is_valid_uuid_with_wrong_format(self):
        """Test _is_valid_uuid with wrong format (but correct length)."""
        assert not ResponseParser._is_valid_uuid("123e4567-e89b-12d3-a456-42661417400g")

    def test_is_valid_uuid_with_uppercase(self):
        """Test _is_valid_uuid with uppercase UUID (should fail as str comparison is case-sensitive)."""
        valid_uuid = str(uuid.uuid4())
        assert not ResponseParser._is_valid_uuid(valid_uuid.upper())

    def test_is_valid_uuid_version_3(self):
        """Test _is_valid_uuid with a UUID3."""
        uuid3 = uuid.uuid3(uuid.NAMESPACE_DNS, "example.com")
        assert ResponseParser._is_valid_uuid(str(uuid3), version=3)

    def test_is_valid_uuid_version_5(self):
        """Test _is_valid_uuid with a UUID5."""
        uuid5 = uuid.uuid5(uuid.NAMESPACE_DNS, "example.com")
        assert ResponseParser._is_valid_uuid(str(uuid5), version=5)

    def test_parse_step_generates_uuid_for_invalid_id(self):
        """Test that parse() generates a valid UUID when tool_call.id is invalid."""
        response_text = """```json
{
  "tool_thought": "Need to calculate",
  "tool_calls": [
    {
      "id": "invalid-id",
      "tool_name": "calculator",
      "parameters": {"a": 5, "b": 3}
    }
  ]
}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentStep)
        assert len(result.tool_calls) == 1
        # The invalid ID should be replaced with a valid UUID
        assert ResponseParser._is_valid_uuid(result.tool_calls[0].id)
        assert result.tool_calls[0].id != "invalid-id"

    def test_parse_step_keeps_valid_uuid(self):
        """Test that parse() keeps a valid UUID unchanged."""
        valid_uuid = str(uuid.uuid4())
        response_text = f"""```json
{{
  "tool_thought": "Need to calculate",
  "tool_calls": [
    {{
      "id": "{valid_uuid}",
      "tool_name": "calculator",
      "parameters": {{"a": 5, "b": 3}}
    }}
  ]
}}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentStep)
        assert len(result.tool_calls) == 1
        # The valid UUID should remain unchanged
        assert result.tool_calls[0].id == valid_uuid
