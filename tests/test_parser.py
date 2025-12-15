"""
Tests for the parser module.
"""


from toolio_agent.agent.models import (
    AgentFinalResponse,
    AgentPlan,
    AgentResponse,
    AgentStep,
    ToolCall,
)
from toolio_agent.agent.parser import ResponseParser


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
  "thought": "Let me plan this",
  "plan": ["Step 1", "Step 2", "Step 3"]
}
```"""

        result = ResponseParser.parse(response_text)
        assert isinstance(result, AgentPlan)
        assert result.thought == "Let me plan this"
        assert len(result.plan) == 3

    def test_parse_step_with_tool_calls(self):
        """Test parsing a step with tool calls."""
        response_text = """```json
{
  "thought": "I need to call a tool",
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
        assert result.thought == "I need to call a tool"
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
        plan = AgentPlan(thought="test", plan=["step1", "step2"])
        assert ResponseParser.validate_response(plan)

    def test_validate_invalid_plan(self):
        """Test validating invalid plan (empty)."""
        plan = AgentPlan(thought="test", plan=[])
        assert not ResponseParser.validate_response(plan)

    def test_validate_valid_step(self):
        """Test validating a valid step."""
        step = AgentStep(
            thought="test", tool_calls=[ToolCall(id="1", tool_name="test", parameters={})]
        )
        assert ResponseParser.validate_response(step)

    def test_validate_invalid_step(self):
        """Test validating invalid step (no tool calls)."""
        step = AgentStep(thought="test", tool_calls=[])
        assert not ResponseParser.validate_response(step)

    def test_validate_valid_final_response(self):
        """Test validating a valid final response."""
        response = AgentFinalResponse(final_answer="The answer")
        assert ResponseParser.validate_response(response)

    def test_validate_invalid_final_response(self):
        """Test validating invalid final response (empty answer)."""
        response = AgentFinalResponse(final_answer="")
        assert not ResponseParser.validate_response(response)

    def test_validate_legacy_response(self):
        """Test validating legacy AgentResponse."""
        # Valid with final answer
        response1 = AgentResponse(final_answer="answer")
        assert ResponseParser.validate_response(response1)

        # Valid with tool calls
        response2 = AgentResponse(tool_calls=[ToolCall(id="1", tool_name="test", parameters={})])
        assert ResponseParser.validate_response(response2)

        # Invalid (neither final answer nor tool calls)
        response3 = AgentResponse()
        assert not ResponseParser.validate_response(response3)


class TestExtractThought:
    """Tests for extracting thought from responses."""

    def test_extract_thought_from_plan(self):
        """Test extracting thought from plan."""
        plan = AgentPlan(thought="planning thought", plan=["step"])
        thought = ResponseParser.extract_thought(plan)
        assert thought == "planning thought"

    def test_extract_thought_from_step(self):
        """Test extracting thought from step."""
        step = AgentStep(thought="step thought", tool_calls=[ToolCall(id="1", tool_name="test")])
        thought = ResponseParser.extract_thought(step)
        assert thought == "step thought"

    def test_extract_thought_from_final_response(self):
        """Test extracting thought from final response."""
        response = AgentFinalResponse(thought="final thought", final_answer="answer")
        thought = ResponseParser.extract_thought(response)
        assert thought == "final thought"

    def test_extract_thought_when_none(self):
        """Test extracting thought when it's None."""
        response = AgentFinalResponse(final_answer="answer")
        thought = ResponseParser.extract_thought(response)
        assert thought is None
