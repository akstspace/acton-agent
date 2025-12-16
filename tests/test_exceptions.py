"""
Tests for the exceptions module.
"""

from acton_agent.agent.exceptions import (
    AgentError,
    InvalidToolSchemaError,
    LLMCallError,
    MaxIterationsError,
    ResponseParseError,
    ToolExecutionError,
    ToolNotFoundError,
)


class TestAgentError:
    """Tests for AgentError base exception."""

    def test_agent_error(self):
        """Test creating base AgentError."""
        error = AgentError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError creation."""
        error = ToolNotFoundError("calculator")
        assert error.tool_name == "calculator"
        assert "calculator" in str(error)
        assert "not found" in str(error)
        assert isinstance(error, AgentError)


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_tool_execution_error(self):
        """Test ToolExecutionError creation."""
        original = ValueError("Original error")
        error = ToolExecutionError("calculator", original)

        assert error.tool_name == "calculator"
        assert error.original_error is original
        assert "calculator" in str(error)
        assert "Original error" in str(error)
        assert isinstance(error, AgentError)


class TestLLMCallError:
    """Tests for LLMCallError."""

    def test_llm_call_error(self):
        """Test LLMCallError creation."""
        original = RuntimeError("API timeout")
        error = LLMCallError(original, retry_count=3)

        assert error.original_error is original
        assert error.retry_count == 3
        assert "3 retries" in str(error)
        assert "API timeout" in str(error)
        assert isinstance(error, AgentError)

    def test_llm_call_error_default_retry(self):
        """Test LLMCallError with default retry count."""
        original = RuntimeError("Error")
        error = LLMCallError(original)

        assert error.retry_count == 0


class TestResponseParseError:
    """Tests for ResponseParseError."""

    def test_response_parse_error(self):
        """Test ResponseParseError creation."""
        response = "invalid json {"
        original = ValueError("JSON decode error")
        error = ResponseParseError(response, original)

        assert error.response_text == response
        assert error.original_error is original
        assert "parse" in str(error).lower()
        assert isinstance(error, AgentError)


class TestMaxIterationsError:
    """Tests for MaxIterationsError."""

    def test_max_iterations_error(self):
        """Test MaxIterationsError creation."""
        error = MaxIterationsError(10)

        assert error.max_iterations == 10
        assert "10" in str(error)
        assert "maximum iterations" in str(error).lower()
        assert isinstance(error, AgentError)


class TestInvalidToolSchemaError:
    """Tests for InvalidToolSchemaError."""

    def test_invalid_tool_schema_error(self):
        """Test InvalidToolSchemaError creation."""
        error = InvalidToolSchemaError("calculator", "missing type field")

        assert error.tool_name == "calculator"
        assert error.reason == "missing type field"
        assert "calculator" in str(error)
        assert "missing type field" in str(error)
        assert isinstance(error, AgentError)


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_agent_error(self):
        """Test that all custom exceptions inherit from AgentError."""
        exceptions = [
            ToolNotFoundError("test"),
            ToolExecutionError("test", ValueError()),
            LLMCallError(ValueError()),
            ResponseParseError("test", ValueError()),
            MaxIterationsError(10),
            InvalidToolSchemaError("test", "reason"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AgentError)
            assert isinstance(exc, Exception)
