"""
Tests for the process_output method in tools.
"""

import json
from unittest.mock import Mock, patch

import pytest

from acton_agent.tools import FunctionTool, RequestsTool, Tool


class CustomProcessingTool(RequestsTool):
    """Test tool that processes output by extracting specific fields."""

    def process_output(self, output: str) -> str:
        """Extract only 'name' and 'value' fields from JSON."""
        try:
            data = json.loads(output)
            simplified = {"name": data.get("name"), "value": data.get("value")}
            return json.dumps(simplified)
        except (json.JSONDecodeError, KeyError):
            return output


class TestProcessOutputMethod:
    """Tests for the process_output method."""

    def test_tool_base_class_has_process_output(self):
        """Verify that the base Tool class has process_output method."""

        # Create a minimal Tool subclass for testing
        class MinimalTool(Tool):
            def execute(self, parameters, toolset_params=None):
                """
                Return a fixed test output string.
                
                Parameters:
                    parameters: Ignored.
                    toolset_params: Ignored.
                
                Returns:
                    str: The literal string "test output".
                """
                return "test output"

            def get_schema(self):
                """
                Provide the JSON Schema describing this tool's input parameters (default empty object).
                
                Returns:
                    dict: A JSON Schema for an object with no defined properties (i.e., {"type": "object", "properties": {}}).
                """
                return {"type": "object", "properties": {}}

        tool = MinimalTool("test", "Test tool")

        # process_output should exist and return unchanged output by default
        assert hasattr(tool, "process_output")
        assert tool.process_output("test") == "test"

    def test_default_process_output_returns_unchanged(self):
        """Default process_output should return input unchanged."""

        class DefaultTool(Tool):
            def execute(self, parameters, toolset_params=None):
                """
                Return a fixed placeholder result used by tests.
                
                Parameters:
                    parameters: Input parameters passed to the tool execution (ignored).
                    toolset_params: Optional toolset-level parameters (ignored).
                
                Returns:
                    The string "result".
                """
                return "result"

            def get_schema(self):
                """
                Provide the JSON Schema describing this tool's input parameters (default empty object).
                
                Returns:
                    dict: A JSON Schema for an object with no defined properties (i.e., {"type": "object", "properties": {}}).
                """
                return {"type": "object", "properties": {}}

        tool = DefaultTool("default", "Default tool")

        original = "some output text"
        processed = tool.process_output(original)

        assert processed == original

    def test_custom_process_output_is_called(self):
        """Verify that custom process_output is invoked during execute."""
        # Mock the requests library
        with patch("acton_agent.tools.requests_tool.requests.request") as mock_request:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "name": "test",
                "value": 123,
                "extra_field": "should be removed",
            }
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            # Create custom tool
            tool = CustomProcessingTool(
                name="test_tool",
                description="Test",
                method="GET",
                url_template="http://test.com/api",
            )

            # Execute the tool
            result = tool.execute({})

            # Verify output was processed
            result_data = json.loads(result)
            assert "name" in result_data
            assert "value" in result_data
            assert "extra_field" not in result_data
            assert result_data["name"] == "test"
            assert result_data["value"] == 123

    def test_process_output_with_non_json_response(self):
        """Test that process_output handles non-JSON responses gracefully."""
        with patch("acton_agent.tools.requests_tool.requests.request") as mock_request:
            # Setup mock response with text (not JSON)
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_response.text = "Plain text response"
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            # Custom tool should handle this gracefully
            tool = CustomProcessingTool(
                name="test_tool",
                description="Test",
                method="GET",
                url_template="http://test.com/api",
            )

            result = tool.execute({})

            # Should return the text as-is since JSON parsing fails
            assert result == "Plain text response"

    def test_multiple_tools_with_different_processing(self):
        """Test that different tool instances can have different processing."""

        class UpperCaseTool(RequestsTool):
            def process_output(self, output: str) -> str:
                return output.upper()

        class LowerCaseTool(RequestsTool):
            def process_output(self, output: str) -> str:
                return output.lower()

        upper_tool = UpperCaseTool("upper", "Upper", "GET", "http://test.com")
        lower_tool = LowerCaseTool("lower", "Lower", "GET", "http://test.com")

        # Each should process differently
        assert upper_tool.process_output("Test") == "TEST"
        assert lower_tool.process_output("Test") == "test"

    def test_function_tool_inherits_default_process_output(self):
        """FunctionTool should also have default process_output behavior."""

        def test_func(param1: str) -> str:
            return f"Result: {param1}"

        tool = FunctionTool(
            name="test_func",
            description="Test function",
            func=test_func,
            schema={"type": "object", "properties": {"param1": {"type": "string"}}},
        )

        # Should have process_output method
        assert hasattr(tool, "process_output")

        # Should return unchanged by default
        assert tool.process_output("test") == "test"

    def test_process_output_with_complex_transformation(self):
        """Test process_output with more complex data transformation."""

        class SummaryTool(RequestsTool):
            """Tool that creates summary from detailed response."""

            def process_output(self, output: str) -> str:
                try:
                    data = json.loads(output)
                    items = data.get("items", [])

                    summary = {
                        "total": len(items),
                        "names": [item.get("name") for item in items],
                        "average_value": sum(item.get("value", 0) for item in items) / len(items) if items else 0,
                    }

                    return json.dumps(summary, indent=2)
                except (json.JSONDecodeError, KeyError, ZeroDivisionError):
                    return output

        with patch("acton_agent.tools.requests_tool.requests.request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {
                "items": [
                    {
                        "name": "Item1",
                        "value": 10,
                        "description": "Long description...",
                    },
                    {
                        "name": "Item2",
                        "value": 20,
                        "description": "Another description...",
                    },
                    {"name": "Item3", "value": 30, "description": "More text..."},
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_request.return_value = mock_response

            tool = SummaryTool("summary", "Summary", "GET", "http://test.com/api")
            result = tool.execute({})

            result_data = json.loads(result)
            assert result_data["total"] == 3
            assert result_data["names"] == ["Item1", "Item2", "Item3"]
            assert result_data["average_value"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])