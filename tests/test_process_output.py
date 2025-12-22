"""
Tests for the process_output method in tools.
"""

import json
from typing import Any

import pytest
from pydantic import Field

from acton_agent.tools import FunctionTool, Tool
from acton_agent.tools.models import ToolInputSchema


class CustomProcessingTool(Tool):
    """Test tool that processes output by extracting specific fields."""

    def __init__(self):
        super().__init__(name="custom_processing", description="Custom processing tool")

    def execute(self, parameters: dict[str, Any]) -> str:
        """Return dummy JSON for testing."""
        raw_output = json.dumps({"name": "test", "value": 123, "extra": "ignored"})
        # Process the output before returning
        return self.process_output(raw_output)

    def get_schema(self) -> dict:
        """Return schema."""
        return {"type": "object", "properties": {}}

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
            def execute(self, parameters: dict[str, Any]) -> str:
                """
                Return a fixed test output string.

                Parameters:
                    parameters: Ignored.

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
            def execute(self, parameters: dict[str, Any]) -> str:
                """
                Return a fixed placeholder result used by tests.

                Parameters:
                    parameters: Input parameters passed to the tool execution (ignored).

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
        # Create custom tool with process_output override
        tool = CustomProcessingTool()

        # Execute the tool - it returns JSON with extra field
        result = tool.execute({})

        # Verify output was processed - extra field should be removed
        result_data = json.loads(result)
        assert "name" in result_data
        assert "value" in result_data
        assert "extra" not in result_data
        assert result_data["name"] == "test"
        assert result_data["value"] == 123

    def test_process_output_with_non_json_response(self):
        """Test that process_output handles non-JSON responses gracefully."""

        class PlainTextTool(Tool):
            """Tool that returns plain text."""

            def execute(self, parameters: dict[str, Any]) -> str:
                """Return plain text."""
                return "Plain text response"

            def get_schema(self) -> dict:
                """Return schema."""
                return {"type": "object", "properties": {}}

        tool = PlainTextTool("text_tool", "Plain text tool")
        result = tool.execute({})

        # Should return the text as-is since no custom processing
        assert result == "Plain text response"

    def test_multiple_tools_with_different_processing(self):
        """Test that different tool instances can have different processing."""

        class UpperCaseTool(Tool):
            """Tool that converts output to uppercase."""

            def execute(self, parameters: dict[str, Any]) -> str:
                """Return test string."""
                return "test"

            def get_schema(self) -> dict:
                """Return schema."""
                return {"type": "object", "properties": {}}

            def process_output(self, output: str) -> str:
                """Convert to uppercase."""
                return output.upper()

        class LowerCaseTool(Tool):
            """Tool that converts output to lowercase."""

            def execute(self, parameters: dict[str, Any]) -> str:
                """Return test string."""
                return "TEST"

            def get_schema(self) -> dict:
                """Return schema."""
                return {"type": "object", "properties": {}}

            def process_output(self, output: str) -> str:
                """Convert to lowercase."""
                return output.lower()

        upper_tool = UpperCaseTool("upper", "Upper")
        lower_tool = LowerCaseTool("lower", "Lower")

        # Each should process differently
        assert upper_tool.process_output("Test") == "TEST"
        assert lower_tool.process_output("Test") == "test"

    def test_function_tool_inherits_default_process_output(self):
        """FunctionTool should also have default process_output behavior."""

        def test_func(param1: str) -> str:
            """Test function."""
            return f"Result: {param1}"

        class TestInputSchema(ToolInputSchema):
            """Input schema for test function."""

            param1: str = Field(description="Test parameter")

        tool = FunctionTool(
            name="test_func",
            description="Test function",
            func=test_func,
            input_schema=TestInputSchema,
        )

        # Should have process_output method
        assert hasattr(tool, "process_output")

        # Should return unchanged by default
        assert tool.process_output("test") == "test"

    def test_process_output_with_complex_transformation(self):
        """Test process_output with more complex data transformation."""

        class SummaryTool(Tool):
            """Tool that creates summary from detailed response."""

            def execute(self, parameters: dict[str, Any]) -> str:
                """Return detailed JSON data."""
                raw_output = json.dumps(
                    {
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
                )
                # Process the output before returning
                return self.process_output(raw_output)

            def get_schema(self) -> dict:
                """Return schema."""
                return {"type": "object", "properties": {}}

            def process_output(self, output: str) -> str:
                """Create summary from detailed data."""
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

        tool = SummaryTool("summary", "Summary")
        result = tool.execute({})

        result_data = json.loads(result)
        assert result_data["total"] == 3
        assert result_data["names"] == ["Item1", "Item2", "Item3"]
        assert result_data["average_value"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
