"""
Tests for the tools module.
"""

import json

import pytest

from acton_agent.agent.exceptions import InvalidToolSchemaError, ToolNotFoundError
from acton_agent.tools import FunctionTool, Tool, ToolRegistry


class SimpleTool(Tool):
    """Simple test tool."""

    def __init__(self):
        """
        Create a SimpleTool with the name "simple" and the description "A simple test tool".
        """
        super().__init__(name="simple", description="A simple test tool")

    def execute(self, parameters: dict, toolset_params: dict | None = None) -> str:
        """
        Format and return a result string using the "value" entry from parameters.
        
        Parameters:
            parameters (dict): Mapping that may contain the key "value"; when absent, "default" is used.
            toolset_params (dict | None): Optional toolset-level parameters; ignored by this tool.
        
        Returns:
            str: The string "Result: {value}" where {value} is the resolved parameter.
        """
        value = parameters.get("value", "default")
        return f"Result: {value}"

    def get_schema(self) -> dict:
        """
        JSON Schema describing this tool's expected input.

        Returns:
            dict: JSON Schema with "type": "object" and a "properties" mapping that includes a "value" property of type "string" with description "Input value".
        """
        return {
            "type": "object",
            "properties": {"value": {"type": "string", "description": "Input value"}},
        }


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = SimpleTool()

        registry.register(tool)
        assert registry.has_tool("simple")
        assert len(registry) == 1

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()

        tool1 = SimpleTool()
        tool2 = SimpleTool()
        tool2.name = "simple2"

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2
        assert "simple" in registry
        assert "simple2" in registry

    def test_get_tool(self):
        """Test getting a tool."""
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)

        retrieved = registry.get("simple")
        assert retrieved is not None
        assert retrieved.name == "simple"

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_unregister_tool(self):
        """
        Ensure a registered tool can be removed from the registry. After unregistering, the tool is no longer present.
        """
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)

        assert registry.has_tool("simple")
        registry.unregister("simple")
        assert not registry.has_tool("simple")

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist."""
        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()

        tool1 = SimpleTool()
        tool2 = SimpleTool()
        tool2.name = "simple2"

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert all(isinstance(t, Tool) for t in tools)

    def test_list_tool_names(self):
        """Test listing tool names."""
        registry = ToolRegistry()

        tool1 = SimpleTool()
        tool2 = SimpleTool()
        tool2.name = "simple2"

        registry.register(tool1)
        registry.register(tool2)

        names = registry.list_tool_names()
        assert names == ["simple", "simple2"]

    def test_format_for_prompt(self):
        """Test formatting tools for prompt."""
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)

        formatted = registry.format_for_prompt()
        assert "simple" in formatted
        assert "A simple test tool" in formatted

    def test_clear_registry(self):
        """Test clearing all tools."""
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)

        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0


class TestFunctionTool:
    """Tests for FunctionTool."""

    def test_create_function_tool(self):
        """Test creating a function tool."""

        def add(a: int, b: int) -> int:
            """
            Return the sum of two integers.

            Returns:
                int: The sum of `a` and `b`.
            """
            return a + b

        schema = {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        }

        tool = FunctionTool(name="add", description="Add two numbers", func=add, schema=schema)

        assert tool.name == "add"
        assert tool.description == "Add two numbers"

    def test_execute_function_tool(self):
        """Test executing a function tool."""

        def multiply(a: int, b: int) -> int:
            """
            Multiply two integers.

            Returns:
                int: The product of a and b.
            """
            return a * b

        schema = {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        }

        tool = FunctionTool(
            name="multiply",
            description="Multiply two numbers",
            func=multiply,
            schema=schema,
        )

        result = tool.execute({"a": 5, "b": 3})
        assert result == "15"

    def test_function_tool_with_string_return(self):
        """Test function tool that returns a string."""

        def greet(name: str) -> str:
            """
            Create a greeting for the given person.

            Returns:
                Greeting string in the form "Hello, {name}!".
            """
            return f"Hello, {name}!"

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        tool = FunctionTool(name="greet", description="Greet someone", func=greet, schema=schema)

        result = tool.execute({"name": "Alice"})
        assert result == "Hello, Alice!"

    def test_function_tool_with_dict_return(self):
        """Test function tool that returns a dict."""

        def get_info(id: int) -> dict:
            """
            Return a dictionary with the provided identifier and a status of "active".

            Parameters:
                id (int): Identifier to include in the returned dictionary.

            Returns:
                info (dict): Dictionary with keys "id" (int) and "status" (str) set to "active".
            """
            return {"id": id, "status": "active"}

        schema = {"type": "object", "properties": {"id": {"type": "number"}}}

        tool = FunctionTool(name="get_info", description="Get info", func=get_info, schema=schema)

        result = tool.execute({"id": 123})
        result_dict = json.loads(result)
        assert result_dict["id"] == 123
        assert result_dict["status"] == "active"

    def test_invalid_schema_not_dict(self):
        """
        Ensure FunctionTool raises InvalidToolSchemaError when provided a schema that is not a dict.

        Attempts to construct a FunctionTool with a non-dict schema value and expects InvalidToolSchemaError to be raised.
        """

        def dummy():
            """
            No-op placeholder used where a callable is required.
            """

        with pytest.raises(InvalidToolSchemaError):
            FunctionTool(
                name="test",
                description="Test",
                func=dummy,
                schema="invalid",  # Not a dict
            )

    def test_invalid_schema_no_type(self):
        """
        Ensure creating a FunctionTool with a schema that lacks a top-level "type" field raises InvalidToolSchemaError.

        Constructs a dummy function and attempts to create a FunctionTool with a schema containing only "properties", asserting that InvalidToolSchemaError is raised.
        """

        def dummy():
            """
            No-op placeholder used where a callable is required.
            """

        with pytest.raises(InvalidToolSchemaError):
            FunctionTool(
                name="test",
                description="Test",
                func=dummy,
                schema={"properties": {}},  # No type field
            )

    def test_invalid_schema_wrong_type(self):
        """Test that schema with non-object type raises error."""

        def dummy():
            """
            No-op placeholder used where a callable is required.
            """

        with pytest.raises(InvalidToolSchemaError):
            FunctionTool(
                name="test",
                description="Test",
                func=dummy,
                schema={"type": "string"},  # Must be object
            )


class TestToolAgentMd:
    """Tests for tool agent_md method."""

    def test_agent_md_formatting(self):
        """Test agent_md method."""
        tool = SimpleTool()
        template = "# {tool_name}\n\n{description}\n\n## Output\n{output}"

        result = tool.agent_md(template, "test output")

        assert "# simple" in result
        assert "A simple test tool" in result
        assert "test output" in result

    def test_agent_md_with_custom_placeholders(self):
        """Test agent_md with various placeholders."""
        tool = SimpleTool()
        template = "{tool_name}: {output}"

        result = tool.agent_md(template, "result value")
        assert result == "simple: result value"