"""
Tests for the tools module.
"""

import json

import pytest
from pydantic import Field

from acton_agent.agent.exceptions import ToolNotFoundError
from acton_agent.tools import FunctionTool, Tool, ToolRegistry
from acton_agent.tools.models import ConfigSchema, ToolInputSchema


class SimpleTool(Tool):
    """Simple test tool."""

    def __init__(self):
        """
        Create a SimpleTool with the name "simple" and the description "A simple test tool".
        """
        super().__init__(name="simple", description="A simple test tool")

    def execute(self, parameters: dict, config: dict | None = None) -> str:
        """
        Format and return a result string using the "value" entry from parameters.

        Parameters:
            parameters (dict): Mapping that may contain the key "value"; when absent, "default" is used.
            config (dict | None): Optional toolset-level parameters; ignored by this tool.

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

        tool = FunctionTool(name="add", description="Add two numbers", func=add)

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

        tool = FunctionTool(
            name="multiply",
            description="Multiply two numbers",
            func=multiply,
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

        tool = FunctionTool(name="greet", description="Greet someone", func=greet)

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

        tool = FunctionTool(name="get_info", description="Get info", func=get_info)

        result = tool.execute({"id": 123})
        result_dict = json.loads(result)
        assert result_dict["id"] == 123
        assert result_dict["status"] == "active"


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


class TestToolConfiguration:
    """Tests for Tool configuration functionality."""

    def test_tool_with_config_schema(self):
        """Test creating a tool with a config schema."""

        class MyToolConfig(ConfigSchema):
            api_key: str = Field(..., description="API key")
            timeout: int = Field(default=30, description="Timeout in seconds")

        class ConfigurableTool(Tool):
            def execute(self, parameters: dict) -> str:
                api_key = self.config.get("api_key", "")
                timeout = self.config.get("timeout", 30)
                return f"API: {api_key}, Timeout: {timeout}"

            def get_schema(self) -> dict:
                return {"type": "object", "properties": {}}

        tool = ConfigurableTool(
            name="config_tool",
            description="A tool with config",
            config_schema=MyToolConfig,
        )

        assert tool.config_schema == MyToolConfig
        assert tool.config == {}

    def test_update_config_success(self):
        """Test updating tool configuration successfully."""

        class MyToolConfig(ConfigSchema):
            api_key: str = Field(..., description="API key")
            timeout: int = Field(default=30, description="Timeout in seconds")

        class ConfigurableTool(Tool):
            def execute(self, parameters: dict) -> str:
                api_key = self.config.get("api_key", "")
                timeout = self.config.get("timeout", 30)
                return f"API: {api_key}, Timeout: {timeout}"

            def get_schema(self) -> dict:
                return {"type": "object", "properties": {}}

        tool = ConfigurableTool(
            name="config_tool",
            description="A tool with config",
            config_schema=MyToolConfig,
        )

        tool.update_config({"api_key": "secret123", "timeout": 60})

        assert tool.config["api_key"] == "secret123"
        assert tool.config["timeout"] == 60

    def test_update_config_without_schema_raises_error(self):
        """Test that updating config without schema raises error."""
        tool = SimpleTool()

        with pytest.raises(ValueError, match="does not have a config schema configured"):
            tool.update_config({"some_key": "value"})

    def test_update_config_validation_failure(self):
        """Test that invalid config raises validation error."""

        class MyToolConfig(ConfigSchema):
            api_key: str = Field(..., description="API key")
            timeout: int = Field(..., description="Timeout in seconds")

        class ConfigurableTool(Tool):
            def execute(self, parameters: dict) -> str:
                return "result"

            def get_schema(self) -> dict:
                return {"type": "object", "properties": {}}

        tool = ConfigurableTool(
            name="config_tool",
            description="A tool with config",
            config_schema=MyToolConfig,
        )

        # Missing required field 'api_key'
        with pytest.raises(ValueError, match="config validation failed"):
            tool.update_config({"timeout": 60})

    def test_tool_with_input_schema(self):
        """Test creating a tool with an input schema."""

        class MyInputSchema(ToolInputSchema):
            query: str = Field(..., description="Search query")
            limit: int = Field(default=10, description="Result limit")

        class SearchTool(Tool):
            def execute(self, parameters: dict) -> str:
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                return f"Searching for '{query}' with limit {limit}"

            def get_schema(self) -> dict:
                if self.input_schema:
                    return self.input_schema.model_json_schema()
                return {"type": "object", "properties": {}}

        tool = SearchTool(
            name="search",
            description="Search tool",
            input_schema=MyInputSchema,
        )

        assert tool.input_schema == MyInputSchema
        schema = tool.get_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]


class TestFunctionToolConfiguration:
    """Tests for FunctionTool configuration functionality."""

    def test_function_tool_with_config_schema(self):
        """Test FunctionTool with config schema."""

        class MyConfig(ConfigSchema):
            prefix: str = Field(..., description="Prefix for output")

        def greet(name: str, prefix: str = "Hello") -> str:
            return f"{prefix}, {name}!"

        tool = FunctionTool(
            name="greet",
            description="Greet someone",
            func=greet,
            config_schema=MyConfig,
        )

        assert tool.config_schema == MyConfig
        tool.update_config({"prefix": "Hi"})
        assert tool.config["prefix"] == "Hi"

    def test_function_tool_with_input_schema(self):
        """Test FunctionTool with input schema."""

        class GreetInput(ToolInputSchema):
            name: str = Field(..., description="Name of person to greet")

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = FunctionTool(
            name="greet",
            description="Greet someone",
            func=greet,
            input_schema=GreetInput,
        )

        assert tool.input_schema == GreetInput
        schema = tool.get_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_function_tool_input_validation_success(self):
        """Test that FunctionTool validates input parameters successfully."""

        class MathInput(ToolInputSchema):
            a: int = Field(..., description="First number")
            b: int = Field(..., description="Second number")

        def add(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(
            name="add",
            description="Add two numbers",
            func=add,
            input_schema=MathInput,
        )

        result = tool.execute({"a": 5, "b": 3})
        assert result == "8"

    def test_function_tool_input_validation_failure(self):
        """Test that FunctionTool raises error on invalid input."""

        class MathInput(ToolInputSchema):
            a: int = Field(..., description="First number")
            b: int = Field(..., description="Second number")

        def add(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(
            name="add",
            description="Add two numbers",
            func=add,
            input_schema=MathInput,
        )

        # Missing required parameter 'b'
        with pytest.raises(ValueError, match="Input validation failed"):
            tool.execute({"a": 5})

    def test_function_tool_config_merging(self):
        """Test that FunctionTool merges config with parameters."""

        class MyConfig(ConfigSchema):
            prefix: str = Field(..., description="Prefix for output")
            suffix: str = Field(default="!", description="Suffix for output")

        def format_text(text: str, prefix: str = "", suffix: str = "") -> str:
            return f"{prefix}{text}{suffix}"

        tool = FunctionTool(
            name="format",
            description="Format text",
            func=format_text,
            config_schema=MyConfig,
        )

        # Set config
        tool.update_config({"prefix": ">>> ", "suffix": " <<<"})

        # Parameters should take precedence over config
        result = tool.execute({"text": "Hello", "suffix": "!!!"})
        assert result == ">>> Hello!!!"

        # Config values used when not in parameters
        result2 = tool.execute({"text": "World"})
        assert result2 == ">>> World <<<"

    def test_function_tool_with_both_schemas(self):
        """Test FunctionTool with both config and input schemas."""

        class MyConfig(ConfigSchema):
            api_key: str = Field(..., description="API key")

        class MyInput(ToolInputSchema):
            query: str = Field(..., description="Search query")

        def search(query: str, api_key: str = "") -> str:
            return f"Searching '{query}' with key '{api_key}'"

        tool = FunctionTool(
            name="search",
            description="Search API",
            func=search,
            config_schema=MyConfig,
            input_schema=MyInput,
        )

        tool.update_config({"api_key": "secret123"})
        result = tool.execute({"query": "test"})
        assert result == "Searching 'test' with key 'secret123'"
