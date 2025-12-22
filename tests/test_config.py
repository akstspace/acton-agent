"""
Tests for ToolSet and Tool config functionality.

This module tests that config (replaces toolset_params) is:
1. Properly stored in the ToolSet model
2. Correctly injected during tool execution
3. Not visible to the LLM in tool schemas or prompts
4. Merged with user-provided parameters (with user params taking precedence)
5. Validated using Pydantic ConfigSchema when provided
"""

from unittest.mock import Mock

import pytest
from pydantic import Field

from acton_agent.agent import Agent, FunctionTool, ToolSet
from acton_agent.tools import ConfigSchema, Tool, ToolCall, ToolRegistry


class CustomToolWithConfig(Tool):
    """A custom tool that uses config."""

    def __init__(self):
        """
        Create a CustomToolWithConfig configured with a fixed name and description.

        Sets the tool's name to "custom_tool" and its description to indicate the tool uses config.
        """
        super().__init__(
            name="custom_tool",
            description="A tool that uses config",
        )

    def execute(self, parameters: dict) -> str:
        """
        Merge user-provided parameters with config and produce a brief diagnostic summary.

        Parameters:
            parameters (dict): User-supplied parameters for the tool.

        Returns:
            str: A summary string that includes the original user parameters, the config, and the resulting merged mapping.
        """
        # Merge config with parameters, with parameters taking precedence
        merged = {}
        merged.update(self.config)
        merged.update(parameters)

        # Return info about received parameters
        return f"User params: {parameters}, Config: {self.config}, Merged: {merged}"

    def get_schema(self) -> dict:
        """
        Schema describing only the tool's user-visible parameters.

        Returns:
            dict: JSON Schema for the tool's user-visible parameters; does not include config.
        """
        return {
            "type": "object",
            "properties": {
                "user_param": {
                    "type": "string",
                    "description": "A user-provided parameter",
                }
            },
        }


def test_toolset_with_config():
    """Test that ToolSet can be created with config."""
    tool = CustomToolWithConfig()

    toolset = ToolSet(
        name="test_toolset",
        description="A test toolset",
        tools=[tool],
        config={"api_key": "secret123", "endpoint": "https://api.example.com"},
    )

    assert toolset.name == "test_toolset"
    assert toolset.config == {"api_key": "secret123", "endpoint": "https://api.example.com"}
    assert len(toolset.tools) == 1


def test_config_not_in_schema():
    """Test that config is not included in the tool schema."""
    tool = CustomToolWithConfig()

    schema = tool.get_schema()

    # Schema should only have user-visible parameters
    assert "properties" in schema
    assert "user_param" in schema["properties"]
    assert "api_key" not in schema["properties"]
    assert "endpoint" not in schema["properties"]


def test_registry_tracks_config():
    """Test that registry can retrieve config for a tool."""
    registry = ToolRegistry()

    tool = CustomToolWithConfig()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        config={"api_key": "secret", "region": "us-west"},
    )

    registry.register_toolset(toolset)

    # Should be able to get config for the tool
    config = registry.get_toolset_config("custom_tool")
    assert config == {"api_key": "secret", "region": "us-west"}


def test_registry_returns_none_for_standalone_tool():
    """Test that standalone tools (not in a toolset) return None for config."""
    registry = ToolRegistry()

    tool = CustomToolWithConfig()
    registry.register(tool)

    # Standalone tool should not have config
    config = registry.get_toolset_config("custom_tool")
    assert config is None


def test_function_tool_merges_config():
    """Test that FunctionTool correctly merges config with user params."""

    def my_function(user_param: str, api_key: str | None = None) -> str:
        """
        Format a display string combining a user-provided value and an optional API key.

        Parameters:
            user_param (str): The user-provided value to include in the output.
            api_key (str | None): Optional API key to include; may be None.

        Returns:
            str: A string containing the `user_param` and the `api_key`.
        """
        return f"User: {user_param}, API Key: {api_key}"

    tool = FunctionTool(
        name="test_func",
        description="Test function",
        func=my_function,
        config={"api_key": "secret123"},
    )

    # Execute - config should be merged from self.config
    result = tool.execute({"user_param": "hello"})

    assert "User: hello" in result
    assert "API Key: secret123" in result


def test_user_params_override_config():
    """Test that user-provided parameters override config."""

    def my_function(param: str) -> str:
        """
        Format a parameter string by prefixing it with "Value: ".

        Parameters:
            param (str): The input string to format.

        Returns:
            str: The input string prefixed with "Value: ".
        """
        return f"Value: {param}"

    tool = FunctionTool(
        name="test_func",
        description="Test function",
        func=my_function,
        config={"param": "config_value"},
    )

    # User param should override config
    result = tool.execute({"param": "user_value"})

    assert "Value: user_value" in result


def test_config_not_in_prompt():
    """Test that config is not exposed in the formatted prompt."""
    registry = ToolRegistry()

    tool = CustomToolWithConfig()
    toolset = ToolSet(
        name="api_toolset",
        description="Tools for API access",
        tools=[tool],
        config={"api_key": "secret_key", "endpoint": "https://secret.api.com"},
    )

    registry.register_toolset(toolset)

    # Get formatted prompt
    prompt = registry.format_for_prompt()

    # Should include toolset name and description
    assert "api_toolset" in prompt
    assert "Tools for API access" in prompt

    # Should NOT include toolset params (they are hidden from LLM)
    assert "secret_key" not in prompt
    assert "https://secret.api.com" not in prompt
    assert "api_key" not in prompt  # param name shouldn't appear either

    # Should include user-visible schema
    assert "user_param" in prompt


def test_unregister_toolset_clears_config_mapping():
    """Test that unregistering a toolset clears the tool-to-toolset mapping."""
    registry = ToolRegistry()

    tool = CustomToolWithConfig()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        config={"key": "value"},
    )

    registry.register_toolset(toolset)
    assert registry.get_toolset_config("custom_tool") == {"key": "value"}

    # Unregister the toolset
    registry.unregister_toolset("test_toolset")

    # Tool should no longer have config
    assert registry.get_toolset_config("custom_tool") is None
    # Tool should also be removed
    assert not registry.has_tool("custom_tool")


def test_clear_registry_clears_config_mapping():
    """Test that clearing the registry also clears the config mapping."""
    registry = ToolRegistry()

    tool = CustomToolWithConfig()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        config={"key": "value"},
    )

    registry.register_toolset(toolset)
    assert registry.get_toolset_config("custom_tool") == {"key": "value"}

    # Clear the registry
    registry.clear()

    # Everything should be gone
    assert registry.get_toolset_config("custom_tool") is None
    assert not registry.has_tool("custom_tool")
    assert len(registry.list_toolsets()) == 0


def test_config_with_agent_execution(mock_llm_client):
    """Test that agent properly injects config during tool execution."""

    received_params = {}

    def capture_params(user_param: str, api_key: str | None = None, endpoint: str | None = None) -> str:
        """
        Record received parameters into the shared `received_params` mapping and return a short confirmation message.

        Parameters:
            user_param (str): User-provided parameter value; stored in `received_params["user_param"]`.
            api_key (str | None): Optional API key; stored in `received_params["api_key"]` when provided.
            endpoint (str | None): Optional endpoint URL; stored in `received_params["endpoint"]` when provided.

        Returns:
            str: Confirmation message that includes the supplied `user_param`.
        """
        received_params["user_param"] = user_param
        received_params["api_key"] = api_key
        received_params["endpoint"] = endpoint
        return f"Executed with user_param={user_param}"

    tool = FunctionTool(
        name="api_call",
        description="Make an API call",
        func=capture_params,
        schema={
            "type": "object",
            "properties": {
                "user_param": {"type": "string"},
            },
            "required": ["user_param"],
        },
    )

    toolset = ToolSet(
        name="api_toolset",
        description="API tools",
        tools=[tool],
        config={
            "api_key": "hidden_key_123",
            "endpoint": "https://api.example.com",
        },
    )

    # Create agent and register toolset
    agent = Agent(llm_client=mock_llm_client)
    agent.register_toolset(toolset)

    # Manually execute the tool (simulating agent execution)
    tool_calls = [
        ToolCall(
            id="call_1",
            tool_name="api_call",
            parameters={"user_param": "test_value"},
        )
    ]

    results = agent._execute_tool_calls(tool_calls)

    # Verify the tool received both user params and config
    assert received_params["user_param"] == "test_value"
    assert received_params["api_key"] == "hidden_key_123"
    assert received_params["endpoint"] == "https://api.example.com"

    # Verify the result was successful
    assert len(results) == 1
    assert results[0].success
    assert "test_value" in results[0].result


@pytest.fixture
def mock_llm_client():
    """
    Create a mock LLM client configured for tests.

    Returns:
        mock_client (unittest.mock.Mock): Mock client whose `create_completion` method returns a response with content "Mock response" (`{"choices": [{"message": {"content": "Mock response"}}]}`).
    """
    mock_client = Mock()
    mock_client.create_completion.return_value = {"choices": [{"message": {"content": "Mock response"}}]}
    return mock_client


def test_empty_config():
    """Test that ToolSet works with empty config."""
    tool = CustomToolWithConfig()

    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        # config defaults to empty dict
    )

    assert toolset.config == {}

    registry = ToolRegistry()
    registry.register_toolset(toolset)

    config = registry.get_toolset_config("custom_tool")
    assert config == {}


def test_multiple_toolsets_different_config():
    """Test that different toolsets can have different config for similar tools."""

    def api_function(user_input: str, api_key: str | None = None) -> str:
        """
        Indicates which API key was provided.

        Parameters:
            user_input (str): User-provided input (not used in the returned string).
            api_key (str | None): Optional API key; its value is included in the returned message.

        Returns:
            str: A message of the form "Called with key: {api_key}" showing the provided `api_key`.
        """
        return f"Called with key: {api_key}"

    tool1 = FunctionTool(
        name="api_tool_1",
        description="API tool 1",
        func=api_function,
        schema={"type": "object", "properties": {"user_input": {"type": "string"}}},
    )

    tool2 = FunctionTool(
        name="api_tool_2",
        description="API tool 2",
        func=api_function,
        schema={"type": "object", "properties": {"user_input": {"type": "string"}}},
    )

    toolset1 = ToolSet(
        name="prod_api",
        description="Production API",
        tools=[tool1],
        config={"api_key": "prod_key_123"},
    )

    toolset2 = ToolSet(
        name="dev_api",
        description="Development API",
        tools=[tool2],
        config={"api_key": "dev_key_456"},
    )

    registry = ToolRegistry()
    registry.register_toolset(toolset1)
    registry.register_toolset(toolset2)

    # Each tool should have its own config
    config1 = registry.get_toolset_config("api_tool_1")
    config2 = registry.get_toolset_config("api_tool_2")

    assert config1 == {"api_key": "prod_key_123"}
    assert config2 == {"api_key": "dev_key_456"}


def test_config_schema_validation():
    """Test that config_schema validates config properly."""

    class MyConfigSchema(ConfigSchema):
        api_key: str = Field(..., description="API key for authentication")
        timeout: int = Field(default=30, description="Request timeout in seconds")

    # Valid config should work
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[],
        config={"api_key": "secret123", "timeout": 60},
        config_schema=MyConfigSchema,
    )

    assert toolset.config == {"api_key": "secret123", "timeout": 60}

    # Missing required field should fail
    with pytest.raises(ValueError, match="config validation failed"):
        ToolSet(
            name="test_toolset",
            description="Test",
            tools=[],
            config={"timeout": 60},  # missing required api_key
            config_schema=MyConfigSchema,
        )
