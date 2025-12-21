"""
Tests for ToolSet parameters functionality.

This module tests that toolset-level parameters are:
1. Properly stored in the ToolSet model
2. Correctly injected during tool execution
3. Not visible to the LLM in tool schemas or prompts
4. Merged with user-provided parameters (with user params taking precedence)
"""

from unittest.mock import Mock

import pytest

from acton_agent.agent import Agent, FunctionTool, ToolSet
from acton_agent.tools import Tool, ToolCall, ToolRegistry


class CustomToolWithParams(Tool):
    """A custom tool that uses toolset parameters."""

    def __init__(self):
        """
        Create a CustomToolWithParams configured with a fixed name and description.
        
        Sets the tool's name to "custom_tool" and its description to indicate the tool uses toolset parameters.
        """
        super().__init__(
            name="custom_tool",
            description="A tool that uses toolset parameters",
        )

    def execute(self, parameters: dict, toolset_params: dict | None = None) -> str:
        """
        Merge user-provided parameters with optional toolset parameters and produce a brief diagnostic summary.
        
        Parameters:
            parameters (dict): User-supplied parameters for the tool.
            toolset_params (dict | None): Optional toolset-level parameters to be merged; user parameters take precedence when keys conflict.
        
        Returns:
            str: A summary string that includes the original user parameters, the provided toolset parameters, and the resulting merged mapping.
        """
        # Merge toolset_params with parameters, with parameters taking precedence
        merged = {}
        if toolset_params:
            merged.update(toolset_params)
        merged.update(parameters)

        # Return info about received parameters
        return f"User params: {parameters}, Toolset params: {toolset_params}, Merged: {merged}"

    def get_schema(self) -> dict:
        """
        Schema describing only the tool's user-visible parameters.
        
        Returns:
            dict: JSON Schema for the tool's user-visible parameters; does not include toolset parameters.
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


def test_toolset_with_params():
    """Test that ToolSet can be created with toolset_params."""
    tool = CustomToolWithParams()

    toolset = ToolSet(
        name="test_toolset",
        description="A test toolset",
        tools=[tool],
        toolset_params={"api_key": "secret123", "endpoint": "https://api.example.com"},
    )

    assert toolset.name == "test_toolset"
    assert toolset.toolset_params == {"api_key": "secret123", "endpoint": "https://api.example.com"}
    assert len(toolset.tools) == 1


def test_toolset_params_not_in_schema():
    """Test that toolset params are not included in the tool schema."""
    tool = CustomToolWithParams()

    schema = tool.get_schema()

    # Schema should only have user-visible parameters
    assert "properties" in schema
    assert "user_param" in schema["properties"]
    assert "api_key" not in schema["properties"]
    assert "endpoint" not in schema["properties"]


def test_registry_tracks_toolset_params():
    """Test that registry can retrieve toolset params for a tool."""
    registry = ToolRegistry()

    tool = CustomToolWithParams()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        toolset_params={"api_key": "secret", "region": "us-west"},
    )

    registry.register_toolset(toolset)

    # Should be able to get toolset params for the tool
    params = registry.get_toolset_params("custom_tool")
    assert params == {"api_key": "secret", "region": "us-west"}


def test_registry_returns_none_for_standalone_tool():
    """Test that standalone tools (not in a toolset) return None for toolset params."""
    registry = ToolRegistry()

    tool = CustomToolWithParams()
    registry.register(tool)

    # Standalone tool should not have toolset params
    params = registry.get_toolset_params("custom_tool")
    assert params is None


def test_function_tool_merges_toolset_params():
    """Test that FunctionTool correctly merges toolset params with user params."""

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
        schema={
            "type": "object",
            "properties": {
                "user_param": {"type": "string"},
            },
            "required": ["user_param"],
        },
    )

    # Execute with toolset params
    result = tool.execute({"user_param": "hello"}, {"api_key": "secret123"})

    assert "User: hello" in result
    assert "API Key: secret123" in result


def test_user_params_override_toolset_params():
    """Test that user-provided parameters override toolset parameters."""

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
        schema={
            "type": "object",
            "properties": {
                "param": {"type": "string"},
            },
        },
    )

    # User param should override toolset param
    result = tool.execute({"param": "user_value"}, {"param": "toolset_value"})

    assert "Value: user_value" in result


def test_toolset_params_not_in_prompt():
    """Test that toolset params are not exposed in the formatted prompt."""
    registry = ToolRegistry()

    tool = CustomToolWithParams()
    toolset = ToolSet(
        name="api_toolset",
        description="Tools for API access",
        tools=[tool],
        toolset_params={"api_key": "secret_key", "endpoint": "https://secret.api.com"},
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


def test_unregister_toolset_clears_params_mapping():
    """Test that unregistering a toolset clears the tool-to-toolset mapping."""
    registry = ToolRegistry()

    tool = CustomToolWithParams()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        toolset_params={"key": "value"},
    )

    registry.register_toolset(toolset)
    assert registry.get_toolset_params("custom_tool") == {"key": "value"}

    # Unregister the toolset
    registry.unregister_toolset("test_toolset")

    # Tool should no longer have toolset params
    assert registry.get_toolset_params("custom_tool") is None
    # Tool should also be removed
    assert not registry.has_tool("custom_tool")


def test_clear_registry_clears_params_mapping():
    """Test that clearing the registry also clears the params mapping."""
    registry = ToolRegistry()

    tool = CustomToolWithParams()
    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        toolset_params={"key": "value"},
    )

    registry.register_toolset(toolset)
    assert registry.get_toolset_params("custom_tool") == {"key": "value"}

    # Clear the registry
    registry.clear()

    # Everything should be gone
    assert registry.get_toolset_params("custom_tool") is None
    assert not registry.has_tool("custom_tool")
    assert len(registry.list_toolsets()) == 0


def test_toolset_params_with_agent_execution(mock_llm_client):
    """Test that agent properly injects toolset params during tool execution."""

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
        toolset_params={
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

    # Verify the tool received both user params and toolset params
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


def test_empty_toolset_params():
    """Test that ToolSet works with empty toolset_params."""
    tool = CustomToolWithParams()

    toolset = ToolSet(
        name="test_toolset",
        description="Test",
        tools=[tool],
        # toolset_params defaults to empty dict
    )

    assert toolset.toolset_params == {}

    registry = ToolRegistry()
    registry.register_toolset(toolset)

    params = registry.get_toolset_params("custom_tool")
    assert params == {}


def test_multiple_toolsets_different_params():
    """Test that different toolsets can have different params for similar tools."""

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
        toolset_params={"api_key": "prod_key_123"},
    )

    toolset2 = ToolSet(
        name="dev_api",
        description="Development API",
        tools=[tool2],
        toolset_params={"api_key": "dev_key_456"},
    )

    registry = ToolRegistry()
    registry.register_toolset(toolset1)
    registry.register_toolset(toolset2)

    # Each tool should have its own toolset params
    params1 = registry.get_toolset_params("api_tool_1")
    params2 = registry.get_toolset_params("api_tool_2")

    assert params1 == {"api_key": "prod_key_123"}
    assert params2 == {"api_key": "dev_key_456"}