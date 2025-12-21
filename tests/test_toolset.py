"""
Tests for ToolSet functionality.
"""

import pytest

from acton_agent.agent import FunctionTool, ToolSet
from acton_agent.agent.tools import ToolRegistry


def sample_function_1(param1: str) -> str:
    """
    Create a result string that includes the provided parameter.

    Parameters:
        param1 (str): Input text to include in the returned message.

    Returns:
        str: A message string containing `param1`.
    """
    return f"Result from function 1: {param1}"


def sample_function_2(param2: int) -> str:
    """
    Builds a formatted message that includes the provided integer.

    Parameters:
        param2 (int): Integer value to include in the returned message.

    Returns:
        str: A string in the form "Result from function 2: {param2}" where {param2} is the provided integer.
    """
    return f"Result from function 2: {param2}"


@pytest.fixture
def sample_tools():
    """
    Constructs two sample FunctionTool instances used by the test suite.

    Each tool is populated with a name, description, callable, and a JSON schema describing its required parameter.

    Returns:
        list[FunctionTool]: A list containing two tools:
            - "tool1": requires "param1" (string)
            - "tool2": requires "param2" (integer)
    """
    tool1 = FunctionTool(
        name="tool1",
        description="First test tool",
        func=sample_function_1,
        schema={
            "type": "object",
            "properties": {"param1": {"type": "string"}},
            "required": ["param1"],
        },
    )

    tool2 = FunctionTool(
        name="tool2",
        description="Second test tool",
        func=sample_function_2,
        schema={
            "type": "object",
            "properties": {"param2": {"type": "integer"}},
            "required": ["param2"],
        },
    )

    return [tool1, tool2]


@pytest.fixture
def sample_toolset(sample_tools):
    """
    Constructs a ToolSet named "test_toolset" containing the provided tools.

    Parameters:
        sample_tools (list): List of tool instances (typically FunctionTool) to include in the created ToolSet.

    Returns:
        ToolSet: A ToolSet instance named "test_toolset" with the given tools and description "A test toolset containing sample tools".
    """
    return ToolSet(
        name="test_toolset",
        description="A test toolset containing sample tools",
        tools=sample_tools,
    )


def test_toolset_creation(sample_toolset, sample_tools):
    """Test that a ToolSet can be created correctly."""
    assert sample_toolset.name == "test_toolset"
    assert sample_toolset.description == "A test toolset containing sample tools"
    assert len(sample_toolset.tools) == 2
    assert sample_toolset.tools == sample_tools


def test_register_toolset(sample_toolset):
    """Test registering a toolset in the registry."""
    registry = ToolRegistry()

    # Register the toolset
    registry.register_toolset(sample_toolset)

    # Check that the toolset is registered
    assert "test_toolset" in registry.list_toolsets()

    # Check that all tools are registered
    assert "tool1" in registry.list_tool_names()
    assert "tool2" in registry.list_tool_names()

    # Verify tools can be retrieved
    tool1 = registry.get("tool1")
    tool2 = registry.get("tool2")
    assert tool1 is not None
    assert tool2 is not None
    assert tool1.name == "tool1"
    assert tool2.name == "tool2"


def test_unregister_toolset(sample_toolset):
    """Test unregistering a toolset removes all its tools."""
    registry = ToolRegistry()
    registry.register_toolset(sample_toolset)

    # Verify toolset and tools are registered
    assert "test_toolset" in registry.list_toolsets()
    assert "tool1" in registry.list_tool_names()
    assert "tool2" in registry.list_tool_names()

    # Unregister the toolset
    registry.unregister_toolset("test_toolset")

    # Verify toolset is removed
    assert "test_toolset" not in registry.list_toolsets()

    # Verify all tools are removed
    assert "tool1" not in registry.list_tool_names()
    assert "tool2" not in registry.list_tool_names()


def test_unregister_nonexistent_toolset():
    """Test that unregistering a non-existent toolset raises an error."""
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="ToolSet 'nonexistent' not found"):
        registry.unregister_toolset("nonexistent")


def test_format_for_prompt_with_toolset(sample_toolset):
    """Test that toolsets are formatted correctly in prompts."""
    registry = ToolRegistry()
    registry.register_toolset(sample_toolset)

    prompt = registry.format_for_prompt()

    # Check that toolset information is included
    assert "AVAILABLE TOOLSETS:" in prompt
    assert "test_toolset" in prompt
    assert "A test toolset containing sample tools" in prompt

    # Check that tools are listed
    assert "AVAILABLE TOOLS:" in prompt
    assert "tool1" in prompt
    assert "tool2" in prompt
    assert "First test tool" in prompt
    assert "Second test tool" in prompt


def test_format_for_prompt_with_toolset_and_standalone_tools(sample_toolset):
    """Test formatting when there are both toolsets and standalone tools."""
    registry = ToolRegistry()

    # Register toolset
    registry.register_toolset(sample_toolset)

    # Register a standalone tool
    standalone_tool = FunctionTool(
        name="standalone",
        description="A standalone tool",
        func=lambda x: x,
        schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    )
    registry.register(standalone_tool)

    prompt = registry.format_for_prompt()

    # Check toolset section
    assert "AVAILABLE TOOLSETS:" in prompt
    assert "test_toolset" in prompt

    # Check that standalone tools are separated
    assert "Standalone Tools" in prompt
    assert "standalone" in prompt
    assert "A standalone tool" in prompt


def test_overwrite_toolset(sample_toolset, sample_tools):
    """Test that registering a toolset with the same name overwrites the previous one."""
    registry = ToolRegistry()

    # Register initial toolset
    registry.register_toolset(sample_toolset)
    assert len(registry.list_toolsets()) == 1

    # Create a new toolset with the same name but different tools
    new_tool = FunctionTool(
        name="new_tool",
        description="New tool",
        func=lambda: "new",
        schema={"type": "object", "properties": {}},
    )

    new_toolset = ToolSet(
        name="test_toolset",
        description="Updated toolset",
        tools=[new_tool],
    )

    # Register the new toolset (should overwrite)
    registry.register_toolset(new_toolset)

    # Should still have one toolset
    assert len(registry.list_toolsets()) == 1

    # The description should be updated
    prompt = registry.format_for_prompt()
    assert "Updated toolset" in prompt


def test_empty_toolset():
    """Test creating an empty toolset."""
    empty_toolset = ToolSet(
        name="empty",
        description="Empty toolset",
        tools=[],
    )

    assert empty_toolset.name == "empty"
    assert len(empty_toolset.tools) == 0

    # Register it
    registry = ToolRegistry()
    registry.register_toolset(empty_toolset)

    assert "empty" in registry.list_toolsets()
