"""
Integration tests for Agent with ToolSet functionality.
"""

from unittest.mock import Mock

import pytest

from acton_agent.agent import Agent, FunctionTool, ToolSet


def sample_func_1() -> str:
    """
    Return a fixed sentinel string used by tests.

    Returns:
        result (str): The literal string "result1".
    """
    return "result1"


def sample_func_2() -> str:
    """Sample function 2."""
    return "result2"


@pytest.fixture
def mock_llm_client():
    """
    Create a Mock object that simulates an LLM client for use in tests.

    Returns:
        unittest.mock.Mock: A Mock instance representing the LLM client.
    """
    return Mock()


@pytest.fixture
def sample_toolset():
    """
    Constructs a ToolSet containing two simple FunctionTool instances for testing.

    The returned ToolSet is named "sample_toolset", has the description "A sample toolset for testing",
    and includes two FunctionTool entries:
    - "func1": wraps sample_func_1 with description "Function 1"
    - "func2": wraps sample_func_2 with description "Function 2"

    Returns:
        ToolSet: A ToolSet with the two FunctionTool objects described above.
    """
    tool1 = FunctionTool(
        name="func1",
        description="Function 1",
        func=sample_func_1,
        schema={"type": "object", "properties": {}},
    )

    tool2 = FunctionTool(
        name="func2",
        description="Function 2",
        func=sample_func_2,
        schema={"type": "object", "properties": {}},
    )

    return ToolSet(
        name="sample_toolset",
        description="A sample toolset for testing",
        tools=[tool1, tool2],
    )


def test_agent_register_toolset(mock_llm_client, sample_toolset):
    """Test that an agent can register a toolset."""
    agent = Agent(llm_client=mock_llm_client)

    # Register the toolset
    agent.register_toolset(sample_toolset)

    # Verify tools are registered
    assert "func1" in agent.list_tools()
    assert "func2" in agent.list_tools()


def test_agent_toolset_in_prompt(mock_llm_client, sample_toolset):
    """Test that toolset information appears in the agent's prompt."""
    agent = Agent(llm_client=mock_llm_client)

    # Register the toolset
    agent.register_toolset(sample_toolset)

    # Build messages and check that toolset info is included
    messages = agent._build_messages()

    # The system message should contain toolset information
    system_message = messages[0]
    assert system_message.role == "system"
    assert "AVAILABLE TOOLSETS:" in system_message.content
    assert "sample_toolset" in system_message.content
    assert "A sample toolset for testing" in system_message.content
    assert "func1" in system_message.content
    assert "func2" in system_message.content


def test_agent_register_multiple_toolsets(mock_llm_client):
    """Test registering multiple toolsets."""
    agent = Agent(llm_client=mock_llm_client)

    # Create first toolset
    toolset1 = ToolSet(
        name="toolset1",
        description="First toolset",
        tools=[
            FunctionTool(
                name="tool1",
                description="Tool 1",
                func=lambda: "1",
                schema={"type": "object", "properties": {}},
            )
        ],
    )

    # Create second toolset
    toolset2 = ToolSet(
        name="toolset2",
        description="Second toolset",
        tools=[
            FunctionTool(
                name="tool2",
                description="Tool 2",
                func=lambda: "2",
                schema={"type": "object", "properties": {}},
            )
        ],
    )

    # Register both toolsets
    agent.register_toolset(toolset1)
    agent.register_toolset(toolset2)

    # Verify all tools are registered
    tools = agent.list_tools()
    assert "tool1" in tools
    assert "tool2" in tools

    # Verify both toolsets appear in prompt
    messages = agent._build_messages()
    system_message = messages[0]
    assert "toolset1" in system_message.content
    assert "toolset2" in system_message.content


def test_agent_register_toolset_and_individual_tools(mock_llm_client, sample_toolset):
    """Test that an agent can have both toolsets and individual tools."""
    agent = Agent(llm_client=mock_llm_client)

    # Register a toolset
    agent.register_toolset(sample_toolset)

    # Register an individual tool
    standalone_tool = FunctionTool(
        name="standalone",
        description="Standalone tool",
        func=lambda: "standalone",
        schema={"type": "object", "properties": {}},
    )
    agent.register_tool(standalone_tool)

    # Verify all tools are available
    tools = agent.list_tools()
    assert "func1" in tools
    assert "func2" in tools
    assert "standalone" in tools

    # Verify prompt formatting
    messages = agent._build_messages()
    system_message = messages[0]
    assert "sample_toolset" in system_message.content
    assert "Standalone Tools" in system_message.content
    assert "standalone" in system_message.content
