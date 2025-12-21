"""
Tool models for the AI Agent Framework.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSet(BaseModel):
    """
    Represents a collection of related tools with a shared description.

    ToolSets allow grouping related tools together and providing a general
    description that applies to the entire group. This is useful for organizing
    tools by domain or functionality.

    Attributes:
        name: Unique name for the toolset
        description: General description of what this group of tools can do
        tools: List of Tool instances in this toolset
        toolset_params: Hidden parameters automatically passed to tools during execution (not visible to LLM)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name for the toolset")
    description: str = Field(..., description="General description of what this group of tools can do")
    tools: list[Any] = Field(default_factory=list, description="List of Tool instances in this toolset")
    toolset_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden parameters automatically passed to all tools in this toolset during execution (not visible to LLM)",
    )


class ToolCall(BaseModel):
    """
    Represents a single tool call request.

    Attributes:
        id: Unique identifier for this tool call
        tool_name: Name of the tool to invoke
        parameters: Dictionary of parameters to pass to the tool
    """

    id: str = Field(..., description="Unique identifier for this tool call")
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolResult(BaseModel):
    """
    Result from executing a tool.

    Attributes:
        tool_call_id: ID of the tool call this result is for
        tool_name: Name of the tool that was executed
        result: Result string from the tool execution
        error: Error message if execution failed
    """

    tool_call_id: str = Field(..., description="ID of the tool call this result is for")
    tool_name: str = Field(..., description="Name of the tool that was called")
    result: str = Field(..., description="Result from the tool execution")
    error: str | None = Field(None, description="Error message if tool execution failed")

    @property
    def success(self) -> bool:
        """
        Indicates whether the tool execution completed without an error.
        
        Returns:
            True if the tool produced no error, False otherwise.
        """
        return self.error is None