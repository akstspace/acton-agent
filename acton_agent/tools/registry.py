"""
Tool registry for managing tools and toolsets.
"""

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..agent.exceptions import ToolNotFoundError
from .base import Tool


if TYPE_CHECKING:
    from .models import ToolSet


class ToolRegistry:
    """
    Registry for managing tools and toolsets.

    Provides methods to register, unregister, and retrieve tools,
    as well as format tool information for LLM prompts.
    Supports organizing tools into toolsets with shared descriptions.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}
        self._toolsets: dict[str, ToolSet] = {}
        self._tool_to_toolset: dict[str, str] = {}  # Maps tool_name -> toolset_name

    def register(self, tool: Tool) -> None:
        """
        Register a Tool under its name in the registry, overwriting any existing registration.

        Parameters:
            tool (Tool): The Tool instance to add to the registry.
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """
        Remove a registered tool from the registry by name.

        Parameters:
                tool_name (str): Name of the tool to remove.

        Raises:
                ToolNotFoundError: If no tool with `tool_name` is registered.
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)

        del self._tools[tool_name]
        logger.info(f"Unregistered tool: {tool_name}")

    def get(self, tool_name: str) -> Tool | None:
        """
        Retrieve a registered tool by name.

        Parameters:
            tool_name (str): The name of the tool to retrieve.

        Returns:
            Optional[Tool]: The Tool instance if found, otherwise None.
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> list[Tool]:
        """
        Get all registered tools.

        Returns:
            List of all registered tool instances
        """
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """
        List the names of all registered tools.

        Returns:
            A list of registered tool names.
        """
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """
        Check whether a tool with the given name is registered.

        Returns:
            True if a tool with `tool_name` is registered, False otherwise.
        """
        return tool_name in self._tools

    def register_toolset(self, toolset: "ToolSet") -> None:
        """
        Register a ToolSet and add all tools contained in it to the registry.

        If a ToolSet with the same name already exists, it is overwritten and its tools are replaced; each tool from the provided ToolSet is registered individually.

        Parameters:
            toolset (ToolSet): The ToolSet instance whose tools should be added to the registry.
        """

        if toolset.name in self._toolsets:
            logger.warning(f"ToolSet '{toolset.name}' already registered, overwriting")

        self._toolsets[toolset.name] = toolset

        for tool in toolset.tools:
            self.register(tool)
            self._tool_to_toolset[tool.name] = toolset.name

        logger.info(f"Registered toolset: {toolset.name} with {len(toolset.tools)} tools")

    def unregister_toolset(self, toolset_name: str) -> None:
        """
        Remove a registered toolset and all its tools from the registry.

        Parameters:
            toolset_name (str): Name of the toolset to remove.

        Raises:
            ValueError: If no toolset with the given name is registered.
        """
        if toolset_name not in self._toolsets:
            raise ValueError(f"ToolSet '{toolset_name}' not found")

        toolset = self._toolsets[toolset_name]

        # Unregister all tools in the toolset and remove from mapping
        for tool in toolset.tools:
            if tool.name in self._tools:
                del self._tools[tool.name]
            if tool.name in self._tool_to_toolset:
                del self._tool_to_toolset[tool.name]

        # Remove the toolset
        del self._toolsets[toolset_name]
        logger.info(f"Unregistered toolset: {toolset_name}")

    def list_toolsets(self) -> list[str]:
        """
        Get the names of all registered toolsets.

        Returns:
            A list of registered toolset names.
        """
        return list(self._toolsets.keys())

    def get_toolset_params(self, tool_name: str) -> dict[str, Any] | None:
        """
        Get the toolset parameters for a specific tool, if it belongs to a toolset.

        Parameters:
            tool_name (str): The name of the tool to get toolset parameters for.

        Returns:
            Optional[Dict[str, Any]]: The toolset parameters if the tool belongs to a toolset, None otherwise.
        """
        toolset_name = self._tool_to_toolset.get(tool_name)
        if toolset_name:
            toolset = self._toolsets.get(toolset_name)
            if toolset:
                return toolset.toolset_params
        return None

    def format_for_prompt(self) -> str:
        """
        Builds a human-readable listing of registered toolsets and tools for inclusion in a prompt.

        Toolsets are listed first with their name, description, and contained tool names; tools are then grouped by toolset and standalone tools follow. Each tool entry includes its name, description, and the tool's JSON schema when available.

        Returns:
            str: Formatted text describing available toolsets and tools, or "No tools available." if the registry is empty.
        """
        if not self._tools and not self._toolsets:
            return "No tools available."

        tools_text = ""

        # Format toolsets first
        if self._toolsets:
            tools_text += "AVAILABLE TOOLSETS:\n\n"
            for _toolset_name, toolset in self._toolsets.items():
                tools_text += f"ToolSet: {toolset.name}\n"
                tools_text += f"Description: {toolset.description}\n"
                tools_text += f"Tools in this set: {', '.join([tool.name for tool in toolset.tools])}\n\n"

        # Format individual tools
        tools_text += "AVAILABLE TOOLS:\n\n"

        # Group tools by toolset
        toolset_tools = set()
        for toolset in self._toolsets.values():
            for tool in toolset.tools:
                toolset_tools.add(tool.name)

        # Format tools that belong to toolsets
        for toolset in self._toolsets.values():
            tools_text += f"--- Tools from {toolset.name} ---\n"
            for tool in toolset.tools:
                tools_text += f"Tool: {tool.name}\n"
                tools_text += f"Description: {tool.description}\n"

                schema = tool.get_schema()
                if schema:
                    tools_text += f"Schema: {json.dumps(schema, indent=2)}\n"

                tools_text += "\n"

        # Format standalone tools (not in any toolset)
        standalone_tools = [tool for tool in self._tools.values() if tool.name not in toolset_tools]

        if standalone_tools:
            tools_text += "--- Standalone Tools ---\n"
            for tool in standalone_tools:
                tools_text += f"Tool: {tool.name}\n"
                tools_text += f"Description: {tool.description}\n"

                schema = tool.get_schema()
                if schema:
                    tools_text += f"Schema: {json.dumps(schema, indent=2)}\n"

                tools_text += "\n"

        return tools_text

    def clear(self) -> None:
        """Remove all registered tools and toolsets."""
        self._tools.clear()
        self._toolsets.clear()
        self._tool_to_toolset.clear()
        logger.info("Cleared all tools from registry")

    def __len__(self) -> int:
        """
        Return the number of registered tools.

        Returns:
            count (int): The number of tools currently registered in the registry.
        """
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """
        Determine whether a tool name is registered in the registry.

        Returns:
            `true` if the tool name is registered, `false` otherwise.
        """
        return tool_name in self._tools
