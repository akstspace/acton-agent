"""
Tool system for the AI Agent Framework.

This module provides the abstract Tool base class, ToolRegistry for managing tools,
and FunctionTool for easily wrapping Python functions as tools.
"""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger

from .exceptions import InvalidToolSchemaError, ToolNotFoundError


if TYPE_CHECKING:
    from .models import ToolSet


class Tool(ABC):
    """
    Abstract base class for tools.

    All tools must inherit from this class and implement the execute
    and get_schema methods.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the Tool with a unique name and a human-readable description.

        Parameters:
            name (str): Unique identifier for the tool used for registration and lookup.
            description (str): Short human-readable description for prompts, listings, and documentation.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> str:
        """
        Run the tool with the provided parameter mapping and return its textual output.

        Parameters:
            parameters (Dict[str, Any]): Mapping of parameter names to values to be used as inputs for execution.

        Returns:
            str: The tool's textual result.
        """

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Retrieve the JSON Schema describing this tool's parameters.

        Returns:
            schema (Dict[str, Any]): The JSON Schema object that specifies expected parameter names, types, and validation rules.
        """

    def process_output(self, output: str) -> str:
        """
        Post-process the raw output from the tool execution.

        This method can be overridden by subclasses to transform, filter, or format
        the output before it's returned to the agent. By default, it returns the
        output unchanged.

        Parameters:
            output (str): The raw output from the tool's execute method.

        Returns:
            str: The processed output (by default, unchanged).
        """
        return output

    def agent_md(self, template: str, tool_output: str) -> str:
        """
        Render the tool's values into a Markdown template by replacing placeholders.

        Parameters:
            template (str): Markdown template that may include `{tool_name}`, `{output}`, and `{description}`.
            tool_output (str): Text to substitute for the `{output}` placeholder.

        Returns:
            str: The template with `{tool_name}`, `{output}`, and `{description}` replaced by the tool's name, the provided output, and the tool's description respectively.
        """
        replacements = {
            "{tool_name}": self.name,
            "{output}": tool_output,
            "{description}": self.description,
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def __repr__(self) -> str:
        """
        Provide a concise developer-facing representation of the tool including its name.

        Returns:
            str: Representation in the form "Tool(name=<name>)".
        """
        return f"Tool(name={self.name})"


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

    def get(self, tool_name: str) -> Optional[Tool]:
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

        # Register the toolset
        self._toolsets[toolset.name] = toolset

        # Register all tools in the toolset
        for tool in toolset.tools:
            self.register(tool)

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

        # Unregister all tools in the toolset
        for tool in toolset.tools:
            if tool.name in self._tools:
                del self._tools[tool.name]

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
        """Remove all registered tools."""
        self._tools.clear()
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


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.

    This is a convenient way to create tools from existing Python functions
    without having to create a custom Tool subclass.
    """

    def __init__(self, name: str, description: str, func: Callable, schema: dict[str, Any]):
        """
        Initialize a FunctionTool that wraps a Python callable together with a JSON Schema describing its parameters.

        Parameters:
            name (str): Unique tool name.
            description (str): Human-readable description of the tool.
            func (Callable): Callable invoked when the tool is executed.
            schema (Dict[str, Any]): JSON Schema for the callable's parameters; must be a dict whose top-level `"type"` is `"object"`.

        Raises:
            InvalidToolSchemaError: If `schema` is not a dict, lacks a `"type"` field, or its `"type"` is not `"object"`.
        """
        super().__init__(name, description)
        self.func = func
        self.schema = schema

        # Validate schema
        self._validate_schema(schema)

    def _validate_schema(self, schema: dict[str, Any]) -> None:
        """
        Ensure the provided JSON Schema describes an object; raise if it is invalid.

        Parameters:
            schema (Dict[str, Any]): JSON Schema for the tool's parameters.

        Raises:
            InvalidToolSchemaError: If `schema` is not a dict, if it lacks a `"type"` field,
                or if `"type"` is not `"object"`.
        """
        if not isinstance(schema, dict):
            raise InvalidToolSchemaError(self.name, "Schema must be a dictionary")

        if "type" not in schema:
            raise InvalidToolSchemaError(self.name, "Schema must have 'type' field")

        if schema["type"] != "object":
            raise InvalidToolSchemaError(self.name, "Schema type must be 'object'")

    def execute(self, parameters: dict[str, Any]) -> str:
        """
        Run the wrapped function with the provided parameters.

        Parameters:
            parameters (Dict[str, Any]): Mapping of argument names to values passed as keyword arguments to the wrapped function.

        Returns:
            str: The wrapped function's return value as a string; non-string results are serialized to JSON.

        Raises:
            Exception: Re-raises any exception raised by the wrapped function.
        """
        try:
            result = self.func(**parameters)

            # Convert result to string
            if isinstance(result, str):
                return result
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Function tool {self.name} execution error: {e}")
            raise

    def get_schema(self) -> dict[str, Any]:
        """
        Return the JSON Schema that describes this tool's parameters.

        Returns:
            schema (Dict[str, Any]): JSON Schema describing the tool's parameters.
        """
        return self.schema
