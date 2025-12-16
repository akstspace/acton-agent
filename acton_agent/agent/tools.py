"""
Tool system for the AI Agent Framework.

This module provides the abstract Tool base class, ToolRegistry for managing tools,
and FunctionTool for easily wrapping Python functions as tools.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .exceptions import InvalidToolSchemaError, ToolNotFoundError


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
    def execute(self, parameters: Dict[str, Any]) -> str:
        """
        Run the tool with the provided parameter mapping and return its textual output.

        Parameters:
            parameters (Dict[str, Any]): Mapping of parameter names to values to be used as inputs for execution.

        Returns:
            str: The tool's textual result.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Retrieve the JSON Schema describing this tool's parameters.

        Returns:
            schema (Dict[str, Any]): The JSON Schema object that specifies expected parameter names, types, and validation rules.
        """
        pass

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
    Registry for managing tools.

    Provides methods to register, unregister, and retrieve tools,
    as well as format tool information for LLM prompts.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a Tool instance in the registry, overwriting any existing tool with the same name.

        Parameters:
            tool (Tool): The Tool instance to register.
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

    def list_tools(self) -> List[Tool]:
        """
        Get all registered tools.

        Returns:
            List of all registered tool instances
        """
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
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

    def format_for_prompt(self) -> str:
        """
        Builds a concise, human-readable text block that describes all registered tools for use in a prompt.

        Each tool entry contains the tool's name, description, and the tool's JSON schema pretty-printed when available.

        Returns:
            str: Formatted listing of available tools; "No tools available." when the registry is empty.
        """
        if not self._tools:
            return "No tools available."

        tools_text = "AVAILABLE TOOLS:\n\n"
        for tool in self._tools.values():
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

    def __init__(
        self, name: str, description: str, func: Callable, schema: Dict[str, Any]
    ):
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

    def _validate_schema(self, schema: Dict[str, Any]) -> None:
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

    def execute(self, parameters: Dict[str, Any]) -> str:
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
            else:
                return json.dumps(result)

        except Exception as e:
            logger.error(f"Function tool {self.name} execution error: {e}")
            raise

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the JSON Schema that describes this tool's parameters.

        Returns:
            schema (Dict[str, Any]): JSON Schema describing the tool's parameters.
        """
        return self.schema
