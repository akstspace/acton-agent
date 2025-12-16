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
        Create a Tool instance identified by a unique name and a human-readable description.

        Parameters:
            name (str): Unique identifier for the tool used in registration and lookup.
            description (str): Brief human-readable description for prompts, listings, and documentation.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> str:
        """
        Execute the tool using the provided parameters and return its textual result.

        Parameters:
            parameters (Dict[str, Any]): Mapping of parameter names to values used during execution.

        Returns:
            str: The tool's result as a string.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Retrieve the JSON Schema that describes this tool's parameters.

        Returns:
            schema (Dict[str, Any]): A JSON Schema object describing expected parameter names, types, and validation rules.
        """
        pass

    def agent_md(self, template: str, tool_output: str) -> str:
        """
        Format the tool's output into a markdown string using a template.

        Supports replacing the placeholders `{tool_name}`, `{output}`, and `{description}` in the template.

        Parameters:
            template (str): Markdown template containing placeholders to replace.
            tool_output (str): Output produced by the tool to insert into the template.

        Returns:
            formatted_markdown (str): The template with placeholders replaced by the tool's values.
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
        Check whether a tool with the given name is registered in the registry.

        Returns:
            `True` if a tool with `tool_name` is registered, `False` otherwise.
        """
        return tool_name in self._tools

    def format_for_prompt(self) -> str:
        """
        Builds a human-readable text block describing all registered tools for inclusion in an LLM prompt.

        Each tool entry includes the tool's name, description, and its JSON schema (pretty-printed) when available.

        Returns:
            A formatted string listing available tools, or "No tools available." if the registry is empty.
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
        Get the number of registered tools.

        Returns:
            count (int): Number of tools currently registered in the registry.
        """
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """
        Check whether a tool name is registered in the registry.

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
        Create a FunctionTool that wraps a Python callable and its JSON Schema.

        Store the provided function and schema on the instance and validate that the schema is a dictionary with a top-level "type" equal to "object".

        Parameters:
            name (str): Unique name for the tool.
            description (str): Human-readable description of the tool.
            func (Callable): Python callable to be invoked when the tool is executed.
            schema (Dict[str, Any]): JSON Schema describing the function's parameters; must be a dict with `"type": "object"`.

        Raises:
            InvalidToolSchemaError: If `schema` is not a dict, missing a `"type"` field, or its `"type"` is not `"object"`.
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
        Invoke the wrapped function with the given parameters and return its result as a string.

        Parameters:
            parameters (Dict[str, Any]): Mapping of argument names to values passed as keyword arguments to the wrapped function.

        Returns:
            result_str (str): The wrapped function's return value converted to a string; non-string results are serialized to JSON.

        Raises:
            Exception: Propagates any exception raised by the wrapped function.
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
        Get the JSON Schema describing the tool's parameters.

        Returns:
            schema (Dict[str, Any]): JSON Schema for the function's parameters.
        """
        return self.schema
