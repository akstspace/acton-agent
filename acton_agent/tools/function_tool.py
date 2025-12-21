"""
FunctionTool - Wraps Python functions as tools.
"""

import json
from collections.abc import Callable
from typing import Any

from loguru import logger

from ..agent.exceptions import InvalidToolSchemaError
from .base import Tool


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

    def execute(self, parameters: dict[str, Any], toolset_params: dict[str, Any] | None = None) -> str:
        """
        Run the wrapped function with the provided parameters.

        Parameters:
            parameters (Dict[str, Any]): Mapping of argument names to values passed as keyword arguments to the wrapped function.
            toolset_params (Optional[Dict[str, Any]]): Hidden parameters from the ToolSet that are automatically
                injected during execution and merged with the user-provided parameters.

        Returns:
            str: The wrapped function's return value as a string; non-string results are serialized to JSON.

        Raises:
            Exception: Re-raises any exception raised by the wrapped function.
        """
        try:
            # Merge toolset_params with parameters, with parameters taking precedence
            merged_params = {}
            if toolset_params:
                merged_params.update(toolset_params)
            merged_params.update(parameters)

            result = self.func(**merged_params)

            # Convert result to string
            if isinstance(result, str):
                return result
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Function tool {self.name} execution error: {e}")
            raise

    def get_schema(self) -> dict[str, Any]:
        """
        Get the JSON Schema describing this tool's parameters.
        
        Returns:
            dict[str, Any]: The JSON Schema that describes the tool's parameters.
        """
        return self.schema