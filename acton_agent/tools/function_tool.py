"""
FunctionTool - Wraps Python functions as tools.
"""

import json
from collections.abc import Callable
from typing import Any

from loguru import logger

from .base import Tool
from .models import ConfigSchema, ToolInputSchema


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.

    This is a convenient way to create tools from existing Python functions
    without having to create a custom Tool subclass.

    Uses Pydantic-based input schemas and configuration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        config_schema: type[ConfigSchema] | None = None,
        input_schema: type[ToolInputSchema] | None = None,
    ):
        """
        Initialize a FunctionTool that wraps a Python callable.

        Parameters:
            name (str): Unique tool name.
            description (str): Human-readable description of the tool.
            func (Callable): Callable invoked when the tool is executed.
            config_schema (Type[ConfigSchema] | None): Optional Pydantic model class defining configuration requirements.
            input_schema (Type[ToolInputSchema] | None): Optional Pydantic model class defining input parameters.
        """
        # Initialize base with config_schema
        super().__init__(
            name=name,
            description=description,
            config_schema=config_schema,
            input_schema=input_schema,
        )

        self.func = func

    def execute(self, parameters: dict[str, Any]) -> str:
        """
        Run the wrapped function with the provided parameters.

        Parameters:
            parameters (dict[str, Any]): Mapping of argument names to values passed as keyword arguments to the wrapped function.

        Returns:
            str: The wrapped function's return value as a string; non-string results are serialized to JSON.

        Raises:
            Exception: Re-raises any exception raised by the wrapped function.
        """
        try:
            # Validate input parameters if input_schema is provided
            if self.input_schema is not None:
                try:
                    validated_params = self.input_schema(**parameters)
                    # Convert Pydantic model back to dict for function call
                    parameters = validated_params.model_dump()
                except Exception as e:
                    logger.error(f"Input validation failed for tool {self.name}: {e}")
                    raise ValueError(f"Input validation failed: {e}") from e

            # Merge config with parameters, with parameters taking precedence
            merged_params = {}
            merged_params.update(self.config)
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

        Returns the JSON schema generated from the Pydantic input_schema model,
        or an empty object schema if no input_schema is defined.

        Returns:
            dict[str, Any]: The JSON Schema that describes the tool's parameters.
        """
        if self.input_schema is not None:
            # Generate JSON schema from Pydantic model
            return self.input_schema.model_json_schema()
        # Return empty object schema if no schema is defined
        return {"type": "object", "properties": {}, "required": []}
