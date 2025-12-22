"""
Abstract base class for tools in the AI Agent Framework.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .models import ConfigSchema, ToolInputSchema


class Tool(ABC):
    """
    Abstract base class for tools.

    All tools must inherit from this class and implement the execute
    and get_schema methods.

    Tools now support configuration through config and config_schema,
    and input validation through input_schema.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config_schema: type["ConfigSchema"] | None = None,
        input_schema: type["ToolInputSchema"] | None = None,
    ):
        """
        Initialize the Tool with a unique name, description, and optional configuration schema.

        Parameters:
            name (str): Unique identifier for the tool used for registration and lookup.
            description (str): Short human-readable description for prompts, listings, and documentation.
            config_schema (Type[ConfigSchema] | None): Optional Pydantic model class defining configuration requirements.
            input_schema (Type[ToolInputSchema] | None): Optional Pydantic model class defining input parameter requirements.
        """
        self.name = name
        self.description = description
        self.config: dict[str, Any] = {}
        self.config_schema = config_schema
        self.input_schema = input_schema

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update the configuration and validate it against the config schema.

        Parameters:
            config: Configuration values to update

        Raises:
            ValueError: If config schema is not configured or if config validation fails
        """
        if self.config_schema is None:
            raise ValueError(
                f"Tool '{self.name}' does not have a config schema configured. "
                "Please provide a config_schema when creating the Tool to enable configuration."
            )

        try:
            self.config_schema(**config)
        except Exception as e:
            raise ValueError(f"Tool '{self.name}' config validation failed: {e}") from e

        self.config = config

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> str:
        """
        Execute the tool using the provided parameters.

        The tool can access its configuration via self.config.

        Parameters:
            parameters (dict[str, Any]): Mapping of parameter names to values used for execution.

        Returns:
            str: The textual output produced by the tool.
        """

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Provide a JSON Schema describing the tool's parameters.

        If input_schema (Pydantic model) is provided, this method should return
        the JSON schema generated from that model. Otherwise, return a dict-based
        JSON Schema.

        Returns:
            dict[str, Any]: A dictionary representing a JSON Schema that describes expected parameter names, types, constraints, and validation rules.
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
