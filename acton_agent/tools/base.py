"""
Abstract base class for tools in the AI Agent Framework.
"""

from abc import ABC, abstractmethod
from typing import Any


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
    def execute(self, parameters: dict[str, Any], toolset_params: dict[str, Any] | None = None) -> str:
        """
        Execute the tool using the provided parameters.
        
        Parameters:
            parameters (dict[str, Any]): Mapping of parameter names to values used for execution.
            toolset_params (dict[str, Any] | None): Optional hidden parameters injected by the ToolSet; not exposed to the LLM but available to the implementation.
        
        Returns:
            str: The textual output produced by the tool.
        """

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Provide a JSON Schema describing the tool's parameters.
        
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