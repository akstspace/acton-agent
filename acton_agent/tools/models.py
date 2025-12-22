"""
Tool models for the AI Agent Framework.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConfigSchema(BaseModel):
    """
    Base class for tool and toolset configuration schemas.

    This class should be subclassed to define specific configuration
    requirements for tools or toolsets. Configuration is used for
    injecting runtime values like API keys, credentials, or other
    settings that should not be exposed to the LLM.

    Example:
        ```python
        class MyToolConfig(ConfigSchema):
            api_key: str = Field(..., description="API key for authentication")
            timeout: int = Field(default=30, description="Request timeout in seconds")
        ```
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class ToolInputSchema(BaseModel):
    """
    Base class for tool input parameter schemas.

    This class should be subclassed to define the input parameters
    that a tool expects from the LLM. These parameters will be validated
    before the tool is executed.

    Example:
        ```python
        class CalculatorInput(ToolInputSchema):
            operation: Literal["add", "subtract", "multiply", "divide"]
            a: float = Field(..., description="First number")
            b: float = Field(..., description="Second number")
        ```
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ToolSet(BaseModel):
    """
    Represents a collection of related tools with a shared description and configuration.

    ToolSets allow grouping related tools together and providing a general
    description that applies to the entire group. Configuration should be set
    using the update_config() method with a config_schema.

    Attributes:
        name: Unique name for the toolset
        description: General description of what this group of tools can do
        tools: List of Tool instances in this toolset
        config_schema: Pydantic model class defining the configuration schema
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = Field(..., description="Unique name for the toolset")
    description: str = Field(..., description="General description of what this group of tools can do")
    tools: list[Any] = Field(default_factory=list, description="List of Tool instances in this toolset")
    config_schema: type[ConfigSchema] | None = Field(
        default=None,
        description="Optional Pydantic model class defining the required and optional configuration parameters for this toolset",
    )

    def __init__(self, **data):
        """Initialize ToolSet and set config to empty dict."""
        super().__init__(**data)
        # Initialize config as an instance attribute (not a field)
        object.__setattr__(self, "config", {})

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update the configuration and validate it against the config schema.
        Also updates the configuration for all tools in this toolset by merging
        the toolset config with each tool's existing config (tool config takes precedence).

        Parameters:
            config: Configuration values to update

        Raises:
            ValueError: If config schema is not configured or if config validation fails
        """
        if self.config_schema is None:
            raise ValueError(
                f"ToolSet '{self.name}' does not have a config schema configured. "
                "Please provide a config_schema when creating the ToolSet to enable configuration."
            )

        # Validate the config against the schema
        try:
            self.config_schema(**config)
        except Exception as e:
            raise ValueError(f"ToolSet config validation failed: {e}") from e

        # Get required fields from the config schema
        required_fields = set()
        for field_name, field_info in self.config_schema.model_fields.items():
            if field_info.is_required():
                required_fields.add(field_name)

        # Check that all required fields are in the config
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(
                f"ToolSet '{self.name}' is missing required config fields: {missing_fields}. "
                f"These fields are required by the config schema."
            )

        # Store config as instance attribute
        object.__setattr__(self, "config", config)

        # Update config for all tools in this toolset
        # Merge toolset config with tool config (tool config takes precedence)
        for tool in self.tools:
            merged_config = {}
            merged_config.update(config)
            merged_config.update(tool.config)

            # If tool has a config_schema, use update_config for validation
            # Otherwise, set config directly
            if tool.config_schema is not None:
                tool.update_config(merged_config)
            else:
                tool.config = merged_config


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
