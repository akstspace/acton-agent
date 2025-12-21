"""
Custom exceptions for the AI Agent Framework.

This module defines specific exceptions that can be raised during
agent operations for better error handling and debugging.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""


class ToolNotFoundError(AgentError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str):
        """
        Initialize a ToolNotFoundError for a missing tool in the registry.

        Parameters:
            tool_name (str): Name of the missing tool; stored on the exception as `tool_name` and included in the exception message.
        """
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, original_error: Exception):
        """
        Initialize the ToolExecutionError with the failing tool's name and the underlying exception.

        Parameters:
            tool_name (str): Name of the tool that failed; stored on the instance as `tool_name`.
            original_error (Exception): The original exception raised during tool execution; stored on the instance as `original_error`.

        Notes:
            The exception message includes the tool name and the original error's message.
        """
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {original_error!s}")


class LLMCallError(AgentError):
    """Raised when LLM call fails."""

    def __init__(self, original_error: Exception, retry_count: int = 0):
        """
        Initialize the exception with the underlying error and the number of retry attempts.

        Parameters:
            original_error (Exception): The original exception raised during the LLM call.
            retry_count (int): Number of retry attempts that were made (defaults to 0).
        """
        self.original_error = original_error
        self.retry_count = retry_count
        super().__init__(f"LLM call failed after {retry_count} retries: {original_error!s}")


class ResponseParseError(AgentError):
    """Raised when agent response cannot be parsed."""

    def __init__(self, response_text: str, original_error: Exception):
        """
        Create a ResponseParseError containing the raw agent response and the underlying parsing exception.

        Parameters:
            response_text (str): Raw agent response that could not be parsed.
            original_error (Exception): The parsing exception that triggered this error.

        Notes:
            The exception exposes `response_text` and `original_error` as attributes and includes the original error message in its exception text.
        """
        self.response_text = response_text
        self.original_error = original_error
        super().__init__(f"Failed to parse agent response: {original_error!s}")


class MaxIterationsError(AgentError):
    """Raised when agent reaches maximum iterations without a final answer."""

    def __init__(self, max_iterations: int):
        """
        Initialize MaxIterationsError indicating the agent reached the maximum allowed iterations.

        Parameters:
            max_iterations (int): Maximum number of iterations that was reached; stored on the exception as `max_iterations`.
        """
        self.max_iterations = max_iterations
        super().__init__(f"Agent reached maximum iterations ({max_iterations}) without producing a final answer")


class InvalidToolSchemaError(AgentError):
    """Raised when a tool schema is invalid."""

    def __init__(self, tool_name: str, reason: str):
        """
        Initialize an InvalidToolSchemaError for a tool with an invalid schema.

        Parameters:
            tool_name (str): Name of the tool with the invalid schema.
            reason (str): Explanation of why the schema is invalid.
        """
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' has invalid schema: {reason}")
