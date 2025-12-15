"""
Custom exceptions for the AI Agent Framework.

This module defines specific exceptions that can be raised during
agent operations for better error handling and debugging.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class ToolNotFoundError(AgentError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {str(original_error)}")


class LLMCallError(AgentError):
    """Raised when LLM call fails."""

    def __init__(self, original_error: Exception, retry_count: int = 0):
        self.original_error = original_error
        self.retry_count = retry_count
        super().__init__(f"LLM call failed after {retry_count} retries: {str(original_error)}")


class ResponseParseError(AgentError):
    """Raised when agent response cannot be parsed."""

    def __init__(self, response_text: str, original_error: Exception):
        self.response_text = response_text
        self.original_error = original_error
        super().__init__(f"Failed to parse agent response: {str(original_error)}")


class MaxIterationsError(AgentError):
    """Raised when agent reaches maximum iterations without a final answer."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent reached maximum iterations ({max_iterations}) without producing a final answer"
        )


class InvalidToolSchemaError(AgentError):
    """Raised when a tool schema is invalid."""

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' has invalid schema: {reason}")
