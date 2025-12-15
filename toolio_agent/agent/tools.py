"""
Tool system for the AI Agent Framework.

This module provides the abstract Tool base class, ToolRegistry for managing tools,
and FunctionTool for easily wrapping Python functions as tools.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import json
from loguru import logger

from .exceptions import ToolNotFoundError, InvalidToolSchemaError


class Tool(ABC):
    """
    Abstract base class for tools.
    
    All tools must inherit from this class and implement the execute
    and get_schema methods.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a tool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> str:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Dictionary of parameters for the tool
            
        Returns:
            Result as a string
            
        Raises:
            Exception: If tool execution fails
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the tool's parameter schema.
        
        Returns:
            JSON Schema describing the tool's parameters
        """
        pass
    
    def agent_md(self, template: str, tool_output: str) -> str:
        """
        Format tool output using a markdown template.
        
        This method replaces variables in the template with the tool output.
        Common template variables:
        - {tool_name}: Name of the tool
        - {output}: The tool execution output
        - {description}: Tool description
        
        Args:
            template: Markdown template string with variables to replace
            tool_output: The output from tool execution
            
        Returns:
            Formatted markdown string
            
        Example:
            ```python
            template = "## {tool_name}\\n\\n{description}\\n\\n**Result:**\\n{output}"
            formatted = tool.agent_md(template, result)
            ```
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
        Register a tool.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)
        
        del self._tools[tool_name]
        logger.info(f"Unregistered tool: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
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
        Get names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self._tools
    
    def format_for_prompt(self) -> str:
        """
        Format tools for inclusion in LLM prompt.
        
        Returns:
            Formatted string describing all available tools
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
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered using 'in' operator."""
        return tool_name in self._tools


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.
    
    This is a convenient way to create tools from existing Python functions
    without having to create a custom Tool subclass.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        schema: Dict[str, Any]
    ):
        """
        Initialize a function tool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description
            func: Python function to wrap
            schema: JSON Schema for function parameters
            
        Raises:
            InvalidToolSchemaError: If schema is invalid
        """
        super().__init__(name, description)
        self.func = func
        self.schema = schema
        
        # Validate schema
        self._validate_schema(schema)
    
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate the tool schema.
        
        Args:
            schema: JSON Schema to validate
            
        Raises:
            InvalidToolSchemaError: If schema is invalid
        """
        if not isinstance(schema, dict):
            raise InvalidToolSchemaError(
                self.name, 
                "Schema must be a dictionary"
            )
        
        if "type" not in schema:
            raise InvalidToolSchemaError(
                self.name,
                "Schema must have 'type' field"
            )
        
        if schema["type"] != "object":
            raise InvalidToolSchemaError(
                self.name,
                "Schema type must be 'object'"
            )
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        """
        Execute the wrapped function.
        
        Args:
            parameters: Parameters to pass to the function
            
        Returns:
            Function result as a string
            
        Raises:
            Exception: If function execution fails
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
        Return the function's parameter schema.
        
        Returns:
            JSON Schema for function parameters
        """
        return self.schema
