"""
Requests Tool for making HTTP API calls.

This module provides a Tool implementation for making HTTP requests
to APIs based on OpenAPI-like specifications.
"""

import json
from typing import Any, Dict, List, Literal, Optional

import requests
from loguru import logger

from ..agent.exceptions import ToolExecutionError
from ..agent.tools import Tool


class RequestsTool(Tool):
    """
    A tool for making HTTP requests to APIs.

    This tool allows the agent to call HTTP endpoints with specified methods,
    headers, query parameters, and body data.

    Example:
        ```python
        # Create a tool for a specific API endpoint
        tool = RequestsTool(
            name="get_weather",
            description="Get current weather for a city",
            method="GET",
            url_template="https://api.weather.com/v1/current",
            query_params_schema={
                "city": {"type": "string", "description": "City name", "required": True},
                "units": {"type": "string", "description": "Temperature units", "enum": ["celsius", "fahrenheit"]}
            }
        )

        # Execute the tool
        result = tool.execute({"city": "London", "units": "celsius"})
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        url_template: str = "",
        headers: Optional[Dict[str, str]] = None,
        query_params_schema: Optional[Dict[str, Any]] = None,
        body_schema: Optional[Dict[str, Any]] = None,
        path_params: Optional[List[str]] = None,
        timeout: int = 30,
        auth: Optional[tuple] = None,
    ):
        """
        Initialize the RequestsTool.

        Args:
            name: Unique name for the tool
            description: Human-readable description of what the API does
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            url_template: URL template with optional {param} placeholders
            headers: Default headers to include in requests
            query_params_schema: Schema defining query parameters
            body_schema: Schema defining request body structure (for POST/PUT/PATCH)
            path_params: List of path parameter names in url_template
            timeout: Request timeout in seconds
            auth: Optional tuple of (username, password) for basic auth
        """
        super().__init__(name, description)
        self.method = method.upper()
        self.url_template = url_template
        self.headers = headers or {}
        self.query_params_schema = query_params_schema or {}
        self.body_schema = body_schema or {}
        self.path_params = path_params or []
        self.timeout = timeout
        self.auth = auth

    def execute(self, parameters: Dict[str, Any]) -> str:
        """
        Execute the HTTP request with given parameters.

        Args:
            parameters: Dictionary containing path params, query params, and/or body data

        Returns:
            JSON response as a string

        Raises:
            ToolExecutionError: If the request fails
        """
        try:
            # Build the URL with path parameters
            url = self.url_template
            path_params = {}
            for param in self.path_params:
                if param in parameters:
                    path_params[param] = parameters[param]

            if path_params:
                url = url.format(**path_params)

            # Separate query params and body data
            query_params = {}
            body_data = {}

            for key, value in parameters.items():
                if key in self.path_params:
                    continue  # Already used for URL
                elif key in self.query_params_schema:
                    query_params[key] = value
                elif key in self.body_schema.get("properties", {}):
                    body_data[key] = value

            # Make the request
            logger.debug(f"Making {self.method} request to {url}")
            logger.debug(f"Query params: {query_params}")
            logger.debug(f"Body data: {body_data}")

            response = requests.request(
                method=self.method,
                url=url,
                params=query_params if query_params else None,
                json=body_data if body_data and self.method in ["POST", "PUT", "PATCH"] else None,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,
            )

            # Raise exception for bad status codes
            response.raise_for_status()

            # Return response
            try:
                # Try to return JSON response
                return json.dumps(response.json(), indent=2)
            except ValueError:
                # If not JSON, return text
                return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise ToolExecutionError(self.name, str(e))
        except Exception as e:
            logger.error(f"Unexpected error in RequestsTool: {e}")
            raise ToolExecutionError(self.name, str(e))

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the tool's parameter schema.

        Combines path parameters, query parameters, and body schema into
        a single JSON Schema.

        Returns:
            JSON Schema describing all parameters
        """
        properties = {}
        required = []

        # Add path parameters
        for param in self.path_params:
            properties[param] = {"type": "string", "description": f"Path parameter: {param}"}
            required.append(param)

        # Add query parameters
        for param_name, param_schema in self.query_params_schema.items():
            properties[param_name] = param_schema.copy()
            if param_schema.get("required", False):
                required.append(param_name)
                # Remove 'required' from individual param schema
                properties[param_name].pop("required", None)

        # Add body parameters
        if self.body_schema.get("properties"):
            for param_name, param_schema in self.body_schema["properties"].items():
                properties[param_name] = param_schema.copy()

            # Add body required fields
            if "required" in self.body_schema:
                required.extend(self.body_schema["required"])

        return {
            "type": "object",
            "properties": properties,
            "required": required if required else [],
        }


def create_api_tool(
    name: str,
    description: str,
    endpoint: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    body_schema: Optional[Dict[str, Any]] = None,
) -> RequestsTool:
    """
    Factory function to quickly create a RequestsTool for an API endpoint.

    Args:
        name: Tool name
        description: Tool description
        endpoint: API endpoint URL
        method: HTTP method
        headers: Request headers
        parameters: Query parameters schema
        body_schema: Request body schema

    Returns:
        Configured RequestsTool instance

    Example:
        ```python
        tool = create_api_tool(
            name="search_repos",
            description="Search GitHub repositories",
            endpoint="https://api.github.com/search/repositories",
            method="GET",
            parameters={
                "q": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                },
                "sort": {
                    "type": "string",
                    "description": "Sort field",
                    "enum": ["stars", "forks", "updated"]
                }
            }
        )
        ```
    """
    return RequestsTool(
        name=name,
        description=description,
        method=method,
        url_template=endpoint,
        headers=headers,
        query_params_schema=parameters,
        body_schema=body_schema,
    )
