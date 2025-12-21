"""
Requests Tool for making HTTP API calls.

This module provides a Tool implementation for making HTTP requests
to APIs based on OpenAPI-like specifications.
"""

import json
import re
from typing import Any, Literal

import requests
from loguru import logger

from ..agent.exceptions import ToolExecutionError
from .base import Tool


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
        headers: dict[str, str] | None = None,
        query_params_schema: dict[str, Any] | None = None,
        body_schema: dict[str, Any] | None = None,
        path_params: list[str] | None = None,
        path_params_schema: dict[str, Any] | None = None,
        header_params_schema: dict[str, Any] | None = None,
        timeout: int = 30,
        auth: tuple | None = None,
    ):
        """
        Configure a RequestsTool for making HTTP API calls to a templated endpoint.

        Parameters:
            name: Unique tool identifier.
            description: Human-readable description of the API.
            method: HTTP method to use (GET, POST, PUT, DELETE, PATCH).
            url_template: URL template that may include `{param}` placeholders for route/path parameters.
            headers: Default headers to include with every request.
            query_params_schema: JSON Schema describing supported query parameters.
            body_schema: JSON Schema describing the request body structure for methods that send a body.
            path_params: List of path parameter names. If None, automatically extracted from url_template.
            path_params_schema: JSON Schema describing path parameters (name -> schema mapping).
            header_params_schema: JSON Schema describing header parameters (name -> schema mapping).
            timeout: Request timeout in seconds.
            auth: Optional (username, password) tuple for basic authentication.
        """
        super().__init__(name, description)
        self.method = method.upper()
        self.url_template = url_template
        self.headers = headers or {}
        self.query_params_schema = query_params_schema or {}
        self.body_schema = body_schema or {}
        self.path_params_schema = path_params_schema or {}
        self.header_params_schema = header_params_schema or {}

        # Auto-extract route/path parameters from URL template if not provided
        if path_params is None:
            # Find all {parameter_name} patterns in the URL template
            self.path_params = re.findall(r"\{([^}]+)\}", url_template)
        else:
            self.path_params = path_params

        self.timeout = timeout
        self.auth = auth

    def execute(self, parameters: dict[str, Any], toolset_params: dict[str, Any] | None = None) -> str:
        """
        Execute the configured HTTP request using the provided parameters and return the response body.

        Parameters:
            parameters (Dict[str, Any]): Mapping of parameter names to values. Values matching configured route/path parameters are substituted into the URL template; values matching the query parameters schema are sent as query string parameters; values matching header parameters schema are sent as headers; values matching the body schema's properties are sent as a JSON body for POST/PUT/PATCH requests.
            toolset_params (Optional[Dict[str, Any]]): Hidden parameters from the ToolSet that are automatically
                injected during execution and merged with the user-provided parameters.

        Returns:
            str: Pretty-printed JSON string if the response is JSON, otherwise the raw response text.

        Raises:
            ToolExecutionError: If the HTTP request fails or an unexpected error occurs while executing the request.
        """
        try:
            # Merge toolset_params with parameters, with parameters taking precedence
            merged_params = {}
            if toolset_params:
                merged_params.update(toolset_params)
            merged_params.update(parameters)

            # Build the URL with route/path parameters
            url = self.url_template
            path_params = {}
            for param in self.path_params:
                if param in merged_params:
                    path_params[param] = merged_params[param]

            if path_params:
                url = url.format(**path_params)

            # Separate query params, header params, and body data
            query_params = {}
            header_params = {}
            body_data = {}

            for key, value in merged_params.items():
                if key in self.path_params:
                    continue  # Already used for URL
                if key in self.query_params_schema:
                    query_params[key] = value
                elif key in self.header_params_schema:
                    header_params[key] = value
                elif key in self.body_schema.get("properties", {}):
                    body_data[key] = value

            # Merge headers: base headers + dynamic header parameters
            request_headers = self.headers.copy()
            request_headers.update(header_params)

            # Make the request
            logger.debug(f"Making {self.method} request to {url}")
            logger.debug(f"Query params: {query_params}")
            logger.debug(f"Header params: {header_params}")
            logger.debug(f"Body data: {body_data}")

            response = requests.request(
                method=self.method,
                url=url,
                params=query_params if query_params else None,
                json=body_data if body_data and self.method in ["POST", "PUT", "PATCH"] else None,
                headers=request_headers,
                auth=self.auth,
                timeout=self.timeout,
            )

            # Raise exception for bad status codes
            response.raise_for_status()

            # Get raw response
            try:
                # Try to return JSON response
                raw_output = json.dumps(response.json(), indent=2)
            except ValueError:
                # If not JSON, return text
                raw_output = response.text

            # Process output through post-processing hook
            return self.process_output(raw_output)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise ToolExecutionError(self.name, str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error in RequestsTool: {e}")
            raise ToolExecutionError(self.name, str(e)) from e

    def get_schema(self) -> dict[str, Any]:
        """
        Builds a combined JSON Schema describing the tool's route/path, query, header, and body parameters.

        Route parameters are added as string properties and marked required. If a query or header parameter schema includes `"required": True`, that name is added to the top-level required list and the flag is removed from the individual schema. Body schema properties and any body-level required list are merged into the resulting properties and required list.

        Returns:
            dict: JSON Schema object with "type": "object", "properties" mapping parameter names to their schemas, and a "required" list of parameter names (empty list if none).
        """
        properties = {}
        required = []

        # Add route/path parameters
        for param in self.path_params:
            if param in self.path_params_schema:
                # Use schema if available
                properties[param] = self.path_params_schema[param].copy()
            else:
                # Fallback to default
                properties[param] = {
                    "type": "string",
                    "description": f"Route parameter: {param}",
                }
            required.append(param)

        # Add query parameters
        for param_name, param_schema in self.query_params_schema.items():
            properties[param_name] = param_schema.copy()
            if param_schema.get("required", False):
                required.append(param_name)
                # Remove 'required' from individual param schema
                properties[param_name].pop("required", None)

        # Add header parameters
        for param_name, param_schema in self.header_params_schema.items():
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
    headers: dict[str, str] | None = None,
    parameters: dict[str, Any] | None = None,
    body_schema: dict[str, Any] | None = None,
) -> RequestsTool:
    """
    Create a configured RequestsTool for a single API endpoint.

    Parameters:
        name: Tool identifier shown in tooling UIs.
        description: Short human-readable description of the tool's purpose.
        endpoint: URL template for the API endpoint (may include `{param}` placeholders).
        method: HTTP method to use (e.g., "GET", "POST").
        headers: Default request headers to include.
        parameters: JSON Schema describing query parameters.
        body_schema: JSON Schema describing the request body.

    Returns:
        A RequestsTool instance configured with the provided endpoint, method, headers, query parameters schema, and body schema.
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
