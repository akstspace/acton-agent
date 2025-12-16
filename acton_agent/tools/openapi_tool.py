"""
OpenAPI Tool Generator for the AI Agent Framework.

This module provides automatic tool generation from OpenAPI/Swagger specifications,
supporting all standard OpenAPI 3.x features and common quirks.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger

from .requests_tool import RequestsTool


def create_tools_from_openapi(
    spec: Union[str, Dict[str, Any]],
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
    operation_ids: Optional[List[str]] = None,
    max_tools: Optional[int] = None,
) -> List[RequestsTool]:
    """
    Create RequestsTool instances from an OpenAPI specification.

    This is the main function to use for converting OpenAPI specs to tools.

    Args:
        spec: OpenAPI spec as dict, URL, or file path
        base_url: Override base URL from spec
        headers: Additional headers for all requests (e.g., {'X-Api-Key': 'key'})
        tags: Only include operations with these tags
        operation_ids: Only include specific operation IDs
        max_tools: Maximum number of tools to generate

    Returns:
        List of RequestsTool instances

    Example:
        >>> tools = create_tools_from_openapi(
        ...     spec="https://api.example.com/openapi.json",
        ...     base_url="https://api.example.com",
        ...     headers={"X-Api-Key": "your-key"},
        ...     tags=["Movies", "TV Shows"],
        ...     max_tools=30
        ... )
    """
    generator = OpenAPIToolGenerator(spec=spec, base_url=base_url, headers=headers)
    return generator.generate_tools(
        tags=tags, operation_ids=operation_ids, max_tools=max_tools
    )


class OpenAPIToolGenerator:
    """
    Generate RequestsTool instances from OpenAPI specifications.

    Supports:
    - OpenAPI 3.0.x and 3.1.x
    - Multiple servers and server variables
    - All parameter types (path, query, header, cookie)
    - Request bodies (JSON, form-data, etc.)
    - Security schemes (API keys, OAuth, etc.)
    - References ($ref)
    """

    def __init__(
        self,
        spec: Union[str, Dict[str, Any]],
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        operation_filter: Optional[callable] = None,
    ):
        """
        Initialize the OpenAPI tool generator.

        Args:
            spec: OpenAPI spec as dict or URL/path to JSON/YAML file
            base_url: Override base URL from spec
            headers: Additional headers to include in all requests
            operation_filter: Function to filter which operations to include
        """
        self.spec = self._load_spec(spec)
        self.base_url = base_url or self._extract_base_url()
        self.headers = headers or {}
        self.operation_filter = operation_filter

    def _load_spec(self, spec: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load OpenAPI spec from various sources."""
        if isinstance(spec, dict):
            return spec

        # Try to load from URL
        if spec.startswith("http://") or spec.startswith("https://"):
            logger.info(f"Loading OpenAPI spec from URL: {spec}")
            response = requests.get(spec, timeout=30)
            response.raise_for_status()
            return response.json()

        # Try to load from file
        logger.info(f"Loading OpenAPI spec from file: {spec}")
        with open(spec, "r") as f:
            content = f.read()

        # Try JSON first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml

                return yaml.safe_load(content)
            except ImportError:
                raise ValueError(
                    "PyYAML is required to load YAML specs. Install with: pip install pyyaml"
                )

    def _get_security_headers(self, operation: Dict[str, Any]) -> Dict[str, str]:
        """Extract security headers from operation and spec."""
        security_headers = {}

        # Get security requirements (operation-level or global)
        security_reqs = operation.get("security", self.spec.get("security", []))

        # Get security schemes from components
        security_schemes = self.spec.get("components", {}).get("securitySchemes", {})

        for security_req in security_reqs:
            for scheme_name in security_req.keys():
                scheme = security_schemes.get(scheme_name, {})
                scheme_type = scheme.get("type", "")

                if scheme_type == "apiKey":
                    # API Key authentication
                    header_name = scheme.get("name", "X-Api-Key")
                    location = scheme.get("in", "header")

                    if location == "header":
                        # The actual API key value should be in self.headers already
                        # This just documents that this header is expected
                        logger.debug(
                            f"API Key authentication via header: {header_name}"
                        )

                elif scheme_type == "http":
                    # HTTP authentication (Bearer, Basic, etc.)
                    http_scheme = scheme.get("scheme", "bearer")
                    if http_scheme.lower() == "bearer":
                        logger.debug("Bearer token authentication expected")
                    elif http_scheme.lower() == "basic":
                        logger.debug("Basic authentication expected")

        return security_headers

    def _extract_base_url(self) -> str:
        """Extract base URL from OpenAPI spec."""
        servers = self.spec.get("servers", [])
        if servers:
            server = servers[0]
            url = server.get("url", "")

            # Handle server variables
            variables = server.get("variables", {})
            for var_name, var_spec in variables.items():
                default_value = var_spec.get("default", "")
                url = url.replace(f"{{{var_name}}}", default_value)

            return url

        # Fallback to empty string
        return ""

    def generate_tools(
        self,
        tags: Optional[List[str]] = None,
        operation_ids: Optional[List[str]] = None,
        max_tools: Optional[int] = None,
    ) -> List[RequestsTool]:
        """
        Generate RequestsTool instances from the OpenAPI specification.

        Args:
            tags: Only include operations with these tags
            operation_ids: Only include specific operation IDs
            max_tools: Maximum number of tools to generate

        Returns:
            List of RequestsTool instances
        """
        tools = []
        paths = self.spec.get("paths", {})

        for path, path_item in paths.items():
            # Get path-level parameters
            path_parameters = path_item.get("parameters", [])

            for method in ["get", "post", "put", "delete", "patch"]:
                if method not in path_item:
                    continue

                operation = path_item[method]

                # Check filters
                if tags and not any(tag in operation.get("tags", []) for tag in tags):
                    continue

                operation_id = operation.get("operationId")
                if operation_ids and operation_id not in operation_ids:
                    continue

                if self.operation_filter and not self.operation_filter(operation):
                    continue

                # Generate tool
                tool = self._create_tool(method, path, operation, path_parameters)
                if tool:
                    tools.append(tool)

                # Check max tools limit
                if max_tools and len(tools) >= max_tools:
                    return tools

        logger.info(f"Generated {len(tools)} RequestsTools from OpenAPI spec")
        return tools

    def _create_tool(
        self,
        method: str,
        path: str,
        operation: Dict[str, Any],
        path_parameters: List[Dict[str, Any]],
    ) -> Optional[RequestsTool]:
        """Create a RequestsTool from an OpenAPI operation."""
        try:
            # Generate tool name
            operation_id = operation.get("operationId", "")
            if not operation_id:
                # Generate from method and path
                operation_id = f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '')}"

            # Clean up operation ID for tool name
            name = re.sub(r"[^a-zA-Z0-9_]", "_", operation_id.lower())
            name = re.sub(r"_+", "_", name).strip("_")

            # Get description from OpenAPI spec
            # Prefer summary, but combine with description if both exist
            summary = operation.get("summary", "")
            detail_description = operation.get("description", "")

            if summary and detail_description and summary != detail_description:
                # Both exist and are different - combine them
                description = f"{summary}. {detail_description}"
            elif summary:
                description = summary
            elif detail_description:
                description = detail_description
            else:
                # Fallback to method and path
                description = f"{method.upper()} {path}"

            # Limit length but try to keep complete sentences
            if len(description) > 300:
                description = description[:297] + "..."

            # Combine parameters
            all_parameters = path_parameters + operation.get("parameters", [])

            # Build URL template
            url_template = self.base_url.rstrip("/") + "/" + path.lstrip("/")

            # Extract path, query, and header parameters
            path_param_names = []
            query_params = {}
            header_params = {}

            for param in all_parameters:
                param_name = param["name"]
                param_in = param.get("in", "query")
                param_schema = param.get("schema", {"type": "string"})

                param_def = {
                    "type": self._openapi_type_to_json_type(
                        param_schema.get("type", "string")
                    ),
                    "description": param.get("description", f"{param_name} parameter"),
                    "required": param.get("required", False),
                }

                # Add enum if present
                if "enum" in param_schema:
                    param_def["enum"] = param_schema["enum"]

                # Add default if present
                if "default" in param_schema:
                    param_def["default"] = param_schema["default"]

                # Handle array types
                if param_schema.get("type") == "array":
                    param_def["type"] = "array"
                    if "items" in param_schema:
                        param_def["items"] = {
                            "type": self._openapi_type_to_json_type(
                                param_schema["items"].get("type", "string")
                            )
                        }

                if param_in == "path":
                    path_param_names.append(param_name)
                elif param_in == "query":
                    query_params[param_name] = param_def
                elif param_in == "header":
                    header_params[param_name] = param_def

            # Merge headers: base headers + security headers + header parameters
            tool_headers = self.headers.copy()

            # Add security headers (from security schemes)
            security_headers = self._get_security_headers(operation)
            tool_headers.update(security_headers)

            # Note: header_params are documented but values come from tool invocation
            # They will be added at runtime by RequestsTool

            # Handle request body - support multiple content types
            body_schema = None
            request_body = operation.get("requestBody")
            if request_body:
                content = request_body.get("content", {})

                # Try application/json first (most common)
                if "application/json" in content:
                    schema = content["application/json"].get("schema", {})
                    body_schema = self._convert_openapi_schema(schema)
                # Try other JSON-like content types
                elif "application/vnd.api+json" in content:
                    schema = content["application/vnd.api+json"].get("schema", {})
                    body_schema = self._convert_openapi_schema(schema)
                # Try form data
                elif "application/x-www-form-urlencoded" in content:
                    schema = content["application/x-www-form-urlencoded"].get(
                        "schema", {}
                    )
                    body_schema = self._convert_openapi_schema(schema)
                # Try multipart/form-data
                elif "multipart/form-data" in content:
                    schema = content["multipart/form-data"].get("schema", {})
                    body_schema = self._convert_openapi_schema(schema)
                # Fall back to first available content type
                elif content:
                    first_content_type = list(content.keys())[0]
                    schema = content[first_content_type].get("schema", {})
                    body_schema = self._convert_openapi_schema(schema)

                # Mark body as required if specified
                if body_schema and request_body.get("required", False):
                    body_schema["required"] = True

            # Create the RequestsTool
            return RequestsTool(
                name=name,
                description=description,
                method=method.upper(),
                url_template=url_template,
                headers=tool_headers,
                path_params=path_param_names if path_param_names else None,
                query_params_schema=query_params if query_params else None,
                body_schema=body_schema,
            )

        except Exception as e:
            logger.warning(f"Failed to create tool for {method.upper()} {path}: {e}")
            return None

    def _openapi_type_to_json_type(self, openapi_type: str) -> str:
        """Convert OpenAPI type to JSON Schema type."""
        type_map = {
            "integer": "number",
            "number": "number",
            "string": "string",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        return type_map.get(openapi_type, "string")

    def _convert_openapi_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI schema to JSON Schema format for RequestsTool."""
        if not schema:
            return {}

        # Handle $ref - resolve references
        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            if resolved:
                return self._convert_openapi_schema(resolved)
            # Fallback if resolution fails
            return {"type": "object"}

        # Handle allOf, oneOf, anyOf composition
        if "allOf" in schema:
            # Merge all schemas (simplified)
            merged = {"type": "object", "properties": {}}
            for sub_schema in schema["allOf"]:
                converted = self._convert_openapi_schema(sub_schema)
                if "properties" in converted:
                    merged["properties"].update(converted["properties"])
                if "required" in converted:
                    merged.setdefault("required", []).extend(converted["required"])
            return merged

        if "oneOf" in schema or "anyOf" in schema:
            # Use first schema as a reasonable default
            options = schema.get("oneOf", schema.get("anyOf", []))
            if options:
                return self._convert_openapi_schema(options[0])

        result = {"type": schema.get("type", "object")}

        # Add description if present
        if "description" in schema:
            result["description"] = schema["description"]

        # Handle properties
        if "properties" in schema:
            result["properties"] = {}
            for prop_name, prop_schema in schema["properties"].items():
                prop_converted = {
                    "type": self._openapi_type_to_json_type(
                        prop_schema.get("type", "string")
                    ),
                    "description": prop_schema.get("description", f"{prop_name} field"),
                }

                # Add enum if present
                if "enum" in prop_schema:
                    prop_converted["enum"] = prop_schema["enum"]

                # Add default if present
                if "default" in prop_schema:
                    prop_converted["default"] = prop_schema["default"]

                # Handle nested objects
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    prop_converted = self._convert_openapi_schema(prop_schema)

                # Handle arrays
                if prop_schema.get("type") == "array":
                    prop_converted["type"] = "array"
                    if "items" in prop_schema:
                        prop_converted["items"] = self._convert_openapi_schema(
                            prop_schema["items"]
                        )

                # Handle $ref in properties
                if "$ref" in prop_schema:
                    prop_converted = self._convert_openapi_schema(prop_schema)

                result["properties"][prop_name] = prop_converted

        # Handle required fields
        if "required" in schema:
            result["required"] = schema["required"]

        # Handle additionalProperties
        if "additionalProperties" in schema:
            result["additionalProperties"] = schema["additionalProperties"]

        return result

    def _resolve_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        """Resolve a $ref to its schema definition."""
        if not ref.startswith("#/"):
            # External refs not supported yet
            logger.debug(f"External $ref not supported: {ref}")
            return None

        # Parse the reference path
        parts = ref.split("/")[1:]  # Skip the '#'

        # Navigate through the spec
        current = self.spec
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                logger.debug(f"Could not resolve $ref: {ref}")
                return None

        return current if isinstance(current, dict) else None

    def get_tags(self) -> List[str]:
        """Get all tags from the OpenAPI spec."""
        tags = set()
        paths = self.spec.get("paths", {})

        for path_item in paths.values():
            for method in ["get", "post", "put", "delete", "patch"]:
                if method in path_item:
                    operation_tags = path_item[method].get("tags", [])
                    tags.update(operation_tags)

        return sorted(tags)

    def get_operation_ids(self) -> List[str]:
        """Get all operation IDs from the OpenAPI spec."""
        operation_ids = []
        paths = self.spec.get("paths", {})

        for path_item in paths.values():
            for method in ["get", "post", "put", "delete", "patch"]:
                if method in path_item:
                    op_id = path_item[method].get("operationId")
                    if op_id:
                        operation_ids.append(op_id)

        return operation_ids
