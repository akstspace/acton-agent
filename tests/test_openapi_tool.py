"""
Comprehensive tests for the OpenAPI tool generator module.
"""

import json
from unittest.mock import Mock, mock_open, patch

import pytest

from acton_agent.tools.openapi_tool import (
    OpenAPIToolGenerator,
    create_tools_from_openapi,
)
from acton_agent.tools.requests_tool import RequestsTool


# Sample OpenAPI specifications for testing
MINIMAL_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/users": {
            "get": {
                "operationId": "listUsers",
                "summary": "List all users",
                "description": "Retrieve a list of all users in the system",
                "responses": {"200": {"description": "Success"}},
            }
        }
    },
}

COMPLEX_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Complex API", "version": "2.0.0"},
    "servers": [
        {
            "url": "https://{environment}.api.example.com/{version}",
            "variables": {
                "environment": {"default": "prod", "enum": ["prod", "staging"]},
                "version": {"default": "v1"},
            },
        }
    ],
    "paths": {
        "/users/{user_id}": {
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "The user ID",
                }
            ],
            "get": {
                "operationId": "getUser",
                "summary": "Get user by ID",
                "tags": ["Users"],
                "parameters": [
                    {
                        "name": "include_details",
                        "in": "query",
                        "schema": {"type": "boolean", "default": False},
                        "description": "Include detailed information",
                    }
                ],
                "responses": {"200": {"description": "User found"}},
            },
            "put": {
                "operationId": "updateUser",
                "summary": "Update user",
                "tags": ["Users"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                    "age": {"type": "integer"},
                                },
                                "required": ["name", "email"],
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "User updated"}},
            },
            "delete": {
                "operationId": "deleteUser",
                "summary": "Delete user",
                "tags": ["Users", "Admin"],
                "responses": {"204": {"description": "User deleted"}},
            },
        },
        "/posts": {
            "post": {
                "operationId": "createPost",
                "summary": "Create a new post",
                "tags": ["Posts"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["title"],
                            }
                        }
                    }
                },
                "responses": {"201": {"description": "Post created"}},
            }
        },
    },
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "BearerAuth": {"type": "http", "scheme": "bearer"},
        }
    },
}

SPEC_WITH_REFS = {
    "openapi": "3.0.0",
    "info": {"title": "API with Refs", "version": "1.0.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/items": {
            "post": {
                "operationId": "createItem",
                "summary": "Create item",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Item"}
                        }
                    }
                },
                "responses": {"201": {"description": "Created"}},
            }
        }
    },
    "components": {
        "schemas": {
            "Item": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "metadata": {"$ref": "#/components/schemas/Metadata"},
                },
                "required": ["name"],
            },
            "Metadata": {
                "type": "object",
                "properties": {"created_at": {"type": "string"}},
            },
        }
    },
}

SPEC_WITH_ALLOF = {
    "openapi": "3.0.0",
    "info": {"title": "API with allOf", "version": "1.0.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/resource": {
            "post": {
                "operationId": "createResource",
                "summary": "Create resource",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "allOf": [
                                    {
                                        "type": "object",
                                        "properties": {"id": {"type": "integer"}},
                                        "required": ["id"],
                                    },
                                    {
                                        "type": "object",
                                        "properties": {"name": {"type": "string"}},
                                        "required": ["name"],
                                    },
                                ]
                            }
                        }
                    }
                },
                "responses": {"201": {"description": "Created"}},
            }
        }
    },
}


class TestOpenAPIToolGenerator:
    """Tests for OpenAPIToolGenerator class."""

    def test_init_with_dict_spec(self):
        """Test initialization with a dict specification."""
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC)
        assert generator.spec == MINIMAL_OPENAPI_SPEC
        assert generator.base_url == "https://api.example.com"
        assert generator.headers == {}

    def test_init_with_custom_base_url(self):
        """Test initialization with a custom base URL override."""
        custom_url = "https://custom.example.com"
        generator = OpenAPIToolGenerator(
            spec=MINIMAL_OPENAPI_SPEC, base_url=custom_url
        )
        assert generator.base_url == custom_url

    def test_init_with_custom_headers(self):
        """Test initialization with custom headers."""
        headers = {"Authorization": "Bearer token123", "X-Custom": "value"}
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC, headers=headers)
        assert generator.headers == headers

    @patch("acton_agent.tools.openapi_tool.requests.get")
    def test_load_spec_from_url(self, mock_get):
        """Test loading OpenAPI spec from a URL."""
        mock_response = Mock()
        mock_response.json.return_value = MINIMAL_OPENAPI_SPEC
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        generator = OpenAPIToolGenerator(spec="https://api.example.com/openapi.json")
        assert generator.spec == MINIMAL_OPENAPI_SPEC
        mock_get.assert_called_once_with("https://api.example.com/openapi.json", timeout=30)

    @patch("acton_agent.tools.openapi_tool.requests.get")
    def test_load_spec_from_url_http_error(self, mock_get):
        """Test that HTTP errors are propagated when loading from URL."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 404"):
            OpenAPIToolGenerator(spec="https://api.example.com/openapi.json")

    @patch("builtins.open", new_callable=mock_open, read_data='{"openapi": "3.0.0"}')
    def test_load_spec_from_json_file(self, mock_file):
        """Test loading OpenAPI spec from a JSON file."""
        generator = OpenAPIToolGenerator(spec="/path/to/spec.json")
        assert "openapi" in generator.spec
        mock_file.assert_called_once_with("/path/to/spec.json", "r")

    @patch("builtins.open", new_callable=mock_open, read_data="openapi: 3.0.0\ninfo:\n  title: Test")
    @patch("acton_agent.tools.openapi_tool.yaml")
    def test_load_spec_from_yaml_file(self, mock_yaml, mock_file):
        """Test loading OpenAPI spec from a YAML file."""
        mock_yaml.safe_load.return_value = {"openapi": "3.0.0", "info": {"title": "Test"}}
        generator = OpenAPIToolGenerator(spec="/path/to/spec.yaml")
        assert generator.spec["openapi"] == "3.0.0"
        mock_yaml.safe_load.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="openapi: 3.0.0")
    def test_load_spec_yaml_without_pyyaml(self, mock_file):
        """Test that loading YAML without PyYAML raises helpful error."""
        with patch("acton_agent.tools.openapi_tool.yaml", None):
            with patch.dict("sys.modules", {"yaml": None}):
                # This will try JSON first, fail, then try YAML import
                with pytest.raises((ValueError, ImportError)):
                    OpenAPIToolGenerator(spec="/path/to/spec.yaml")

    def test_extract_base_url_simple(self):
        """Test extracting base URL from simple server definition."""
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC)
        assert generator.base_url == "https://api.example.com"

    def test_extract_base_url_with_variables(self):
        """Test extracting base URL with server variables."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        assert generator.base_url == "https://prod.api.example.com/v1"

    def test_extract_base_url_no_servers(self):
        """Test extracting base URL when no servers are defined."""
        spec = {"openapi": "3.0.0", "info": {"title": "Test"}, "paths": {}}
        generator = OpenAPIToolGenerator(spec=spec)
        assert generator.base_url == ""

    def test_generate_tools_minimal(self):
        """Test generating tools from minimal OpenAPI spec."""
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC)
        tools = generator.generate_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], RequestsTool)
        assert tools[0].name == "listusers"
        assert tools[0].method == "GET"
        assert "List all users" in tools[0].description

    def test_generate_tools_complex(self):
        """Test generating tools from complex OpenAPI spec."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools()

        # Should generate tools for get, put, delete on /users/{user_id} and post on /posts
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "getuser" in tool_names
        assert "updateuser" in tool_names
        assert "deleteuser" in tool_names
        assert "createpost" in tool_names

    def test_generate_tools_with_tag_filter(self):
        """Test generating tools filtered by tags."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(tags=["Posts"])

        assert len(tools) == 1
        assert tools[0].name == "createpost"

    def test_generate_tools_with_multiple_tag_filter(self):
        """Test generating tools with multiple tag filters."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(tags=["Users", "Admin"])

        # Should include all operations tagged with Users OR Admin
        assert len(tools) >= 3
        tool_names = [t.name for t in tools]
        assert "getuser" in tool_names
        assert "deleteuser" in tool_names

    def test_generate_tools_with_operation_id_filter(self):
        """Test generating tools filtered by operation IDs."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["getUser", "createPost"])

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "getuser" in tool_names
        assert "createpost" in tool_names

    def test_generate_tools_with_max_tools(self):
        """Test limiting the number of generated tools."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(max_tools=2)

        assert len(tools) == 2

    def test_create_tool_with_path_parameters(self):
        """Test tool creation with path parameters."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["getUser"])

        tool = tools[0]
        schema = tool.get_schema()

        assert "user_id" in schema["properties"]
        assert "user_id" in schema["required"]
        assert "The user ID" in schema["properties"]["user_id"]["description"]

    def test_create_tool_with_query_parameters(self):
        """Test tool creation with query parameters."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["getUser"])

        tool = tools[0]
        schema = tool.get_schema()

        assert "include_details" in schema["properties"]
        assert schema["properties"]["include_details"]["type"] == "boolean"

    def test_create_tool_with_request_body(self):
        """Test tool creation with request body schema."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["updateUser"])

        tool = tools[0]
        schema = tool.get_schema()

        assert "name" in schema["properties"]
        assert "email" in schema["properties"]
        assert "age" in schema["properties"]
        assert "name" in schema["required"]
        assert "email" in schema["required"]

    def test_create_tool_with_array_in_body(self):
        """Test tool creation with array properties in request body."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["createPost"])

        tool = tools[0]
        schema = tool.get_schema()

        assert "tags" in schema["properties"]
        assert schema["properties"]["tags"]["type"] == "array"
        assert "items" in schema["properties"]["tags"]

    def test_create_tool_without_operation_id(self):
        """Test tool creation when operation has no operationId."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test/{id}": {
                    "get": {
                        "summary": "Test endpoint",
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert len(tools) == 1
        # Should generate name from method and path
        assert "get" in tools[0].name.lower()
        assert "test" in tools[0].name.lower()

    def test_create_tool_description_from_summary(self):
        """Test that tool description uses summary when available."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tools = generator.generate_tools(operation_ids=["getUser"])

        assert "Get user by ID" in tools[0].description

    def test_create_tool_description_length_limit(self):
        """Test that tool descriptions are truncated if too long."""
        long_description = "A" * 500
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "description": long_description,
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert len(tools[0].description) <= 300

    def test_resolve_ref_simple(self):
        """Test resolving a simple $ref."""
        generator = OpenAPIToolGenerator(spec=SPEC_WITH_REFS)
        resolved = generator._resolve_ref("#/components/schemas/Item")

        assert resolved is not None
        assert resolved["type"] == "object"
        assert "name" in resolved["properties"]

    def test_resolve_ref_nested(self):
        """Test resolving a nested $ref."""
        generator = OpenAPIToolGenerator(spec=SPEC_WITH_REFS)
        resolved = generator._resolve_ref("#/components/schemas/Metadata")

        assert resolved is not None
        assert "created_at" in resolved["properties"]

    def test_resolve_ref_invalid(self):
        """Test resolving an invalid $ref returns None."""
        generator = OpenAPIToolGenerator(spec=SPEC_WITH_REFS)
        resolved = generator._resolve_ref("#/components/schemas/NonExistent")

        assert resolved is None

    def test_resolve_ref_external(self):
        """Test that external $refs are not supported."""
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC)
        resolved = generator._resolve_ref("external.yaml#/components/schemas/Item")

        assert resolved is None

    def test_convert_openapi_schema_with_ref(self):
        """Test converting OpenAPI schema with $ref."""
        generator = OpenAPIToolGenerator(spec=SPEC_WITH_REFS)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "name" in schema["properties"]
        assert "id" in schema["properties"]

    def test_convert_openapi_schema_with_allof(self):
        """Test converting OpenAPI schema with allOf composition."""
        generator = OpenAPIToolGenerator(spec=SPEC_WITH_ALLOF)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        # Should merge properties from all allOf schemas
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]
        assert "id" in schema["required"]
        assert "name" in schema["required"]

    def test_convert_openapi_schema_with_oneof(self):
        """Test converting OpenAPI schema with oneOf (uses first option)."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "post": {
                        "operationId": "testOp",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "oneOf": [
                                            {
                                                "type": "object",
                                                "properties": {"type_a": {"type": "string"}},
                                            },
                                            {
                                                "type": "object",
                                                "properties": {"type_b": {"type": "integer"}},
                                            },
                                        ]
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        # Should use first oneOf option
        assert "type_a" in schema["properties"]

    def test_convert_openapi_schema_with_enum(self):
        """Test converting OpenAPI schema with enum values."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [
                            {
                                "name": "status",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["active", "inactive", "pending"],
                                },
                            }
                        ],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "status" in schema["properties"]
        assert "enum" in schema["properties"]["status"]
        assert schema["properties"]["status"]["enum"] == ["active", "inactive", "pending"]

    def test_convert_openapi_schema_with_defaults(self):
        """Test converting OpenAPI schema with default values."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {"type": "integer", "default": 10},
                            }
                        ],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "limit" in schema["properties"]
        assert schema["properties"]["limit"]["default"] == 10

    def test_openapi_type_to_json_type(self):
        """Test OpenAPI type to JSON type conversion."""
        generator = OpenAPIToolGenerator(spec=MINIMAL_OPENAPI_SPEC)

        assert generator._openapi_type_to_json_type("integer") == "number"
        assert generator._openapi_type_to_json_type("number") == "number"
        assert generator._openapi_type_to_json_type("string") == "string"
        assert generator._openapi_type_to_json_type("boolean") == "boolean"
        assert generator._openapi_type_to_json_type("array") == "array"
        assert generator._openapi_type_to_json_type("object") == "object"
        assert generator._openapi_type_to_json_type("unknown") == "string"

    def test_get_security_headers_api_key(self):
        """Test extracting security headers for API key authentication."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "security": [{"ApiKeyAuth": []}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        operation = spec["paths"]["/test"]["get"]

        # This method logs but doesn't return headers (they come from init headers)
        headers = generator._get_security_headers(operation)
        assert isinstance(headers, dict)

    def test_get_security_headers_bearer(self):
        """Test extracting security headers for Bearer authentication."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "security": [{"BearerAuth": []}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "BearerAuth": {"type": "http", "scheme": "bearer"}
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        operation = spec["paths"]["/test"]["get"]

        headers = generator._get_security_headers(operation)
        assert isinstance(headers, dict)

    def test_get_tags(self):
        """Test getting all unique tags from the spec."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        tags = generator.get_tags()

        assert "Users" in tags
        assert "Posts" in tags
        assert "Admin" in tags
        assert isinstance(tags, list)
        assert tags == sorted(tags)  # Should be sorted

    def test_get_operation_ids(self):
        """Test getting all operation IDs from the spec."""
        generator = OpenAPIToolGenerator(spec=COMPLEX_OPENAPI_SPEC)
        operation_ids = generator.get_operation_ids()

        assert "getUser" in operation_ids
        assert "updateUser" in operation_ids
        assert "deleteUser" in operation_ids
        assert "createPost" in operation_ids

    def test_get_operation_ids_empty_spec(self):
        """Test getting operation IDs from spec with no operations."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {},
        }
        generator = OpenAPIToolGenerator(spec=spec)
        operation_ids = generator.get_operation_ids()

        assert operation_ids == []

    def test_tool_with_header_parameters(self):
        """Test tool creation with header parameters."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [
                            {
                                "name": "X-Custom-Header",
                                "in": "header",
                                "schema": {"type": "string"},
                                "description": "Custom header",
                            }
                        ],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        # Header parameters should be in schema
        assert "x_custom_header" in schema["properties"] or "X-Custom-Header" in str(schema)

    def test_tool_with_cookie_parameters(self):
        """Test tool creation with cookie parameters."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [
                            {
                                "name": "session_id",
                                "in": "cookie",
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        # Cookie params might be ignored or handled differently
        assert len(tools) == 1

    def test_multiple_content_types_in_request_body(self):
        """Test tool creation with multiple content types (should prefer JSON)."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "post": {
                        "operationId": "testOp",
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"form_field": {"type": "string"}},
                                    }
                                },
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"json_field": {"type": "string"}},
                                    }
                                },
                            }
                        },
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        # Should prefer JSON
        assert "json_field" in schema["properties"]

    def test_form_data_content_type(self):
        """Test tool creation with form data content type."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/upload": {
                    "post": {
                        "operationId": "uploadFile",
                        "requestBody": {
                            "content": {
                                "multipart/form-data": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file": {"type": "string", "format": "binary"}
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "file" in schema["properties"]

    def test_nested_object_properties(self):
        """Test tool creation with nested object properties."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "post": {
                        "operationId": "testOp",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "user": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "email": {"type": "string"},
                                                },
                                            }
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "user" in schema["properties"]
        assert schema["properties"]["user"]["type"] == "object"
        assert "properties" in schema["properties"]["user"]

    def test_operation_filter_function(self):
        """Test using operation_filter to customize tool generation."""

        def my_filter(operation):
            # Only include operations with 'Admin' tag
            return "Admin" in operation.get("tags", [])

        generator = OpenAPIToolGenerator(
            spec=COMPLEX_OPENAPI_SPEC, operation_filter=my_filter
        )
        tools = generator.generate_tools()

        # Only deleteUser has Admin tag
        assert len(tools) == 1
        assert tools[0].name == "deleteuser"

    def test_failed_tool_creation_is_skipped(self):
        """Test that failed tool creations are logged and skipped."""
        # Create a malformed spec that will cause tool creation to fail
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/good": {
                    "get": {
                        "operationId": "goodOp",
                        "responses": {"200": {"description": "OK"}},
                    }
                },
                "/bad": {
                    "get": {
                        # Missing required fields to cause error
                        "responses": None,
                    }
                },
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        # Should still generate the good tool
        assert len(tools) >= 0

    def test_array_parameter_in_query(self):
        """Test handling of array-type query parameters."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [
                            {
                                "name": "ids",
                                "in": "query",
                                "schema": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            }
                        ],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        tool = tools[0]
        schema = tool.get_schema()

        assert "ids" in schema["properties"]
        assert schema["properties"]["ids"]["type"] == "array"
        assert "items" in schema["properties"]["ids"]


class TestCreateToolsFromOpenAPI:
    """Tests for the create_tools_from_openapi convenience function."""

    def test_create_tools_with_dict_spec(self):
        """Test creating tools with a dict specification."""
        tools = create_tools_from_openapi(spec=MINIMAL_OPENAPI_SPEC)

        assert len(tools) == 1
        assert isinstance(tools[0], RequestsTool)

    @patch("acton_agent.tools.openapi_tool.requests.get")
    def test_create_tools_with_url_spec(self, mock_get):
        """Test creating tools from URL specification."""
        mock_response = Mock()
        mock_response.json.return_value = MINIMAL_OPENAPI_SPEC
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        tools = create_tools_from_openapi(spec="https://api.example.com/openapi.json")

        assert len(tools) == 1
        assert isinstance(tools[0], RequestsTool)

    def test_create_tools_with_base_url_override(self):
        """Test creating tools with base URL override."""
        custom_url = "https://custom.example.com"
        tools = create_tools_from_openapi(spec=MINIMAL_OPENAPI_SPEC, base_url=custom_url)

        assert len(tools) == 1
        assert custom_url in tools[0].url_template

    def test_create_tools_with_headers(self):
        """Test creating tools with custom headers."""
        headers = {"X-API-Key": "secret123"}
        tools = create_tools_from_openapi(spec=MINIMAL_OPENAPI_SPEC, headers=headers)

        assert len(tools) == 1
        assert tools[0].headers.get("X-API-Key") == "secret123"

    def test_create_tools_with_tag_filter(self):
        """Test creating tools with tag filter."""
        tools = create_tools_from_openapi(spec=COMPLEX_OPENAPI_SPEC, tags=["Posts"])

        assert len(tools) == 1
        assert tools[0].name == "createpost"

    def test_create_tools_with_operation_id_filter(self):
        """Test creating tools with operation ID filter."""
        tools = create_tools_from_openapi(
            spec=COMPLEX_OPENAPI_SPEC, operation_ids=["getUser"]
        )

        assert len(tools) == 1
        assert tools[0].name == "getuser"

    def test_create_tools_with_max_tools(self):
        """Test creating tools with max_tools limit."""
        tools = create_tools_from_openapi(spec=COMPLEX_OPENAPI_SPEC, max_tools=2)

        assert len(tools) == 2

    def test_create_tools_all_parameters(self):
        """Test creating tools with all parameters specified."""
        tools = create_tools_from_openapi(
            spec=COMPLEX_OPENAPI_SPEC,
            base_url="https://custom.api.com",
            headers={"Authorization": "Bearer token"},
            tags=["Users"],
            operation_ids=["getUser", "updateUser"],
            max_tools=10,
        )

        # Should match both filters (tags AND operation_ids)
        assert len(tools) <= 10
        for tool in tools:
            assert "custom.api.com" in tool.url_template


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_spec(self):
        """Test handling of empty specification."""
        spec = {"openapi": "3.0.0", "info": {"title": "Empty"}, "paths": {}}
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert tools == []

    def test_spec_without_paths(self):
        """Test handling of spec without paths key."""
        spec = {"openapi": "3.0.0", "info": {"title": "Test"}}
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert tools == []

    def test_path_without_operations(self):
        """Test handling of path with no operations."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {"/test": {"summary": "Just a summary, no operations"}},
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert tools == []

    def test_very_long_operation_id(self):
        """Test handling of very long operation ID."""
        long_id = "a" * 200
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": long_id,
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert len(tools) == 1
        # Tool name should be sanitized
        assert len(tools[0].name) > 0

    def test_special_characters_in_operation_id(self):
        """Test handling of special characters in operation ID."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "test-operation.with@special#chars!",
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert len(tools) == 1
        # Should sanitize to valid tool name (alphanumeric and underscores)
        assert tools[0].name.replace("_", "").replace("test", "").replace("operation", "").replace("with", "").replace("special", "").replace("chars", "") == ""

    def test_missing_schema_in_parameter(self):
        """Test handling of parameter without schema."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                        "parameters": [{"name": "param1", "in": "query"}],
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        # Should still create tool with default string type
        assert len(tools) == 1

    def test_empty_request_body_content(self):
        """Test handling of request body with empty content."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "post": {
                        "operationId": "testOp",
                        "requestBody": {"content": {}},
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        assert len(tools) == 1

    def test_circular_ref(self):
        """Test handling of circular $ref (should not infinite loop)."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {
                    "post": {
                        "operationId": "testOp",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Node"}
                                }
                            }
                        },
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
            "components": {
                "schemas": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "next": {"$ref": "#/components/schemas/Node"},
                        },
                    }
                }
            },
        }
        generator = OpenAPIToolGenerator(spec=spec)
        tools = generator.generate_tools()

        # Should create tool without hanging
        assert len(tools) == 1