"""
Tests for the requests_tool module.
"""

from unittest.mock import Mock, patch

import pytest

from acton_agent.agent.exceptions import ToolExecutionError
from acton_agent.tools.requests_tool import RequestsTool, create_api_tool


class TestRequestsTool:
    """Tests for RequestsTool."""

    def test_create_tool(self):
        """Test creating a RequestsTool."""
        tool = RequestsTool(
            name="test_api",
            description="Test API endpoint",
            method="GET",
            url_template="https://api.example.com/data",
        )

        assert tool.name == "test_api"
        assert tool.method == "GET"
        assert tool.url_template == "https://api.example.com/data"

    def test_get_schema_with_query_params(self):
        """Test getting schema with query parameters."""
        tool = RequestsTool(
            name="search",
            description="Search API",
            method="GET",
            url_template="https://api.example.com/search",
            query_params_schema={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True,
                },
                "limit": {"type": "number", "description": "Result limit"},
            },
        )

        schema = tool.get_schema()
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema["required"]

    def test_get_schema_with_path_params(self):
        """Test getting schema with path parameters."""
        tool = RequestsTool(
            name="get_user",
            description="Get user by ID",
            method="GET",
            url_template="https://api.example.com/users/{user_id}",
            path_params=["user_id"],
        )

        schema = tool.get_schema()
        assert "user_id" in schema["properties"]
        assert "user_id" in schema["required"]

    def test_get_schema_with_body(self):
        """Test getting schema with body parameters."""
        tool = RequestsTool(
            name="create_user",
            description="Create a new user",
            method="POST",
            url_template="https://api.example.com/users",
            body_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name", "email"],
            },
        )

        schema = tool.get_schema()
        assert "name" in schema["properties"]
        assert "email" in schema["properties"]
        assert "name" in schema["required"]
        assert "email" in schema["required"]

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_get_request(self, mock_request):
        """Test executing a GET request."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="test_get",
            description="Test GET",
            method="GET",
            url_template="https://api.example.com/data",
        )

        result = tool.execute({})

        assert "success" in result
        mock_request.assert_called_once()

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_post_request(self, mock_request):
        """Test executing a POST request."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123, "status": "created"}
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="create",
            description="Create resource",
            method="POST",
            url_template="https://api.example.com/resources",
            body_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )

        result = tool.execute({"name": "Test Resource"})

        assert "123" in result
        mock_request.assert_called_once()

        # Check that json parameter was passed
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"] == {"name": "Test Resource"}

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_with_query_params(self, mock_request):
        """Test executing request with query parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="search",
            description="Search",
            method="GET",
            url_template="https://api.example.com/search",
            query_params_schema={"q": {"type": "string"}},
        )

        tool.execute({"q": "test query"})

        # Check that params were passed
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"] == {"q": "test query"}

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_with_path_params(self, mock_request):
        """Test executing request with path parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 123, "name": "Test"}
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="get_user",
            description="Get user",
            method="GET",
            url_template="https://api.example.com/users/{user_id}",
            path_params=["user_id"],
        )

        tool.execute({"user_id": "123"})

        # Check that URL was formatted correctly
        call_args = mock_request.call_args
        assert "users/123" in call_args[1]["url"]

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_request_failure(self, mock_request):
        """
        Verify that RequestsTool raises ToolExecutionError when the HTTP request fails.

        Asserts that calling execute results in a ToolExecutionError if the underlying HTTP response's raise_for_status raises an exception (simulating an error status like 404).
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="test",
            description="Test",
            method="GET",
            url_template="https://api.example.com/notfound",
        )

        with pytest.raises(ToolExecutionError):
            tool.execute({})

    @patch("acton_agent.tools.requests_tool.requests.request")
    def test_execute_with_headers(self, mock_request):
        """Test executing request with custom headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_request.return_value = mock_response

        tool = RequestsTool(
            name="auth_api",
            description="API with auth",
            method="GET",
            url_template="https://api.example.com/secure",
            headers={"Authorization": "Bearer token123"},
        )

        tool.execute({})

        # Check that headers were passed
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"] == {"Authorization": "Bearer token123"}


class TestCreateApiTool:
    """Tests for create_api_tool factory function."""

    def test_create_simple_tool(self):
        """Test creating a simple API tool."""
        tool = create_api_tool(
            name="weather",
            description="Get weather",
            endpoint="https://api.weather.com/current",
            method="GET",
        )

        assert isinstance(tool, RequestsTool)
        assert tool.name == "weather"
        assert tool.method == "GET"

    def test_create_tool_with_parameters(self):
        """Test creating tool with parameters."""
        tool = create_api_tool(
            name="search",
            description="Search",
            endpoint="https://api.example.com/search",
            method="GET",
            parameters={"q": {"type": "string", "description": "Query", "required": True}},
        )

        schema = tool.get_schema()
        assert "q" in schema["properties"]

    def test_create_post_tool_with_body(self):
        """Test creating POST tool with body schema."""
        tool = create_api_tool(
            name="create",
            description="Create item",
            endpoint="https://api.example.com/items",
            method="POST",
            body_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert tool.method == "POST"
        schema = tool.get_schema()
        assert "name" in schema["properties"]
