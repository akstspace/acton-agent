"""
Tests for OpenAI and OpenRouter clients.
"""

import os
from unittest.mock import Mock, patch

import pytest

from acton_agent.agent.models import Message
from acton_agent.client.openai_client import OpenAIClient
from acton_agent.client.openrouter import OpenRouterClient


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    def test_client_creation_with_api_key(self):
        """Test creating client with explicit API key."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o")
        assert client.model == "gpt-4o"

    def test_client_creation_without_api_key(self):
        """Test that client raises error without API key."""
        # Clear environment variable if exists
        old_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                OpenAIClient()
        finally:
            # Restore old key
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_client_uses_env_var(self):
        """Test that client uses OPENAI_API_KEY env var."""
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "env-key"

        try:
            client = OpenAIClient()
            # Just check it doesn't raise an error
            assert client is not None
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                del os.environ["OPENAI_API_KEY"]

    @patch("acton_agent.client.openai_client.OpenAI")
    def test_call_method(self, mock_openai_class):
        """Test call method."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_completion

        # Create client and call
        client = OpenAIClient(api_key="test-key")
        messages = [Message(role="user", content="Hello")]
        result = client.call(messages)

        assert result == "Test response"
        assert mock_client.chat.completions.create.called

    @patch("acton_agent.client.openai_client.OpenAI")
    def test_call_stream_method(self, mock_openai_class):
        """Test call_stream method."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" "))]),
            Mock(choices=[Mock(delta=Mock(content="world"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        # Create client and stream
        client = OpenAIClient(api_key="test-key")
        messages = [Message(role="user", content="Hi")]

        chunks = list(client.call_stream(messages))

        assert chunks == ["Hello", " ", "world"]
        assert mock_client.chat.completions.create.called


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    def test_client_creation_with_api_key(self):
        """Test creating OpenRouter client with explicit API key."""
        client = OpenRouterClient(
            api_key="test-key",
            model="openai/gpt-4o",
            site_url="https://example.com",
            site_name="Test App",
        )
        assert client.model == "openai/gpt-4o"
        assert client.site_url == "https://example.com"
        assert client.site_name == "Test App"

    def test_client_creation_without_api_key(self):
        """Test that client raises error without API key."""
        old_key = os.environ.get("OPENROUTER_API_KEY")
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                OpenRouterClient()
        finally:
            if old_key:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_client_uses_env_var(self):
        """Test that client uses OPENROUTER_API_KEY env var."""
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "env-key"

        try:
            client = OpenRouterClient()
            assert client is not None
        finally:
            if old_key:
                os.environ["OPENROUTER_API_KEY"] = old_key
            else:
                del os.environ["OPENROUTER_API_KEY"]

    @patch("acton_agent.client.openai_client.OpenAI")
    def test_openrouter_inherits_from_openai(self, mock_openai_class):
        """Test that OpenRouter inherits from OpenAI client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = OpenRouterClient(api_key="test-key")

        # Should have call and call_stream methods
        assert hasattr(client, "call")
        assert hasattr(client, "call_stream")

    @patch("acton_agent.client.openai_client.OpenAI")
    def test_openrouter_custom_headers(self, mock_openai_class):
        """Test that OpenRouter sets custom headers."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        OpenRouterClient(api_key="test-key", site_url="https://myapp.com", site_name="My App")

        # Check that OpenAI was initialized with custom headers
        call_kwargs = mock_openai_class.call_args[1]
        assert "default_headers" in call_kwargs
        headers = call_kwargs["default_headers"]
        assert headers["HTTP-Referer"] == "https://myapp.com"
        assert headers["X-Title"] == "My App"
