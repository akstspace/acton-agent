"""
Tests for dependency imports and requirements.
"""

import sys
from unittest.mock import patch

import pytest


class TestRequiredDependencies:
    """Test that required dependencies are properly imported."""

    def test_openai_is_available(self):
        """Test that openai module is available (now a required dependency)."""
        import openai

        assert openai is not None

    def test_acton_agent_imports_successfully(self):
        """Test that acton_agent can be imported with openai installed."""
        import acton_agent

        assert acton_agent is not None

    def test_openai_client_imports_successfully(self):
        """Test that OpenAIClient can be imported."""
        from acton_agent.client import OpenAIClient

        assert OpenAIClient is not None

    def test_openrouter_client_imports_successfully(self):
        """Test that OpenRouterClient can be imported."""
        from acton_agent.client import OpenRouterClient

        assert OpenRouterClient is not None


class TestOptionalDependencies:
    """Test handling of optional dependencies."""

    def test_missing_openai_raises_import_error(self):
        """Test that missing openai raises ImportError when importing client module."""
        # This test simulates what would happen if openai was not installed
        # Since openai is now required, this test verifies the expected behavior
        # if someone manually removes it or imports fail

        with patch.dict(sys.modules, {"openai": None}):
            # Clear the already imported modules to force reimport
            modules_to_clear = [
                k for k in sys.modules.keys() if k.startswith("acton_agent.client")
            ]
            for module in modules_to_clear:
                del sys.modules[module]

            # Now try to import - should fail with ImportError
            with pytest.raises((ImportError, AttributeError)):
                from acton_agent.client import OpenAIClient  # noqa: F401


class TestClientInstantiation:
    """Test that clients can be instantiated properly."""

    def test_openai_client_requires_api_key(self):
        """Test that OpenAIClient requires an API key."""
        from acton_agent.client import OpenAIClient

        with pytest.raises(ValueError, match="API key must be provided"):
            OpenAIClient()

    def test_openrouter_client_requires_api_key(self):
        """Test that OpenRouterClient requires an API key."""
        from acton_agent.client import OpenRouterClient

        with pytest.raises(ValueError, match="API key must be provided"):
            OpenRouterClient()
