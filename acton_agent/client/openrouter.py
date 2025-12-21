"""
OpenRouter LLM Client implementation.
"""

import os

from .openai_client import OpenAIClient


class OpenRouterClient(OpenAIClient):
    """
    LLM client implementation for OpenRouter.

    OpenRouter provides a unified API to access multiple LLM providers
    using the OpenAI-compatible API format. This client extends OpenAIClient
    to add OpenRouter-specific headers and configuration.

    Example:
        ```python
        client = OpenRouterClient(
            api_key="your-api-key",
            model="openai/gpt-4o",
            site_url="https://yoursite.com",
            site_name="Your App Name"
        )

        messages = [
            Message(role="user", content="What is the meaning of life?")
        ]

        response = client.call(messages)
        print(response)

        # Streaming
        for chunk in client.call_stream(messages):
            print(chunk, end="", flush=True)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o",
        site_url: str | None = None,
        site_name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Create an OpenRouter client configured with an API key, model, and optional site headers used for OpenRouter ranking.
        
        Parameters:
            api_key: OpenRouter API key; if omitted the OPENROUTER_API_KEY environment variable will be used.
            model: Model identifier to use (e.g., "openai/gpt-4o" or "anthropic/claude-3-opus").
            site_url: Optional URL sent as the "HTTP-Referer" header to influence OpenRouter ranking.
            site_name: Optional site name sent as the "X-Title" header to influence OpenRouter ranking.
            base_url: OpenRouter API base URL.
        
        Raises:
            ValueError: If no API key is provided via the api_key parameter or the OPENROUTER_API_KEY environment variable.
        """
        final_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not final_api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as 'api_key' parameter "
                "or via OPENROUTER_API_KEY environment variable"
            )

        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        self.site_url = site_url
        self.site_name = site_name

        super().__init__(
            api_key=final_api_key,
            model=model,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
        )