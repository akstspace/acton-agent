"""
OpenRouter LLM Client implementation.
"""

import os
from typing import List, Optional, Generator

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
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: Your OpenRouter API key (reads from OPENROUTER_API_KEY env var if not provided)
            model: The model to use (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
            site_url: Optional site URL for rankings on openrouter.ai
            site_name: Optional site name for rankings on openrouter.ai
            base_url: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
            
        Raises:
            ValueError: If no API key is provided and OPENROUTER_API_KEY env var is not set
        """
        # Get API key from parameter or environment variable
        final_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        
        if not final_api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as 'api_key' parameter "
                "or via OPENROUTER_API_KEY environment variable"
            )
        
        # Prepare OpenRouter-specific headers
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name
        
        # Store for reference
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize parent OpenAIClient with OpenRouter configuration
        super().__init__(
            api_key=final_api_key,
            model=model,
            base_url=base_url,
            default_headers=default_headers if default_headers else None
        )
