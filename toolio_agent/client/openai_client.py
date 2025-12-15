"""
OpenAI LLM Client implementation with streaming support.
"""

import os
from typing import Generator, List, Optional

from openai import OpenAI

from ..agent.models import Message


class OpenAIClient:
    """
    Base LLM client implementation for OpenAI-compatible APIs with streaming support.

    This client supports both regular and streaming responses. When streaming is enabled,
    it yields token chunks as they arrive from the API.

    Example:
        ```python
        # Non-streaming
        client = OpenAIClient(
            api_key="your-api-key",
            model="gpt-4o"
        )

        messages = [Message(role="user", content="Hello!")]
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
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        default_headers: Optional[dict] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: Your OpenAI API key (reads from OPENAI_API_KEY env var if not provided)
            model: The model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
            base_url: API base URL (default: https://api.openai.com/v1)
            organization: Optional organization ID
            default_headers: Optional default headers to include in all requests

        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY env var is not set
        """
        # Get API key from parameter or environment variable
        final_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not final_api_key:
            raise ValueError(
                "OpenAI API key must be provided either as 'api_key' parameter "
                "or via OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=final_api_key,
            organization=organization,
            default_headers=default_headers,
        )
        self.model = model

    def call(self, messages: List[Message], **kwargs) -> str:
        """
        Make a synchronous call to the LLM.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens)

        Returns:
            The LLM's response as a string

        Raises:
            Exception: If the LLM call fails
        """
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        completion = self.client.chat.completions.create(
            model=self.model, messages=message_dicts, stream=False, **kwargs
        )

        return completion.choices[0].message.content

    def call_stream(
        self, messages: List[Message], **kwargs
    ) -> Generator[str, None, None]:
        """
        Make a streaming call to the LLM, yielding tokens as they arrive.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens)

        Yields:
            Token chunks as they arrive from the API

        Raises:
            Exception: If the LLM call fails
        """
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        stream = self.client.chat.completions.create(
            model=self.model, messages=message_dicts, stream=True, **kwargs
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
