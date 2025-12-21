"""
OpenAI LLM Client implementation with streaming support.
"""

import os
from collections.abc import Generator

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
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        organization: str | None = None,
        default_headers: dict | None = None,
    ):
        """
        Initialize the OpenAIClient with API credentials and connection settings.
        
        Parameters:
            api_key (str | None): OpenAI API key; if None, the OPENAI_API_KEY environment variable is used.
            model (str): Model identifier to use for requests (e.g., "gpt-4o", "gpt-3.5-turbo").
            base_url (str): Base URL for the OpenAI-compatible API.
            organization (str | None): Optional organization ID to include with requests.
            default_headers (dict | None): Optional default HTTP headers to include on all requests.
        
        Raises:
            ValueError: If no API key is provided via the `api_key` parameter or the OPENAI_API_KEY environment variable.
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

    def call(self, messages: list[Message], **kwargs) -> str:
        """
        Request a chat completion from the configured model and return the assistant's reply.

        Parameters:
            messages (List[Message]): Conversation messages in order (each with `role` and `content`).
            **kwargs: Additional request parameters forwarded to the underlying API (e.g., `temperature`, `max_tokens`).

        Returns:
            str: The assistant's response text from the first completion choice.
        """
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        completion = self.client.chat.completions.create(
            model=self.model, messages=message_dicts, stream=False, **kwargs
        )

        if completion.choices and completion.choices[0].message.content is not None:
            return completion.choices[0].message.content
        return ""

    def call_stream(self, messages: list[Message], **kwargs) -> Generator[str, None, None]:
        """
        Stream content chunks from a chat completion for the given conversation.

        Parameters:
            messages (List[Message]): Conversation messages in chronological order.
            **kwargs: Additional parameters forwarded to the API (e.g., temperature, max_tokens).

        Yields:
            str: Incremental content chunks emitted by the model as they arrive.
        """
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        stream = self.client.chat.completions.create(model=self.model, messages=message_dicts, stream=True, **kwargs)

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content