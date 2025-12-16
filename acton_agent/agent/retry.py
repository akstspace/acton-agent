"""
Retry configuration for the AI Agent Framework.

This module provides configuration for retry logic using the tenacity library.
"""

from typing import Callable

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RetryConfig(BaseModel):
    """
    Configuration for retry logic using tenacity.

    Attributes:
        max_attempts: Maximum number of retry attempts
        wait_multiplier: Multiplier for exponential backoff
        wait_min: Minimum wait time in seconds
        wait_max: Maximum wait time in seconds
    """

    max_attempts: int = Field(
        default=3, ge=1, description="Maximum number of retry attempts"
    )
    wait_multiplier: float = Field(
        default=1.0, ge=0, description="Multiplier for exponential backoff"
    )
    wait_min: float = Field(
        default=1.0, ge=0, description="Minimum wait time in seconds"
    )
    wait_max: float = Field(
        default=10.0, ge=0, description="Maximum wait time in seconds"
    )

    def create_retry_decorator(self, exception_types: tuple = (Exception,)):
        """
        Create a tenacity retry decorator configured from this RetryConfig.

        Parameters:
            exception_types (tuple): Exception classes that should trigger a retry.

        Returns:
            A tenacity retry decorator that retries on the given exception types, uses exponential backoff with this instance's multiplier/min/max, stops after this instance's max_attempts, and re-raises the final exception when retries are exhausted.
        """
        return retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.wait_multiplier, min=self.wait_min, max=self.wait_max
            ),
            retry=retry_if_exception_type(exception_types),
            reraise=True,
        )

    def wrap_function(
        self, func: Callable, exception_types: tuple = (Exception,)
    ) -> Callable:
        """
        Wraps a callable with retry behavior based on this configuration.

        Parameters:
            func (Callable): The function or callable to wrap.
            exception_types (tuple): Exception classes that trigger a retry; defaults to (Exception,).

        Returns:
            Callable: The input callable wrapped with the configured tenacity retry decorator.
        """
        decorator = self.create_retry_decorator(exception_types)
        return decorator(func)
