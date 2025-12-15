"""
Tests for the retry module.
"""

import time

import pytest

from toolio_agent.agent.retry import RetryConfig


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.wait_multiplier == 1.0
        assert config.wait_min == 1.0
        assert config.wait_max == 10.0

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5, wait_multiplier=2.0, wait_min=0.5, wait_max=20.0
        )
        assert config.max_attempts == 5
        assert config.wait_multiplier == 2.0
        assert config.wait_min == 0.5
        assert config.wait_max == 20.0

    def test_wrap_function_success(self):
        """Test wrapping a function that succeeds."""
        config = RetryConfig(max_attempts=3)

        call_count = [0]

        def successful_func():
            call_count[0] += 1
            return "success"

        wrapped = config.wrap_function(successful_func)
        result = wrapped()

        assert result == "success"
        assert call_count[0] == 1  # Should only call once

    def test_wrap_function_retry_then_success(self):
        """Test function that fails then succeeds."""
        config = RetryConfig(max_attempts=3, wait_min=0.01, wait_max=0.1)

        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary error")
            return "success"

        wrapped = config.wrap_function(flaky_func, exception_types=(ValueError,))
        result = wrapped()

        assert result == "success"
        assert call_count[0] == 2  # Should retry once

    def test_wrap_function_max_retries(self):
        """Test function that exhausts retries."""
        config = RetryConfig(max_attempts=3, wait_min=0.01, wait_max=0.1)

        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        wrapped = config.wrap_function(always_fails, exception_types=(ValueError,))

        with pytest.raises(ValueError):
            wrapped()

        assert call_count[0] == 3  # Should try max_attempts times

    def test_create_retry_decorator(self):
        """Test creating a retry decorator."""
        config = RetryConfig(max_attempts=2, wait_min=0.01)
        decorator = config.create_retry_decorator(exception_types=(RuntimeError,))

        call_count = [0]

        @decorator
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("Retry me")
            return "done"

        result = test_func()
        assert result == "done"
        assert call_count[0] == 2

    def test_retry_only_specified_exceptions(self):
        """Test that only specified exceptions are retried."""
        config = RetryConfig(max_attempts=3, wait_min=0.01)

        call_count = [0]

        def func_with_wrong_exception():
            call_count[0] += 1
            raise TypeError("Wrong exception type")

        # Wrap to only retry ValueError
        wrapped = config.wrap_function(
            func_with_wrong_exception, exception_types=(ValueError,)
        )

        # Should not retry TypeError
        with pytest.raises(TypeError):
            wrapped()

        assert call_count[0] == 1  # Should not retry


class TestRetryWaitTimes:
    """Tests for retry wait times."""

    def test_exponential_backoff(self):
        """Test that exponential backoff is applied."""
        config = RetryConfig(
            max_attempts=3, wait_multiplier=1.0, wait_min=0.1, wait_max=1.0
        )

        call_times = []

        def timing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "done"

        wrapped = config.wrap_function(timing_func, exception_types=(ValueError,))
        wrapped()

        # Check that there were delays between calls
        assert len(call_times) == 3
        if len(call_times) >= 2:
            # There should be some delay between attempts
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # At least wait_min
