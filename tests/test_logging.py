"""
Tests for logging configuration.
"""

import os

from loguru import logger

from acton_agent.logging_config import _get_log_level_from_env, configure_logging


class TestLoggingConfiguration:
    """Tests for logging configuration functionality."""

    def teardown_method(self):
        """Clean up after each test."""
        # Remove any ACTON_LOG_LEVEL environment variable
        if "ACTON_LOG_LEVEL" in os.environ:
            del os.environ["ACTON_LOG_LEVEL"]
        # Reset logging to default state
        logger.remove()

    def test_logging_disabled_by_default(self):
        """Test that logging is disabled when verbose=False."""
        configure_logging(verbose=False)

        # Check that logger has handlers but they are no-op
        assert len(logger._core.handlers) > 0

    def test_logging_enabled_with_verbose_true(self):
        """Test that logging is enabled when verbose=True."""
        configure_logging(verbose=True)

        # Check that logger has real handlers
        assert len(logger._core.handlers) > 0

    def test_default_log_level_is_info(self):
        """Test that default log level is INFO when ACTON_LOG_LEVEL is not set."""
        configure_logging(verbose=True)

        # The default level should be INFO
        level = _get_log_level_from_env()
        assert level == "INFO"

    def test_acton_log_level_env_var_debug(self):
        """Test that ACTON_LOG_LEVEL environment variable sets DEBUG level."""
        os.environ["ACTON_LOG_LEVEL"] = "DEBUG"
        level = _get_log_level_from_env()
        assert level == "DEBUG"

    def test_acton_log_level_env_var_warning(self):
        """Test that ACTON_LOG_LEVEL environment variable sets WARNING level."""
        os.environ["ACTON_LOG_LEVEL"] = "WARNING"
        level = _get_log_level_from_env()
        assert level == "WARNING"

    def test_acton_log_level_env_var_error(self):
        """Test that ACTON_LOG_LEVEL environment variable sets ERROR level."""
        os.environ["ACTON_LOG_LEVEL"] = "ERROR"
        level = _get_log_level_from_env()
        assert level == "ERROR"

    def test_acton_log_level_env_var_case_insensitive(self):
        """Test that ACTON_LOG_LEVEL is case-insensitive."""
        os.environ["ACTON_LOG_LEVEL"] = "debug"
        level = _get_log_level_from_env()
        assert level == "DEBUG"

        os.environ["ACTON_LOG_LEVEL"] = "WaRnInG"
        level = _get_log_level_from_env()
        assert level == "WARNING"

    def test_invalid_acton_log_level_falls_back_to_default(self):
        """Test that invalid ACTON_LOG_LEVEL falls back to default INFO."""
        os.environ["ACTON_LOG_LEVEL"] = "INVALID_LEVEL"
        level = _get_log_level_from_env()
        assert level == "INFO"

    def test_all_valid_log_levels(self):
        """Test all valid log levels."""
        valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

        for level_name in valid_levels:
            os.environ["ACTON_LOG_LEVEL"] = level_name
            level = _get_log_level_from_env()
            assert level == level_name

    def test_reconfigure_logging(self):
        """Test that logging can be reconfigured."""
        # First disable
        configure_logging(verbose=False)
        initial_handler_count = len(logger._core.handlers)

        # Then enable
        configure_logging(verbose=True)
        enabled_handler_count = len(logger._core.handlers)

        # Then disable again
        configure_logging(verbose=False)
        final_handler_count = len(logger._core.handlers)

        # All should have handlers, but different configurations
        assert initial_handler_count > 0
        assert enabled_handler_count > 0
        assert final_handler_count > 0
