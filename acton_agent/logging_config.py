"""
Logging configuration for Acton Agent.

This module provides centralized logging configuration using loguru.
Logging can be controlled via the verbose parameter and ACTON_LOG_LEVEL environment variable.
"""

import os
import sys

from loguru import logger


# Valid log levels for ACTON_LOG_LEVEL environment variable
VALID_LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

# Default log level when verbose is enabled but ACTON_LOG_LEVEL is not set
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(verbose: bool = False) -> None:
    """
    Configure logging for the Acton Agent framework.

    When verbose is False, all logging is disabled to reduce noise.
    When verbose is True, logging is enabled and the level is determined by:
    1. ACTON_LOG_LEVEL environment variable (if set and valid)
    2. Default level (INFO) if environment variable is not set

    Parameters:
        verbose (bool): Whether to enable logging. Default: False.

    Example:
        ```python
        # Disable logging (default)
        configure_logging(verbose=False)

        # Enable logging with default INFO level
        configure_logging(verbose=True)

        # Enable logging with custom level via environment variable
        import os
        os.environ['ACTON_LOG_LEVEL'] = 'DEBUG'
        configure_logging(verbose=True)
        ```
    """
    # Remove all existing handlers
    logger.remove()

    if not verbose:
        # Logging disabled - add a no-op handler to sink to nowhere
        # This prevents any log messages from being displayed
        logger.add(lambda _: None, level="CRITICAL")
        return

    # Logging enabled - determine the log level
    log_level = _get_log_level_from_env()

    # Add stderr handler with the determined log level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )


def _get_log_level_from_env() -> str:
    """
    Get the log level from ACTON_LOG_LEVEL environment variable.

    Returns:
        str: The log level to use. Returns the value from ACTON_LOG_LEVEL if set and valid,
             otherwise returns DEFAULT_LOG_LEVEL.
    """
    env_level = os.environ.get("ACTON_LOG_LEVEL", "").upper()

    if env_level and env_level in VALID_LOG_LEVELS:
        return env_level

    # If environment variable is set but invalid, log a warning (if logging is enabled)
    if env_level and env_level not in VALID_LOG_LEVELS:
        # We can't use logger here as it might not be configured yet
        # Just fallback to default
        pass

    return DEFAULT_LOG_LEVEL
