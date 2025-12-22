#!/usr/bin/env python3
"""
Example: Controlling Logging Output

This example demonstrates how to control logging in Acton Agent using
the verbose parameter and ACTON_LOG_LEVEL environment variable.
"""

import os
from acton_agent import Agent, FunctionTool
from acton_agent.client import OpenAIClient


def simple_calculator(a: float, b: float, operation: str) -> float:
    """Simple calculator function."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else 0,
    }
    return operations.get(operation, 0)


def example_logging_disabled():
    """Example with logging disabled (default behavior)."""
    print("\n=== Example 1: Logging Disabled (Default) ===\n")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable to run this example")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create agent with verbose=False (default)
    # No logging output will be shown
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful calculator assistant",
        verbose=False,  # Logging disabled - this is the default
    )

    # Register a tool
    agent.register_tool(
        FunctionTool(
            name="calculator",
            description="Perform basic arithmetic operations",
            func=simple_calculator,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                },
                "required": ["a", "b", "operation"],
            },
        )
    )

    print("Running agent with logging disabled...")
    print("(You won't see any debug/info log messages)\n")

    # Run the agent - no logging output will appear
    # result = agent.run("What is 15 multiplied by 3?")
    # print(f"Result: {result}\n")

    print("✓ Agent ran successfully with no logging output\n")


def example_logging_enabled_default_level():
    """Example with logging enabled at default INFO level."""
    print("\n=== Example 2: Logging Enabled (Default INFO Level) ===\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable to run this example")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create agent with verbose=True
    # Logging will be enabled at INFO level (default)
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful calculator assistant",
        verbose=True,  # Enable logging with default INFO level
    )

    agent.register_tool(
        FunctionTool(
            name="calculator",
            description="Perform basic arithmetic operations",
            func=simple_calculator,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                },
                "required": ["a", "b", "operation"],
            },
        )
    )

    print("Running agent with logging enabled at INFO level...")
    print("(You'll see INFO, SUCCESS, WARNING, and ERROR messages)\n")

    # Run the agent - logging output will appear
    # result = agent.run("What is 20 plus 5?")
    # print(f"\nResult: {result}\n")

    print("✓ Agent ran with INFO-level logging\n")


def example_logging_with_debug_level():
    """Example with logging enabled at DEBUG level using environment variable."""
    print("\n=== Example 3: Logging Enabled (DEBUG Level via Environment Variable) ===\n")

    # Set the log level via environment variable
    os.environ["ACTON_LOG_LEVEL"] = "DEBUG"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable to run this example")
        # Clean up
        del os.environ["ACTON_LOG_LEVEL"]
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create agent with verbose=True
    # The log level will be DEBUG because of the environment variable
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful calculator assistant",
        verbose=True,  # Enable logging - level controlled by ACTON_LOG_LEVEL
    )

    agent.register_tool(
        FunctionTool(
            name="calculator",
            description="Perform basic arithmetic operations",
            func=simple_calculator,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                },
                "required": ["a", "b", "operation"],
            },
        )
    )

    print("Running agent with logging enabled at DEBUG level...")
    print("(You'll see detailed DEBUG messages in addition to INFO/SUCCESS/WARNING/ERROR)\n")

    # Run the agent - verbose logging output will appear
    # result = agent.run("What is 100 divided by 4?")
    # print(f"\nResult: {result}\n")

    print("✓ Agent ran with DEBUG-level logging\n")

    # Clean up environment variable
    del os.environ["ACTON_LOG_LEVEL"]


def example_logging_with_error_level():
    """Example with logging enabled at ERROR level."""
    print("\n=== Example 4: Logging Enabled (ERROR Level Only) ===\n")

    # Set the log level to ERROR to only see errors
    os.environ["ACTON_LOG_LEVEL"] = "ERROR"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable to run this example")
        # Clean up
        del os.environ["ACTON_LOG_LEVEL"]
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create agent with verbose=True
    # Only ERROR and CRITICAL messages will be shown
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful calculator assistant",
        verbose=True,  # Enable logging - only ERROR level messages will show
    )

    agent.register_tool(
        FunctionTool(
            name="calculator",
            description="Perform basic arithmetic operations",
            func=simple_calculator,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                },
                "required": ["a", "b", "operation"],
            },
        )
    )

    print("Running agent with logging enabled at ERROR level...")
    print("(You'll only see ERROR and CRITICAL messages)\n")

    # Run the agent - only errors will be logged
    # result = agent.run("What is 7 times 6?")
    # print(f"\nResult: {result}\n")

    print("✓ Agent ran with ERROR-level logging (minimal output)\n")

    # Clean up environment variable
    del os.environ["ACTON_LOG_LEVEL"]


def main():
    """
    Demonstrate logging control in Acton Agent.

    Shows different ways to configure logging:
    1. Disabled (default) - no logging output
    2. Enabled with default INFO level
    3. Enabled with DEBUG level via ACTON_LOG_LEVEL
    4. Enabled with ERROR level for minimal output
    """
    print("=" * 70)
    print("Acton Agent - Logging Configuration Examples")
    print("=" * 70)

    # Example 1: Logging disabled (default)
    example_logging_disabled()

    # Example 2: Logging enabled with default level
    example_logging_enabled_default_level()

    # Example 3: Logging with DEBUG level
    example_logging_with_debug_level()

    # Example 4: Logging with ERROR level only
    example_logging_with_error_level()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("""
Logging Control Options:

1. Disable Logging (Default):
   agent = Agent(llm_client=client, verbose=False)

2. Enable Logging (INFO level by default):
   agent = Agent(llm_client=client, verbose=True)

3. Enable Logging with Custom Level:
   export ACTON_LOG_LEVEL=DEBUG  # or INFO, WARNING, ERROR, etc.
   agent = Agent(llm_client=client, verbose=True)

Valid ACTON_LOG_LEVEL values:
- TRACE: Most detailed logging
- DEBUG: Detailed debugging information
- INFO: General informational messages (default)
- SUCCESS: Success messages
- WARNING: Warning messages
- ERROR: Error messages only
- CRITICAL: Critical errors only

Note: ACTON_LOG_LEVEL is only used when verbose=True
    """)


if __name__ == "__main__":
    main()
