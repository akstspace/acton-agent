#!/usr/bin/env python3
"""
Example: Using FunctionTool to expose Python functions to your agent

This example demonstrates how to wrap Python functions as tools that
can be called by the agent.
"""

import os
import datetime
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import FunctionTool


def calculate(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time. For simplicity, only supports UTC."""
    if timezone != "UTC":
        return f"Only UTC timezone is supported. Current time in UTC: {datetime.datetime.utcnow().isoformat()}"
    return datetime.datetime.utcnow().isoformat()


def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def main():
    # Initialize the OpenAI client
    """
    Run an interactive demo that wraps several Python functions as callable tools for an agent.

    Sets up an OpenAI client from the OPENAI_API_KEY environment variable, constructs an Agent, registers four FunctionTools (calculator, get_time, count_words, reverse_text), and walks the user through a sequence of example queries demonstrating each tool and combined usage. If OPENAI_API_KEY is not set, prints an error message and exits early.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create an agent
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful assistant with various utility functions available.",
    )

    # Create FunctionTools with their schemas

    # Calculator tool
    calculator_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
        },
        "required": ["a", "b", "operation"],
    }

    calculator_tool = FunctionTool(
        name="calculator",
        description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
        func=calculate,
        schema=calculator_schema,
    )

    # Time tool
    time_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone for the time (only UTC supported)",
                "default": "UTC",
            }
        },
        "required": [],
    }

    time_tool = FunctionTool(
        name="get_time",
        description="Get the current time in UTC",
        func=get_current_time,
        schema=time_schema,
    )

    # Word count tool
    word_count_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to count words in"}
        },
        "required": ["text"],
    }

    word_count_tool = FunctionTool(
        name="count_words",
        description="Count the number of words in a text string",
        func=word_count,
        schema=word_count_schema,
    )

    # String reverse tool
    reverse_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to reverse"}
        },
        "required": ["text"],
    }

    reverse_tool = FunctionTool(
        name="reverse_text",
        description="Reverse a string",
        func=reverse_string,
        schema=reverse_schema,
    )

    # Register all tools
    agent.register_tool(calculator_tool)
    agent.register_tool(time_tool)
    agent.register_tool(word_count_tool)
    agent.register_tool(reverse_tool)

    print("\n" + "=" * 70)
    print("🛠️  Welcome to the Function Tool Demo!")
    print("=" * 70)
    print("\nThis demo shows how to give your agent custom Python functions as tools.")
    print(
        "Our agent has access to: calculator, clock, word counter, and text reverser.\n"
    )
    input("Press Enter to start the demo...")
    print()

    # Query 1: Simple calculation
    print("\n" + "─" * 70)
    print("🧮 Let's start with a simple math problem...")
    print("─" * 70)
    query = "What is 25 multiplied by 4?"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 2: Multi-step calculation
    print("\n" + "─" * 70)
    print("🔢 Now let's try something that requires multiple steps...")
    print("─" * 70)
    query = "Calculate 100 divided by 5, then add 10 to the result"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 3: Get current time
    print("\n" + "─" * 70)
    print("⏰ Let's check what time it is...")
    print("─" * 70)
    query = "What time is it now?"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 4: Word count
    print("\n" + "─" * 70)
    print("📊 How about counting some words?")
    print("─" * 70)
    query = "How many words are in 'The quick brown fox jumps over the lazy dog'?"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 5: String reverse
    print("\n" + "─" * 70)
    print("🔄 Let's try reversing some text...")
    print("─" * 70)
    query = "Reverse the string 'Hello World'"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 6: Multiple operations
    print("\n" + "─" * 70)
    print("🎭 Finally, let's combine multiple tools in one query!")
    print("─" * 70)
    query = "Reverse 'Python' and count how many words in the result"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()

    print("\n" + "=" * 70)
    print("✅ Demo completed! The agent used multiple Python functions to help you.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
