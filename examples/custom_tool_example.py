#!/usr/bin/env python3
"""
Example: Creating custom tools by subclassing the Tool class

This example demonstrates how to create your own custom tool classes
for more complex or specialized functionality.
"""

import os
import random
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.agent import Tool


class WeatherTool(Tool):
    """Custom tool for getting weather information (simulated)."""

    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a city. Returns temperature and conditions.",
        )

    def execute(self, parameters: dict) -> str:
        """Execute the tool with the given parameters."""
        city = parameters.get("city", "Unknown")

        # Simulate weather data (in a real implementation, you would call a weather API)
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "clear"]
        temperature = random.randint(50, 90)
        condition = random.choice(conditions)

        return f"The weather in {city} is {condition} with a temperature of {temperature}°F"

    def get_schema(self) -> dict:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Name of the city"}
            },
            "required": ["city"],
        }


class DiceRollerTool(Tool):
    """Custom tool for rolling dice."""

    def __init__(self):
        super().__init__(
            name="roll_dice",
            description="Roll dice with specified number of sides and count. Returns the results.",
        )

    def execute(self, parameters: dict) -> str:
        """Execute the tool with the given parameters."""
        num_dice = parameters.get("num_dice", 1)
        num_sides = parameters.get("num_sides", 6)

        if num_dice < 1 or num_dice > 100:
            return "Error: Number of dice must be between 1 and 100"

        if num_sides < 2 or num_sides > 1000:
            return "Error: Number of sides must be between 2 and 1000"

        rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
        total = sum(rolls)

        result = f"Rolling {num_dice}d{num_sides}:\n"
        result += f"Rolls: {rolls}\n"
        result += f"Total: {total}"

        return result

    def get_schema(self) -> dict:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "num_dice": {
                    "type": "integer",
                    "description": "Number of dice to roll",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 100,
                },
                "num_sides": {
                    "type": "integer",
                    "description": "Number of sides on each die",
                    "default": 6,
                    "minimum": 2,
                    "maximum": 1000,
                },
            },
            "required": [],
        }


class TextAnalyzerTool(Tool):
    """Custom tool for analyzing text properties."""

    def __init__(self):
        super().__init__(
            name="analyze_text",
            description="Analyze text and return various statistics: character count, word count, sentence count, etc.",
        )

    def execute(self, parameters: dict) -> str:
        """Execute the tool with the given parameters."""
        text = parameters.get("text", "")

        if not text:
            return "Error: No text provided"

        # Calculate statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        line_count = text.count("\n") + 1

        # Find longest word
        words = text.split()
        longest_word = max(words, key=len) if words else ""

        result = "Text Analysis:\n"
        result += f"- Characters: {char_count}\n"
        result += f"- Words: {word_count}\n"
        result += f"- Sentences: {sentence_count}\n"
        result += f"- Lines: {line_count}\n"
        result += f"- Longest word: '{longest_word}' ({len(longest_word)} characters)\n"
        result += (
            f"- Average word length: {char_count / word_count:.2f} characters"
            if word_count > 0
            else "N/A"
        )

        return result

    def get_schema(self) -> dict:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"],
        }


def main():
    # Initialize the OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create an agent
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful assistant with access to various utility tools.",
    )

    # Create and register custom tools
    weather_tool = WeatherTool()
    dice_tool = DiceRollerTool()
    text_analyzer_tool = TextAnalyzerTool()

    agent.register_tool(weather_tool)
    agent.register_tool(dice_tool)
    agent.register_tool(text_analyzer_tool)

    print("\n" + "=" * 70)
    print("🎨 Welcome to the Custom Tool Demo!")
    print("=" * 70)
    print("\nThis demo shows how to create your own custom tool classes.")
    print("We've built: a weather checker, dice roller, and text analyzer!\n")
    input("Press Enter to start the demo...")
    print()

    # Query 1: Weather
    print("\n" + "─" * 70)
    print("☀️  Let's check the weather...")
    print("─" * 70)
    query = "What's the weather in San Francisco?"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 2: Dice rolling
    print("\n" + "─" * 70)
    print("🎲 Time to roll some dice!")
    print("─" * 70)
    query = "Roll 3 six-sided dice for me"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 3: Roll different dice
    print("\n" + "─" * 70)
    print("🎲 Let's try different dice...")
    print("─" * 70)
    query = "Roll 2 twenty-sided dice"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 4: Text analysis
    print("\n" + "─" * 70)
    print("📊 Now let's analyze some text...")
    print("─" * 70)
    query = "Analyze this text: 'The quick brown fox jumps over the lazy dog'"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 5: Combined queries
    print("\n" + "─" * 70)
    print("🌟 Finally, let's combine multiple custom tools!")
    print("─" * 70)
    query = "Check the weather in New York and then roll a die"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()

    print("\n" + "=" * 70)
    print("✅ Demo completed! You've seen how custom tools can extend your agent!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
