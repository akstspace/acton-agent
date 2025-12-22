#!/usr/bin/env python3
"""
Example: Creating custom tools by subclassing the Tool class

This example demonstrates how to create your own custom tool classes
for more complex or specialized functionality.
"""

import os
import random

from acton_agent import Agent, Tool
from acton_agent.client import OpenAIClient


class WeatherTool(Tool):
    """Custom tool for getting weather information (simulated)."""

    def __init__(self):
        """
        Initialize the WeatherTool.

        Sets the tool's name to "get_weather" and its description to indicate it provides the current temperature and conditions for a given city when executed.
        """
        super().__init__(
            name="get_weather",
            description="Get current weather for a city. Returns temperature and conditions.",
        )

    def execute(self, parameters: dict, toolset_params: dict = None) -> str:
        """
        Produce a human-readable weather summary for the specified city.

        Parameters:
            parameters (dict): Input parameters; may include the key "city" with the city name (defaults to "Unknown").
            toolset_params (dict): Optional toolset parameters; not used by this implementation.

        Returns:
            str: A sentence describing simulated current weather for the specified city, including condition and temperature in °F (e.g., "The weather in Seattle is rainy with a temperature of 62°F").
        """
        city = parameters.get("city", "Unknown")

        # Simulate weather data (in a real implementation, you would call a weather API)
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "clear"]
        temperature = random.randint(50, 90)
        condition = random.choice(conditions)

        return f"The weather in {city} is {condition} with a temperature of {temperature}°F"

    def get_schema(self) -> dict:
        """
        JSON Schema describing this tool's parameters.

        Returns:
            dict: A JSON Schema object defining a required string property "city".
        """
        return {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Name of the city"}},
            "required": ["city"],
        }


class DiceRollerTool(Tool):
    """Custom tool for rolling dice."""

    def __init__(self):
        """
        Create a DiceRollerTool configured with name "roll_dice" and a description indicating it rolls dice with configurable count and sides.
        """
        super().__init__(
            name="roll_dice",
            description="Roll dice with specified number of sides and count. Returns the results.",
        )

    def execute(self, parameters: dict, toolset_params: dict = None) -> str:
        """
        Roll a set of dice according to the provided parameters and return a formatted result.

        Parameters:
            parameters (dict): Dict that may contain:
                - "num_dice" (int, optional): Number of dice to roll; defaults to 1. Must be between 1 and 100.
                - "num_sides" (int, optional): Number of sides per die; defaults to 6. Must be between 2 and 1000.
            toolset_params (dict): Optional toolset parameters (not used in this example).

        Returns:
            str: A formatted string describing the roll (e.g., "Rolling 3d6:\nRolls: [2, 5, 4]\nTotal: 11"), or an error message if input ranges are invalid.
        """
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
        """
        Provide the JSON Schema describing valid parameters for the dice roller tool.

        The schema describes an object with two optional integer properties:
        - `num_dice`: number of dice to roll (default 1, minimum 1, maximum 100)
        - `num_sides`: number of sides on each die (default 6, minimum 2, maximum 1000)

        Returns:
            dict: A JSON Schema dictionary matching the described structure and constraints.
        """
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
        """
        Initialize the TextAnalyzerTool with its tool name and human-readable description.

        The tool is registered as "analyze_text" and described as analyzing text to return statistics such as character count, word count, sentence count, line count, longest word, and average word length.
        """
        super().__init__(
            name="analyze_text",
            description="Analyze text and return various statistics: character count, word count, sentence count, etc.",
        )

    def execute(self, parameters: dict, toolset_params: dict = None) -> str:
        """
        Analyze text and produce a formatted summary of character, word, sentence, and line counts, the longest word, and average word length.

        Parameters:
            parameters (dict): Dictionary expected to contain the key "text" with the string to analyze.
            toolset_params (dict): Optional toolset parameters (unused).

        Returns:
            str: A formatted report beginning with "Text Analysis:" that lists characters, words, sentences, lines, longest word (with length), and average word length; if "text" is empty or missing, returns "Error: No text provided".
        """
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
        result += f"- Average word length: {char_count / word_count:.2f} characters" if word_count > 0 else "N/A"

        return result

    def get_schema(self) -> dict:
        """
        Provide the JSON Schema describing this tool's input parameters.

        @returns A dict representing a JSON Schema that requires a single string property "text" (the text to analyze).
        """
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to analyze"}},
            "required": ["text"],
        }


def main():
    # Initialize the OpenAI client
    """
    Run an interactive demo that registers custom tools with an agent and executes sample queries.

    This function reads the OPENAI_API_KEY environment variable and exits early with a printed error if the key is missing. It creates an OpenAI client and an Agent, registers three custom tools (weather checker, dice roller, and text analyzer), and then interactively runs a sequence of example queries demonstrating each tool. Outputs are printed to the console and the demo pauses for user input between steps.
    """
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