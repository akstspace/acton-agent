#!/usr/bin/env python3
"""
Example: Making HTTP requests using FunctionTool

This example demonstrates how to create tools that make HTTP requests
using the FunctionTool wrapper. This is useful for integrating with
external APIs without requiring the requests library.
"""

import json
import os
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from pydantic import Field

from acton_agent import Agent, FunctionTool
from acton_agent.client import OpenAIClient
from acton_agent.tools.models import ToolInputSchema


class GetWeatherInput(ToolInputSchema):
    """Input schema for weather API requests."""

    city: str = Field(description="City name to get weather for")
    units: str = Field(
        default="metric", description="Temperature units (metric or imperial)"
    )


class GetJokeInput(ToolInputSchema):
    """Input schema for joke API requests."""

    category: str = Field(
        default="general", description="Category of joke (e.g., general, programming)"
    )


def get_weather(city: str, units: str = "metric") -> dict[str, Any]:
    """
    Get current weather for a city using a public weather API.

    This is a simulated example - in production, you would call a real API.

    Args:
        city: City name to get weather for
        units: Temperature units (metric or imperial)

    Returns:
        Weather data as a dictionary
    """
    # Simulated weather data
    # In production, you would make an actual HTTP request like:
    # url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"
    # req = Request(url)
    # with urlopen(req) as response:
    #     data = json.loads(response.read().decode())
    #     return data

    # Simulated response
    temp = 20 if units == "metric" else 68
    return {
        "city": city,
        "temperature": temp,
        "units": units,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": 10,
    }


def get_random_joke(category: str = "general") -> dict[str, str]:
    """
    Get a random joke from a public API.

    This demonstrates making a real HTTP request using urllib.

    Args:
        category: Category of joke

    Returns:
        Joke data as a dictionary
    """
    try:
        # Using a public joke API that doesn't require authentication
        url = "https://official-joke-api.appspot.com/random_joke"

        req = Request(url)
        req.add_header("User-Agent", "Acton-Agent/1.0")

        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {
                "category": data.get("type", category),
                "setup": data.get("setup", ""),
                "punchline": data.get("punchline", ""),
            }

    except (HTTPError, URLError) as e:
        return {"error": f"Failed to fetch joke: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def search_github_repos(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Search GitHub repositories using the GitHub API.

    This demonstrates making HTTP requests with query parameters.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        Search results as a dictionary
    """
    try:
        # GitHub API endpoint
        url = f"https://api.github.com/search/repositories?q={query}&per_page={limit}"

        req = Request(url)
        req.add_header("User-Agent", "Acton-Agent/1.0")
        req.add_header("Accept", "application/vnd.github.v3+json")

        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            # Extract relevant information
            repos = []
            for item in data.get("items", [])[:limit]:
                repos.append(
                    {
                        "name": item.get("full_name"),
                        "description": item.get("description"),
                        "stars": item.get("stargazers_count"),
                        "url": item.get("html_url"),
                    }
                )

            return {
                "total_count": data.get("total_count", 0),
                "repositories": repos,
            }

    except (HTTPError, URLError) as e:
        return {"error": f"Failed to search GitHub: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def main():
    """Run the HTTP request example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Create tools using FunctionTool
    weather_tool = FunctionTool(
        name="get_weather",
        description="Get current weather information for a city",
        func=get_weather,
        input_schema=GetWeatherInput,
    )

    joke_tool = FunctionTool(
        name="get_joke",
        description="Get a random joke from the internet",
        func=get_random_joke,
        input_schema=GetJokeInput,
    )

    # Create GitHub search tool without input schema (using dict-based parameters)
    github_tool = FunctionTool(
        name="search_github",
        description="Search GitHub repositories by query",
        func=search_github_repos,
    )

    # Create client
    client = OpenAIClient(api_key=api_key, model="gpt-4")

    # Create agent with all tools
    agent = Agent(client=client, tools=[weather_tool, joke_tool, github_tool])

    # Example 1: Get weather
    print("\n=== Example 1: Weather Information ===")
    response = agent.run("What's the weather like in London?")
    print(response)

    # Example 2: Get a joke
    print("\n=== Example 2: Random Joke ===")
    response = agent.run("Tell me a programming joke")
    print(response)

    # Example 3: Search GitHub
    print("\n=== Example 3: GitHub Search ===")
    response = agent.run(
        "Find the top 3 most popular Python repositories on GitHub"
    )
    print(response)


if __name__ == "__main__":
    main()
