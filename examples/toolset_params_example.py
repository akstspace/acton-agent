#!/usr/bin/env python3
"""
Example: Using ToolSet Parameters for Hidden Configuration

This example demonstrates how to use toolset_params to pass hidden
configuration (like API keys, credentials, or context) to tools without
exposing them to the LLM.
"""

import os
from acton_agent import Agent, ToolSet, FunctionTool
from acton_agent.client import OpenAIClient


# Example 1: API Key Injection
# These functions expect an api_key parameter that will be auto-injected
def fetch_weather(city: str, api_key: str) -> str:
    """Fetch weather data using an API key."""
    # In a real implementation, this would call an actual API
    # The api_key is automatically injected from toolset_params
    return f"Weather in {city}: Sunny, 72°F (authenticated with key: {api_key[:8]}...)"


def fetch_forecast(city: str, days: int, api_key: str) -> str:
    """Fetch weather forecast using an API key."""
    return f"{days}-day forecast for {city}: Clear skies (authenticated with key: {api_key[:8]}...)"


# Example 2: Database Connection Injection
def query_user(user_id: int, db_connection: str) -> str:
    """Query user information from database."""
    # db_connection is auto-injected from toolset_params
    return f"User {user_id} data from {db_connection}: John Doe, john@example.com"


def query_orders(user_id: int, limit: int, db_connection: str) -> str:
    """Query user orders from database."""
    return f"Found {limit} orders for user {user_id} in {db_connection}"


# Example 3: Session Context Injection
def get_user_preferences(setting_name: str, user_id: str, session_token: str) -> str:
    """Get user preferences using session context."""
    # user_id and session_token are auto-injected from toolset_params
    return f"Preference '{setting_name}' for user {user_id}: enabled (session: {session_token[:8]}...)"


def main():
    """
    Demonstrate toolset parameters with three different use cases:
    1. API key injection for weather tools
    2. Database connection injection for data query tools
    3. Session context injection for user preference tools
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create an agent
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful assistant with access to various data sources.",
    )

    # ToolSet 1: Weather API with hidden API key
    weather_toolset = ToolSet(
        name="weather_api",
        description="Weather data from external API",
        tools=[
            FunctionTool(
                name="current_weather",
                description="Get current weather for a city",
                func=fetch_weather,
                schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                    # Note: 'api_key' is NOT in the schema
                }
            ),
            FunctionTool(
                name="forecast",
                description="Get weather forecast for multiple days",
                func=fetch_forecast,
                schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "days": {"type": "integer", "description": "Number of days", "minimum": 1, "maximum": 10}
                    },
                    "required": ["city", "days"]
                    # Note: 'api_key' is NOT in the schema
                }
            )
        ],
        toolset_params={
            "api_key": "sk-weather-api-key-12345678"  # Hidden from LLM
        }
    )

    # ToolSet 2: Database queries with hidden connection string
    database_toolset = ToolSet(
        name="database",
        description="Query user and order data from database",
        tools=[
            FunctionTool(
                name="get_user",
                description="Get user information by ID",
                func=query_user,
                schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "description": "User ID"}
                    },
                    "required": ["user_id"]
                    # Note: 'db_connection' is NOT in the schema
                }
            ),
            FunctionTool(
                name="get_user_orders",
                description="Get recent orders for a user",
                func=query_orders,
                schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "description": "User ID"},
                        "limit": {"type": "integer", "description": "Max number of orders", "default": 10}
                    },
                    "required": ["user_id"]
                    # Note: 'db_connection' is NOT in the schema
                }
            )
        ],
        toolset_params={
            "db_connection": "postgresql://user:pass@localhost/mydb"  # Hidden from LLM
        }
    )

    # ToolSet 3: User preferences with session context
    preferences_toolset = ToolSet(
        name="user_preferences",
        description="Access user preferences and settings",
        tools=[
            FunctionTool(
                name="get_preference",
                description="Get a user preference or setting",
                func=get_user_preferences,
                schema={
                    "type": "object",
                    "properties": {
                        "setting_name": {"type": "string", "description": "Name of the setting to retrieve"}
                    },
                    "required": ["setting_name"]
                    # Note: 'user_id' and 'session_token' are NOT in the schema
                }
            )
        ],
        toolset_params={
            "user_id": "user-12345",
            "session_token": "sess-abcdefgh123456"
        }
    )

    # Register all toolsets
    agent.register_toolset(weather_toolset)
    agent.register_toolset(database_toolset)
    agent.register_toolset(preferences_toolset)

    print("\n" + "=" * 70)
    print("🔐 Welcome to the ToolSet Parameters Demo!")
    print("=" * 70)
    print("\nThis demo shows how to use toolset_params to inject hidden configuration")
    print("like API keys, database connections, and session data into tools.")
    print("\nThe LLM doesn't see these parameters - they're automatically injected!")
    print()
    input("Press Enter to start the demo...")
    print()

    # Demo 1: Weather API with hidden API key
    print("\n" + "─" * 70)
    print("☁️  Demo 1: Weather API with Hidden API Key")
    print("─" * 70)
    print("\nThe weather tools require an API key, but the LLM doesn't provide it.")
    print("The API key is automatically injected from toolset_params.\n")
    query = "What's the weather in Seattle?"
    print(f"💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Demo 2: Database with hidden connection string
    print("\n" + "─" * 70)
    print("🗄️  Demo 2: Database Queries with Hidden Connection String")
    print("─" * 70)
    print("\nThe database tools need a connection string, which is hidden from the LLM.")
    print("It's automatically injected when the tools execute.\n")
    query = "Get information about user 42"
    print(f"💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Demo 3: Session context
    print("\n" + "─" * 70)
    print("👤 Demo 3: User Preferences with Session Context")
    print("─" * 70)
    print("\nThe preference tool needs user_id and session_token for authentication.")
    print("These are injected automatically from toolset_params.\n")
    query = "What is my notification setting?"
    print(f"💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Demo 4: Combined query
    print("\n" + "─" * 70)
    print("🎯 Demo 4: Combined Query Using Multiple ToolSets")
    print("─" * 70)
    print("\nLet's use tools from different toolsets in one query.")
    print("Each tool gets its own hidden parameters automatically.\n")
    query = "Get the weather in Paris and also get user 10's orders"
    print(f"💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()

    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("• toolset_params keep sensitive data (API keys, credentials) hidden from LLM")
    print("• Parameters are automatically merged with LLM-provided parameters")
    print("• Each toolset can have different hidden parameters")
    print("• LLM parameters override toolset_params if there's a conflict")
    print("• Perfect for authentication, configuration, and runtime context")
    print()


if __name__ == "__main__":
    main()
