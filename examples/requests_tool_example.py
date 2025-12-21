#!/usr/bin/env python3
"""
Example: Using RequestsTool to interact with REST APIs

This example demonstrates how to use the RequestsTool to enable your agent
to make HTTP API calls. We'll use the JSONPlaceholder API as an example.
"""

import os

from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import RequestsTool, create_api_tool


def main():
    # Initialize the OpenAI client
    # Make sure to set your OPENAI_API_KEY environment variable
    """
    Run an interactive demo that showcases an Acton Agent using RequestsTool and create_api_tool against the JSONPlaceholder API.

    Initializes an OpenAI client from the OPENAI_API_KEY environment variable (exits early and prints an error if the key is missing), builds an agent, registers example API tools (posts, single post, comments, user, and user posts), and executes a sequence of five interactive demo queries that print prompts, invoke the agent, display results, and pause for user input between steps.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create an agent
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful assistant that can fetch data from APIs. "
        "When presenting data, format it nicely for readability.",
    )

    # Example 1: Create a RequestsTool for fetching all posts
    posts_tool = RequestsTool(
        name="get_posts",
        description="Fetch posts from JSONPlaceholder API. Can optionally filter by user ID.",
        method="GET",
        url_template="https://jsonplaceholder.typicode.com/posts",
        query_params_schema={
            "userId": {
                "type": "number",
                "description": "Filter posts by user ID",
                "required": False,
            }
        },
    )

    # Example 2: Create a tool for fetching a specific post using route parameters
    # Route parameters are automatically extracted from {post_id} in the URL
    post_tool = create_api_tool(
        name="get_post",
        description="Fetch a specific post by ID using route parameters",
        endpoint="https://jsonplaceholder.typicode.com/posts/{post_id}",
        method="GET",
    )

    # Example 3: Create a tool for fetching comments on a post (route parameter: post_id)
    comments_tool = create_api_tool(
        name="get_comments",
        description="Fetch comments for a specific post using route parameters",
        endpoint="https://jsonplaceholder.typicode.com/posts/{post_id}/comments",
        method="GET",
    )

    # Example 4: Create a tool for fetching user information (route parameter: user_id)
    user_tool = create_api_tool(
        name="get_user",
        description="Fetch user information by user ID using route parameters",
        endpoint="https://jsonplaceholder.typicode.com/users/{user_id}",
        method="GET",
    )

    # Example 5: Create a tool for fetching a specific user's specific post
    # (multiple route parameters: user_id and post_id with query params)
    user_posts_tool = RequestsTool(
        name="get_user_posts",
        description="Fetch posts from a specific user",
        method="GET",
        url_template="https://jsonplaceholder.typicode.com/posts",
        query_params_schema={
            "userId": {
                "type": "number",
                "description": "User ID to filter posts",
                "required": True,
            },
            "_limit": {
                "type": "number",
                "description": "Limit number of results",
                "required": False,
            },
        },
    )

    # Register all tools with the agent
    agent.register_tool(posts_tool)
    agent.register_tool(post_tool)
    agent.register_tool(comments_tool)
    agent.register_tool(user_tool)
    agent.register_tool(user_posts_tool)

    print("\n" + "=" * 70)
    print("🌐 Welcome to the API Integration Example!")
    print("=" * 70)
    print("\nThis demo shows how your agent can interact with REST APIs.")
    print("We'll be using JSONPlaceholder API with route parameters and query params.\n")
    input("Press Enter to start the demo...")
    print()

    # Query 1: Get posts from a specific user using query parameters
    print("\n" + "─" * 70)
    print("📝 Let's fetch posts using query parameters...")
    print("─" * 70)
    query = "Get me the posts from user ID 1"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 2: Get a specific post using route parameters
    print("\n" + "─" * 70)
    print("🔍 Now let's use route parameters to get a specific post...")
    print("─" * 70)
    query = "Tell me about post number 5"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 3: Get comments using route parameters
    print("\n" + "─" * 70)
    print("💭 Let's fetch comments using route parameters...")
    print("─" * 70)
    query = "What are the comments on post 1?"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 4: Complex query involving multiple API calls with route parameters
    print("\n" + "─" * 70)
    print("🎯 Let's try multiple API calls with route parameters...")
    print("─" * 70)
    query = "Get post 10, then tell me who wrote it"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()
    input("\nPress Enter to continue...")

    # Query 5: Show route parameters in action
    print("\n" + "─" * 70)
    print("🌟 Finally, let's demonstrate the power of route parameters!")
    print("─" * 70)
    query = "Show me user 2's information and their first 3 posts"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    result = agent.run(query)
    print(result)
    print()

    print("\n" + "=" * 70)
    print("✅ Demo completed! The agent used route parameters & query params seamlessly.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
