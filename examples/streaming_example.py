#!/usr/bin/env python3
"""
Example: Streaming responses from the agent

This example demonstrates how to use the streaming feature to get
real-time responses from the agent as they are generated.
"""

import os
from acton_agent import Agent
from acton_agent.client import OpenAIClient
from acton_agent.tools import RequestsTool


def main():
    # Initialize the OpenAI client
    """
    Run an interactive command-line demo that showcases real-time streaming responses from an agent.

    This function reads OPENAI_API_KEY from the environment and, if present, creates an OpenAI client and a streaming Agent, registers a sample HTTP RequestsTool, and runs three interactive examples:
    1) a simple streaming story,
    2) a streaming request that demonstrates tool usage and shows tool-related status events,
    3) multiple short queries streamed sequentially.
    The demo prints agent tokens and status messages to stdout and pauses for user input between examples. If OPENAI_API_KEY is not set, the function prints an error and returns early.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    # Create an agent with streaming enabled
    agent = Agent(
        llm_client=client,
        system_prompt="You are a helpful assistant. Provide detailed and informative responses.",
        stream=True,  # Enable streaming
    )

    # Add a tool for demonstration
    posts_tool = RequestsTool(
        name="get_posts",
        description="Fetch posts from JSONPlaceholder API",
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

    agent.register_tool(posts_tool)

    print("\n" + "=" * 70)
    print("⚡ Welcome to the Streaming Demo!")
    print("=" * 70)
    print("\nThis demo shows real-time streaming responses from your agent.")
    print("Watch as the agent's thoughts appear word-by-word, just like ChatGPT!\n")
    input("Press Enter to start the demo...")
    print()

    # Example 1: Simple streaming without tools
    print("\n" + "─" * 70)
    print("✨ First, let's see the agent create a story in real-time...")
    print("─" * 70)
    query = "Tell me a short story about a robot learning to paint"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)

    for event in agent.run_stream(query):
        if event.type == "token":
            # Print tokens as they arrive
            print(event.content, end="", flush=True)
        elif event.type == "final_response":
            # Final response received
            print("\n")
            break

    input("\nPress Enter to continue...")
    print()

    # Example 2: Streaming with tool calls
    print("\n" + "─" * 70)
    print("🔧 Now watch as the agent uses tools while streaming...")
    print("─" * 70)
    query = "Get me posts from user ID 1 and summarize them"
    print(f"\n💬 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)

    for event in agent.run_stream(query):
        if event.type == "token":
            # Print tokens as they arrive
            print(event.content, end="", flush=True)
        elif event.type == "agent_plan":
            # Agent is making a plan (may include tool calls)
            if event.plan.tool_calls:
                print("\n\n   🔧 [Planning to use tools...]", flush=True)
        elif event.type == "tool_results":
            # Tool execution complete
            print("\n   ✓ [Got the data!]\n\n🤖 Agent: ", flush=True)
        elif event.type == "final_response":
            print("\n")
            break

    input("\nPress Enter to continue...")
    print()

    # Example 3: Multiple queries with streaming
    print("\n" + "─" * 70)
    print("💫 Finally, let's have a quick conversation...")
    print("─" * 70)

    queries = [
        "What is 2 + 2?",
        "Explain in one sentence what an API is",
        "List 3 benefits of using AI agents",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n💬 You: {query}")
        print("🤖 Agent: ", end="", flush=True)

        for event in agent.run_stream(query):
            if event.type == "token":
                print(event.content, end="", flush=True)
            elif event.type == "final_response":
                print("\n")
                break

        # Reset agent conversation for next query (optional)
        # agent.reset()

    print("\n" + "=" * 70)
    print("✅ Demo completed! You experienced real-time streaming responses.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
