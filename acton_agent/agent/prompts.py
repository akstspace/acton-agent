"""
Prompt templates for the Agent system.

This module contains all system prompt templates used by the agent,
including response format instructions and schema definitions.
"""

import json

from .models import AgentFinalResponse, AgentPlan, AgentStep


def build_system_prompt(custom_instructions: str = None) -> str:
    """
    Construct the system prompt used by the Agent, embedding response format instructions, examples, critical rules, and the JSON schemas for response types.
    
    Parameters:
        custom_instructions (str | None): Optional text to place at the top of the prompt; if omitted a default instruction ("You are a helpful AI agent with access to tools.") is used.
    
    Returns:
        system_prompt (str): The complete system prompt text with injected, pretty-printed JSON schemas for AgentPlan, AgentStep, and AgentFinalResponse.
    """
    # Get JSON schemas for the response types
    plan_schema = json.dumps(AgentPlan.model_json_schema(), indent=2)
    step_schema = json.dumps(AgentStep.model_json_schema(), indent=2)
    final_schema = json.dumps(AgentFinalResponse.model_json_schema(), indent=2)

    # Start with custom instructions if provided, otherwise use default
    prompt_parts = []
    if custom_instructions:
        prompt_parts.append(custom_instructions)
    else:
        prompt_parts.append("You are a helpful AI agent with access to tools.")

    prompt_parts.append("\n" + "=" * 60 + "\n")

    # Add the standard instructions
    prompt_parts.append(
        f"""RESPONSE FORMAT INSTRUCTIONS:

You MUST ALWAYS respond with valid JSON wrapped in a markdown code block. No exceptions.

You can respond with one of three types of responses:

1. AgentPlan - Initial planning response (use when you first receive a task)
2. AgentStep - Intermediate step with tool calls (use when you need to call tools)
3. AgentFinalResponse - Final answer to user (use when you have the complete answer)

RESPONSE TYPE SCHEMAS:

AgentPlan Schema:
{plan_schema}

AgentStep Schema:
{step_schema}

AgentFinalResponse Schema:
{final_schema}

RESPONSE FORMAT EXAMPLES:

For initial planning:
```json
{{
  "thought": "your reasoning about the task",
  "plan": ["step 1", "step 2", "step 3"]
}}
```

For tool execution:
```json
{{
  "thought": "reasoning for this step",
  "tool_calls": [
    {{
      "id": "call_1",
      "tool_name": "tool_name",
      "parameters": {{"param": "value"}}
    }}
  ]
}}
```

For final answer:
```json
{{
  "thought": "final reasoning (optional)",
  "final_answer": "your complete answer to the user"
}}
```

CRITICAL RULES:
1. ALWAYS wrap your JSON response in markdown code fences with 'json' language tag
2. Your response must be ONLY the JSON code block, nothing else
3. Use AgentPlan when you first receive a complex task (optional)
4. Use AgentStep when you need to call one or more tools
5. Use AgentFinalResponse when you have the complete answer
6. Each tool call must have a unique "id" field
7. Never respond with plain text - ALWAYS use one of the JSON formats above
8. The "final_answer" field MUST be a STRING containing your complete answer to the user
9. DO NOT put structured data (dicts/objects) in final_answer - format it as readable text

Available tools will be listed below."""
    )

    return "\n".join(prompt_parts)


def get_default_system_prompt() -> str:
    """
    Default system prompt that includes injected JSON schemas for AgentPlan, AgentStep, and AgentFinalResponse.
    
    Returns:
        str: The complete system prompt string containing response format instructions, examples, critical rules, and the embedded JSON schemas.
    """
    return build_system_prompt(
        custom_instructions="You are a helpful AI agent with access to tools."
    )