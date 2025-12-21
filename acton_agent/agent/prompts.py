"""
Prompt templates for the Agent system.

This module contains all system prompt templates used by the agent,
including response format instructions and schema definitions.
"""

import json

from .models import AgentFinalResponse, AgentPlan, AgentStep


# Constants
DEFAULT_CUSTOM_INSTRUCTIONS = """You are a helpful AI agent with access to tools.

CRITICAL - SCOPE AND EFFICIENCY:
- ONLY answer questions that can be addressed using your available tools
- If a question is outside your tool capabilities, respond immediately with: "I can only help with tasks related to my available tools. Please ask me something I can assist with using my tools."
- Do NOT attempt to answer general knowledge questions, math problems, or any topic unrelated to your tools
- Do NOT wait or iterate if a task cannot be solved with your available tools
- If tools cannot solve the user's request, explain what you CAN do instead and respond immediately
- Be efficient: If you can answer based on your capabilities without tool calls, do so immediately
- Examples:
  ✅ User: "What can you do?" → Respond immediately listing your tool capabilities
  ✅ User asks something unrelated to your tools → Respond immediately that this is outside your scope
  ❌ Don't iterate or plan when you already know the answer or that it's impossible

CRITICAL - COMPLETE INFORMATION BEFORE TOOL CALLS:
- NEVER use generic placeholders like "/path/to/", "<value>", "example", "placeholder", etc.
- If you need information (IDs, paths, names, etc.), first use appropriate search/list tools to get actual values
- ONLY call action tools after you have confirmed real, specific parameter values
- If the user's request is ambiguous, ask clarifying questions rather than guessing or using fake values"""

SEPARATOR = "=" * 60

DEFAULT_FORMAT_INSTRUCTIONS = """FINAL ANSWER FORMATTING:
When providing your final_answer, format it as well-structured markdown:
- Use proper headings (##, ###) for sections
- Use bullet points (-) or numbered lists (1.) for clarity
- Use **bold** for emphasis on important information
- Use code blocks with language tags for code snippets
- For images, use markdown syntax with alt text: ![Description](url)
- For responsive images, consider using HTML with width attributes when needed
- Keep paragraphs concise and readable
- Use tables for structured data when appropriate
- Add line breaks between sections for better readability

Example:
## Results

Here are the findings:

- **Item 1**: Description here
- **Item 2**: Another description

### Details

Additional information in a clear format.
"""

RESPONSE_FORMAT_INSTRUCTIONS_TEMPLATE = """RESPONSE FORMAT INSTRUCTIONS:

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
{{{{
  "plan": "Outline of steps to accomplish the task"
}}}}
```

For tool execution:
```json
{{{{
  "tool_thought": "reasoning for this step",
  "tool_calls": [
    {{{{
      "id": "call_1",
      "tool_name": "tool_name",
      "parameters": {{{{"param": "value"}}}}
    }}}}
  ]
}}}}
```

For final answer:
```json
{{{{
  "final_answer": "your complete answer to the user"
}}}}
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


def get_default_format_instructions() -> str:
    """
    Get default formatting instructions for final answers.

    Returns:
        str: Default markdown formatting instructions.
    """
    return DEFAULT_FORMAT_INSTRUCTIONS


def build_system_prompt(
    custom_instructions: str | None = None, final_answer_format_instructions: str | None = None
) -> str:
    """
    Builds the complete system prompt for the Agent, injecting response-format instructions, examples, critical rules, and the JSON schemas for response types.

    Parameters:
        custom_instructions (str | None): Optional top-level instruction text to place at the start of the prompt; when omitted the module's DEFAULT_CUSTOM_INSTRUCTIONS is used.
        final_answer_format_instructions (str | None): Optional final-answer formatting instructions to append (separated by a divider) if provided.

    Returns:
        str: The assembled system prompt containing the top instructions, a response-format section with pretty-printed JSON schemas for AgentPlan, AgentStep, and AgentFinalResponse, and optional final-answer formatting instructions.
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
        prompt_parts.append(DEFAULT_CUSTOM_INSTRUCTIONS)

    prompt_parts.append("\n" + SEPARATOR + "\n")

    formatted_template = RESPONSE_FORMAT_INSTRUCTIONS_TEMPLATE.replace("{{{{", "{{").replace("}}}}", "}}")
    formatted_template = formatted_template.format(
        plan_schema=plan_schema, step_schema=step_schema, final_schema=final_schema
    )

    # Add the formatted instructions
    prompt_parts.append(formatted_template)

    # Add final answer formatting instructions if provided
    if final_answer_format_instructions:
        prompt_parts.append("\n" + SEPARATOR + "\n")
        prompt_parts.append(final_answer_format_instructions)

    return "\n".join(prompt_parts)


def get_default_system_prompt() -> str:
    """
    Default system prompt that includes injected JSON schemas for AgentPlan, AgentStep, and AgentFinalResponse.

    Returns:
        str: The complete system prompt string containing response format instructions, examples, critical rules, and the embedded JSON schemas.
    """
    return build_system_prompt(custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS)
