"""
Tests for the prompts module.
"""

import json
import re

import pytest

from acton_agent.agent.prompts import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    build_system_prompt,
    get_default_system_prompt,
)


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_build_default_prompt(self):
        """Test building prompt with no custom instructions."""
        prompt = build_system_prompt()

        assert "helpful AI agent" in prompt
        assert "RESPONSE FORMAT" in prompt
        assert "AgentPlan" in prompt
        assert "AgentStep" in prompt
        assert "AgentFinalResponse" in prompt

    def test_build_custom_prompt(self):
        """Test building prompt with custom instructions."""
        custom = "You are a math expert."
        prompt = build_system_prompt(custom_instructions=custom)

        assert custom in prompt
        assert "RESPONSE FORMAT" in prompt

    def test_prompt_contains_schemas(self):
        """Test that prompt contains JSON schemas."""
        prompt = build_system_prompt()

        # Should contain schema information
        assert "Schema" in prompt or "schema" in prompt
        assert "thought" in prompt
        assert "plan" in prompt
        assert "tool_calls" in prompt
        assert "final_answer" in prompt

    def test_prompt_contains_examples(self):
        """
        Verify the generated system prompt includes JSON example blocks and an examples section.

        Checks that the prompt contains at least one '```json' code block and either the string 'EXAMPLES' or the word 'example' (case-insensitive).
        """
        prompt = build_system_prompt()

        assert "```json" in prompt
        assert "EXAMPLES" in prompt or "example" in prompt.lower()

    def test_prompt_contains_rules(self):
        """Test that prompt contains critical rules."""
        prompt = build_system_prompt()

        assert "RULES" in prompt or "rule" in prompt.lower()
        assert "ALWAYS" in prompt or "must" in prompt.lower()


class TestGetDefaultSystemPrompt:
    """Tests for get_default_system_prompt function."""

    def test_get_default_prompt(self):
        """Test getting default system prompt."""
        prompt = get_default_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "helpful AI agent" in prompt

    def test_default_prompt_equals_build_with_default(self):
        """Test that get_default matches build with default instructions."""
        default_prompt = get_default_system_prompt()
        built_prompt = build_system_prompt(custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS)

        assert default_prompt == built_prompt


class TestPromptFormat:
    """Tests for prompt formatting and structure."""

    def test_prompt_sections_separated(self):
        """Test that prompt has clear section separation."""
        prompt = build_system_prompt()

        # Should have section separators
        assert "=" * 60 in prompt or "-" * 60 in prompt or "\n\n" in prompt

    def test_prompt_has_tool_section(self):
        """Test that prompt mentions tools."""
        prompt = build_system_prompt()

        assert "tool" in prompt.lower() or "TOOL" in prompt

    def test_json_examples_valid(self):
        """
        Checks that any JSON code blocks included in the built system prompt parse as valid JSON.

        Fails the test if any JSON block in the prompt is not valid JSON, reporting the parse error and the offending block.
        """
        prompt = build_system_prompt()

        # Extract JSON code blocks
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", prompt, re.DOTALL)

        # Each should be valid JSON
        for block in json_blocks:
            try:
                json.loads(block)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in prompt: {e}\nBlock: {block}")
