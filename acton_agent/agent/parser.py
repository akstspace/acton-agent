"""
Response parser for the AI Agent Framework.

This module provides functionality to parse and validate LLM responses
into structured response objects.
"""

import json
import re
from typing import Optional, Union

from loguru import logger

from .models import AgentFinalResponse, AgentPlan, AgentStep


class ResponseParser:
    """
    Parse and validate LLM responses into structured response objects.

    Handles JSON parsing, markdown code block removal, and supports multiple
    response types: AgentPlan, AgentStep, and AgentFinalResponse.
    """

    @staticmethod
    def parse(
        response_text: str,
    ) -> Union[AgentPlan, AgentStep, AgentFinalResponse]:
        """
        Parse LLM response text into a structured agent response model.

        Parameters:
            response_text (str): Raw LLM output; may contain JSON inside Markdown code fences.

        Returns:
            AgentPlan | AgentStep | AgentFinalResponse: An instantiated response model inferred from the parsed JSON:
                - AgentPlan when the parsed data contains a plan.
                - AgentStep when the parsed data contains one or more tool_calls.
                - AgentFinalResponse when the parsed data contains a final_answer or when no structured type is detected.
                If JSON parsing fails, returns an AgentFinalResponse with the original raw text. If an unexpected error occurs, returns an AgentFinalResponse with an error message.
        """
        try:
            response_text = response_text.strip()

            # Step 1: ALWAYS try to extract JSON from markdown code block first
            json_text = ResponseParser._extract_json_from_markdown(response_text)

            # Step 2: Parse the JSON
            data = json.loads(json_text)

            # Step 3: Detect response type and create appropriate model
            if "plan" in data:
                # This is an AgentPlan
                response = AgentPlan(**data)
                logger.debug("Parsed as AgentPlan")
                return response
            elif "final_answer" in data and data["final_answer"] is not None:
                # This is an AgentFinalResponse
                response = AgentFinalResponse(**data)
                logger.debug("Parsed as AgentFinalResponse")
                return response
            elif "tool_calls" in data and len(data.get("tool_calls", [])) > 0:
                # This is an AgentStep
                response = AgentStep(**data)
                logger.debug("Parsed as AgentStep")
                return response
            else:
                # If no recognizable structure, treat as final answer
                logger.debug("No recognizable structure, treating as AgentFinalResponse")
                return AgentFinalResponse(final_answer=response_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response, treating as final answer: {e}")
            logger.debug(f"Raw response text: {response_text[:200]}...")
            # Fallback: treat entire response as final answer
            return AgentFinalResponse(final_answer=response_text)

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Raw response text: {response_text[:200]}...")
            # Last resort fallback
            return AgentFinalResponse(final_answer=f"Error parsing response: {e!s}")

    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """
        Extract JSON text from a Markdown fenced code block if one is present.

        Searches for a fenced code block starting with ```json or ``` and returns the inner content trimmed of surrounding whitespace. Also handles single-line inline fenced blocks. If no fenced code block is found, returns the original input trimmed.

        Parameters:
            text (str): Text that may contain a Markdown fenced code block with JSON.

        Returns:
            str: The extracted JSON text from the code block, or the original trimmed text if no code block is found.
        """
        text = text.strip()

        # Pattern to match ```json or ``` at start, content, and ``` at end
        # Match ```json or just ```
        pattern = r"^```(?:json)?\s*\n(.*?)\n```\s*$"
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)

        if match:
            # Extract the content between the code block markers
            json_content = match.group(1).strip()
            logger.debug("Extracted JSON from markdown code block")
            return json_content

        # Try without newlines (in case of single line code block)
        pattern = r"^```(?:json)?\s*(.*?)```\s*$"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            json_content = match.group(1).strip()
            logger.debug("Extracted JSON from inline markdown code block")
            return json_content

        # No code block found, return original text
        logger.debug("No markdown code block found, using raw text")
        return text

    @staticmethod
    def validate_response(
        response: Union[AgentPlan, AgentStep, AgentFinalResponse],
    ) -> bool:
        """
        Validate that a parsed agent response has the required fields for its concrete type.

        AgentPlan requires a non-empty `plan`. AgentStep requires `tool_calls` and each tool call must include `id` and `tool_name`. AgentFinalResponse requires a non-empty `final_answer`.

        Parameters:
            response (AgentPlan | AgentStep | AgentFinalResponse): The parsed response object to validate.

        Returns:
            bool: `True` if the response meets the validation rules for its type, `False` otherwise.
        """
        # AgentPlan must have plan
        if isinstance(response, AgentPlan):
            if not response.plan or len(response.plan) == 0:
                logger.warning("Invalid AgentPlan: must have non-empty plan")
                return False

        # AgentStep must have tool_calls
        elif isinstance(response, AgentStep):
            if not response.has_tool_calls:
                logger.warning("Invalid AgentStep: must have tool_calls")
                return False
            # Validate each tool call
            for tool_call in response.tool_calls:
                if not tool_call.id or not tool_call.tool_name:
                    logger.warning("Invalid tool call: missing id or tool_name")
                    return False

        # AgentFinalResponse must have final_answer
        elif isinstance(response, AgentFinalResponse) and not response.final_answer:
            logger.warning("Invalid AgentFinalResponse: must have final_answer")
            return False

        return True

    @staticmethod
    def extract_thought(
        response: Union[AgentPlan, AgentStep, AgentFinalResponse],
    ) -> Optional[str]:
        """
        Retrieve the thought text from a response object.

        For AgentStep, returns the `tool_thought` attribute.
        For AgentFinalResponse, returns the `thought` attribute if present.
        For AgentPlan, returns None (plans don't have thought attribute).

        Parameters:
            response (Union[AgentPlan, AgentStep, AgentFinalResponse]): The response to extract thought from.

        Returns:
            Optional[str]: The thought text if available, `None` otherwise.
        """
        if isinstance(response, AgentStep):
            return getattr(response, "tool_thought", None)
        return getattr(response, "thought", None)
