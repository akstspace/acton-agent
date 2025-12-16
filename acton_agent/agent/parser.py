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
    response types: AgentPlan, AgentStep, AgentFinalResponse, or legacy AgentResponse.
    """

    @staticmethod
    def parse(
        response_text: str,
    ) -> Union[AgentPlan, AgentStep, AgentFinalResponse]:
        """
        Parse LLM response text into a structured Agent response model.
        
        Parameters:
            response_text (str): Raw text from the LLM, optionally containing JSON inside markdown code fences.
        
        Returns:
            Union[AgentPlan, AgentStep, AgentFinalResponse]: An instantiated response model inferred from the parsed JSON.
            If JSON parsing fails or an error occurs, returns an AgentFinalResponse containing the raw response text or an error message.
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
            elif "final_answer" in data and data["final_answer"] is not None:
                # This is an AgentFinalResponse
                response = AgentFinalResponse(**data)
                logger.debug("Parsed as AgentFinalResponse")
            elif "tool_calls" in data and len(data.get("tool_calls", [])) > 0:
                # This is an AgentStep
                response = AgentStep(**data)
                logger.debug("Parsed as AgentStep")
            else:
                # If no recognizable structure, treat as final answer
                logger.debug("No recognizable structure, treating as AgentFinalResponse")
                response = AgentFinalResponse(final_answer=response_text)

            return response

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON response, treating as final answer: {e}"
            )
            logger.debug(f"Raw response text: {response_text[:200]}...")
            # Fallback: treat entire response as final answer
            return AgentFinalResponse(final_answer=response_text)

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Raw response text: {response_text[:200]}...")
            # Last resort fallback
            return AgentFinalResponse(final_answer=f"Error parsing response: {str(e)}")

    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """
        Extract JSON content from a Markdown code block if present.
        
        Searches the input for a fenced code block that begins with ```json or ``` and returns the inner content trimmed of surrounding whitespace and fences. If no such code block is found, returns the original input trimmed.
        
        Parameters:
            text (str): Text that may contain a Markdown fenced code block with JSON.
        
        Returns:
            str: The JSON text extracted from the code block, or the original trimmed text if no code block is found.
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
        Check whether a parsed agent response object satisfies the required structure for its specific response type.
        
        Validation rules:
        - AgentPlan: must have a non-empty `plan`.
        - AgentStep: must have `tool_calls`; each tool call must include `id` and `tool_name`.
        - AgentFinalResponse: must have a non-empty `final_answer`.
        
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
        elif isinstance(response, AgentFinalResponse):
            if not response.final_answer:
                logger.warning("Invalid AgentFinalResponse: must have final_answer")
                return False

        return True

    @staticmethod
    def extract_thought(
        response: Union[AgentPlan, AgentStep, AgentFinalResponse],
    ) -> Optional[str]:
        """
        Retrieve the thought text from a response object.
        
        For AgentPlan, AgentStep, and AgentFinalResponse, returns the object's `thought` attribute if present.
        
        Parameters:
            response (Union[AgentPlan, AgentStep, AgentFinalResponse]): The response to extract thought from.
        
        Returns:
            Optional[str]: The thought text if available, `None` otherwise.
        """
        return getattr(response, "thought", None)