"""
Response parser for the AI Agent Framework.

This module provides functionality to parse and validate LLM responses
into structured AgentResponse objects.
"""

import json
import re
from typing import Optional, Union
from loguru import logger

from .models import AgentResponse, AgentPlan, AgentStep, AgentFinalResponse
from .exceptions import ResponseParseError


class ResponseParser:
    """
    Parse and validate LLM responses into structured response objects.
    
    Handles JSON parsing, markdown code block removal, and supports multiple
    response types: AgentPlan, AgentStep, AgentFinalResponse, or legacy AgentResponse.
    """
    
    @staticmethod
    def parse(response_text: str) -> Union[AgentPlan, AgentStep, AgentFinalResponse, AgentResponse]:
        """
        Parse LLM response into appropriate response model.
        
        The parser detects the type of response based on the fields present:
        - AgentPlan: has 'plan' field
        - AgentStep: has 'tool_calls' field (without 'plan' or 'final_answer')
        - AgentFinalResponse: has 'final_answer' field (without 'plan' or 'tool_calls')
        - AgentResponse: legacy format fallback
        
        Process:
        1. Extract JSON from markdown code block (```json ... ```)
        2. Parse the extracted JSON
        3. Detect response type and create appropriate model
        4. Fallback to treating as final answer if parsing fails
        
        Args:
            response_text: Raw text response from the LLM
            
        Returns:
            Parsed response object (AgentPlan, AgentStep, AgentFinalResponse, or AgentResponse)
            
        Raises:
            ResponseParseError: If parsing fails critically (should be rare)
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
                # Fallback to legacy AgentResponse
                response = AgentResponse(**data)
                logger.debug("Parsed as legacy AgentResponse")
            
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
            return AgentFinalResponse(
                final_answer=f"Error parsing response: {str(e)}"
            )
    
    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """
        Extract JSON content from markdown code block.
        
        Looks for ```json ... ``` or ``` ... ``` blocks and extracts the content.
        If no code block is found, returns the original text.
        
        Args:
            text: Text potentially containing markdown code blocks
            
        Returns:
            Extracted JSON text (without code block markers)
        """
        text = text.strip()
        
        # Pattern to match ```json or ``` at start, content, and ``` at end
        # Match ```json or just ```
        pattern = r'^```(?:json)?\s*\n(.*?)\n```\s*$'
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        
        if match:
            # Extract the content between the code block markers
            json_content = match.group(1).strip()
            logger.debug("Extracted JSON from markdown code block")
            return json_content
        
        # Try without newlines (in case of single line code block)
        pattern = r'^```(?:json)?\s*(.*?)```\s*$'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            json_content = match.group(1).strip()
            logger.debug("Extracted JSON from inline markdown code block")
            return json_content
        
        # No code block found, return original text
        logger.debug("No markdown code block found, using raw text")
        return text
    
    @staticmethod
    def validate_response(response: Union[AgentPlan, AgentStep, AgentFinalResponse, AgentResponse]) -> bool:
        """
        Validate that a response is well-formed.
        
        Args:
            response: Response object to validate
            
        Returns:
            True if response is valid, False otherwise
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
        
        # Legacy AgentResponse validation
        elif isinstance(response, AgentResponse):
            # Response must have either tool_calls or final_answer
            if not response.has_tool_calls and not response.is_final:
                logger.warning(
                    "Invalid response: must have either tool_calls or final_answer"
                )
                return False
            
            # If has tool calls, validate each one
            if response.has_tool_calls:
                for tool_call in response.tool_calls:
                    if not tool_call.id or not tool_call.tool_name:
                        logger.warning(
                            f"Invalid tool call: missing id or tool_name"
                        )
                        return False
        
        return True
    
    @staticmethod
    def extract_thought(response: Union[AgentPlan, AgentStep, AgentFinalResponse, AgentResponse]) -> Optional[str]:
        """
        Extract thought content from response.
        
        Args:
            response: Response object to extract from
            
        Returns:
            Thought content as string, or None if no thought
        """
        if isinstance(response, (AgentPlan, AgentStep, AgentFinalResponse)):
            return getattr(response, 'thought', None)
        
        # Legacy AgentResponse handling
        if isinstance(response, AgentResponse):
            if response.thought is None:
                return None
            
            if isinstance(response.thought, str):
                return response.thought
            
            return response.thought.content
        
        return None
