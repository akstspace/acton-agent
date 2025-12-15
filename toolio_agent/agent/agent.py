"""
Main Agent implementation for the AI Agent Framework.

This module contains the core Agent class that orchestrates LLM interactions,
tool execution, and conversation management.
"""

from typing import List, Optional, Generator, Union
from loguru import logger
import json

from .models import Message, ToolCall, ToolResult, AgentResponse, AgentPlan, AgentStep, AgentFinalResponse
from .client import LLMClient
from .tools import Tool, ToolRegistry
from .parser import ResponseParser
from .retry import RetryConfig
from .prompts import build_system_prompt
from .exceptions import (
    ToolNotFoundError,
    ToolExecutionError,
    LLMCallError,
    MaxIterationsError
)


class Agent:
    """
    Production-ready LLM agent with tool execution capabilities.
    
    Features:
    - Extensible tool system
    - Automatic retries with tenacity
    - Structured conversation history
    - Comprehensive error handling
    - Loguru logging throughout
    
    Example:
        ```python
        agent = Agent(
            llm_client=my_llm_client,
            system_prompt="You are a helpful assistant",
            max_iterations=10
        )
        
        agent.register_tool(my_tool)
        result = agent.run("What is 2+2?")
        ```
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        retry_config: Optional[RetryConfig] = None,
        stream: bool = False
    ):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLM client implementing the LLMClient protocol
            system_prompt: Optional custom instructions to prepend to the auto-generated prompt
            max_iterations: Maximum number of reasoning iterations
            retry_config: Configuration for retry logic
            stream: Enable streaming responses (yields tokens as they arrive)
        """
        self.llm_client = llm_client
        self.custom_instructions = system_prompt  # Store custom instructions separately
        self.system_prompt = build_system_prompt(system_prompt)
        self.max_iterations = max_iterations
        self.retry_config = retry_config or RetryConfig()
        self.stream = stream
        
        self.tool_registry = ToolRegistry()
        self.conversation_history: List[Message] = []
        self.response_parser = ResponseParser()
        
        logger.success("Agent initialized successfully")
    

    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: Tool instance to register
        """
        self.tool_registry.register(tool)
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from the agent.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        self.tool_registry.unregister(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return self.tool_registry.list_tool_names()
    
    def _build_messages(self) -> List[Message]:
        """
        Build the full message list for LLM.
        
        Returns:
            List of messages including system prompt and conversation history
        """
        messages = [
            Message(
                role="system",
                content=f"{self.system_prompt}\n\n{self.tool_registry.format_for_prompt()}"
            )
        ]
        messages.extend(self.conversation_history)
        return messages
    
    def _execute_single_tool(self, tool: Tool, parameters: dict) -> str:
        """
        Execute a single tool with retry logic.
        
        Args:
            tool: Tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolExecutionError: If tool execution fails after retries
        """
        def _execute():
            logger.debug(f"Executing tool: {tool.name} with parameters: {parameters}")
            result = tool.execute(parameters)
            logger.debug(f"Tool {tool.name} execution completed")
            return result
        
        try:
            # Wrap with retry logic
            wrapped_func = self.retry_config.wrap_function(_execute)
            return wrapped_func()
        except Exception as e:
            logger.error(
                f"Tool {tool.name} failed after {self.retry_config.max_attempts} attempts: {e}"
            )
            raise ToolExecutionError(tool.name, e)
    
    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tool calls and return results.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tool results
        """
        results = []
        
        for tool_call in tool_calls:
            tool = self.tool_registry.get(tool_call.tool_name)
            
            if tool is None:
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    result="",
                    error=f"Tool '{tool_call.tool_name}' not found"
                )
                logger.error(f"Tool not found: {tool_call.tool_name}")
            else:
                try:
                    # Execute with retry
                    result_text = self._execute_single_tool(tool, tool_call.parameters)
                    
                    # Check if result indicates an error
                    error = None
                    if result_text.startswith("Error"):
                        error = result_text
                        result_text = ""
                    
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        result=result_text,
                        error=error
                    )
                    
                    if result.success:
                        logger.success(f"Tool {tool_call.tool_name} executed successfully")
                    else:
                        logger.warning(f"Tool {tool_call.tool_name} returned error: {error}")
                        
                except ToolExecutionError as e:
                    logger.error(f"Tool {tool_call.tool_name} execution failed: {e}")
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        result="",
                        error=str(e)
                    )
            
            results.append(result)
        
        return results
    
    def _format_tool_results(self, results: List[ToolResult]) -> str:
        """
        Format tool results for conversation history.
        
        Args:
            results: List of tool results
            
        Returns:
            Formatted string representation of results
        """
        results_text = "Tool Results:\n"
        for result in results:
            results_text += f"\n[{result.tool_name}] (ID: {result.tool_call_id})\n"
            if result.success:
                results_text += f"Success: {result.result}\n"
            else:
                results_text += f"Error: {result.error}\n"
        return results_text
    
    def _call_llm_with_retry(self, messages: List[Message]) -> str:
        """
        Call LLM with retry logic.
        
        Args:
            messages: Messages to send to LLM
            
        Returns:
            LLM response text
            
        Raises:
            LLMCallError: If LLM call fails after retries
        """
        def _call():
            logger.debug("Calling LLM...")
            result = self.llm_client.call(messages)
            logger.debug("LLM call completed")
            return result
        
        try:
            # Wrap with retry logic
            wrapped_func = self.retry_config.wrap_function(_call)
            return wrapped_func()
        except Exception as e:
            logger.error(
                f"LLM call failed after {self.retry_config.max_attempts} attempts: {e}"
            )
            raise LLMCallError(e, self.retry_config.max_attempts)
    
    def _call_llm_with_retry_stream(self, messages: List[Message]) -> Generator[str, None, str]:
        """
        Call LLM with retry logic and streaming support.
        
        Args:
            messages: Messages to send to LLM
            
        Yields:
            Token chunks as they arrive
            
        Returns:
            Complete accumulated response
            
        Raises:
            LLMCallError: If LLM call fails after retries
        """
        def _call_stream():
            logger.debug("Calling LLM (streaming)...")
            # Check if client has call_stream method
            if not hasattr(self.llm_client, 'call_stream'):
                raise AttributeError("LLM client does not support streaming. Use stream=False or use a client with call_stream() method.")
            
            accumulated = ""
            for chunk in self.llm_client.call_stream(messages):
                accumulated += chunk
                yield chunk
            logger.debug("LLM streaming call completed")
            return accumulated
        
        try:
            # For streaming, we don't wrap with retry since it's a generator
            # If we need retry for streaming, it needs more complex logic
            return _call_stream()
        except Exception as e:
            logger.error(f"LLM streaming call failed: {e}")
            raise LLMCallError(e, self.retry_config.max_attempts)
    
    def run_stream(self, user_input: str) -> Generator[Union[AgentPlan, AgentStep, AgentFinalResponse, dict], None, None]:
        """
        Run the agent on a user input as a generator, yielding each step.
        
        This method allows streaming of agent execution steps. The agent will:
        1. Add the user input to conversation history
        2. Call the LLM to get a response
        3. Yield AgentPlan, AgentStep, or AgentFinalResponse at each iteration
        4. Execute any requested tools (for AgentStep)
        5. Repeat until a final answer is produced or max iterations reached
        
        Args:
            user_input: The user's question or request
            
        Yields:
            AgentPlan, AgentStep, or AgentFinalResponse objects, or error dict
            
        Raises:
            MaxIterationsError: If max iterations reached without final answer
            
        Example:
            ```python
            for step in agent.run_stream("What is the weather?"):
                if isinstance(step, AgentPlan):
                    print(f"Plan: {step.plan}")
                elif isinstance(step, AgentStep):
                    print(f"Executing {len(step.tool_calls)} tools")
                elif isinstance(step, AgentFinalResponse):
                    print(f"Final: {step.final_answer}")
            ```
        """
        logger.info(f"Agent starting run with input: {user_input[:100]}...")
        self.conversation_history.append(Message(role="user", content=user_input))
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
            
            # Build messages
            messages = self._build_messages()
            
            # Get LLM response with retry
            try:
                if self.stream:
                    # Streaming mode - yield tokens and accumulate response
                    llm_response_text = ""
                    yield {"type": "stream_start"}
                    for chunk in self._call_llm_with_retry_stream(messages):
                        llm_response_text += chunk
                        yield {"type": "token", "content": chunk}
                    yield {"type": "stream_end"}
                else:
                    # Non-streaming mode
                    llm_response_text = self._call_llm_with_retry(messages)
            except LLMCallError as e:
                logger.error(f"LLM call failed: {e}")
                error_response = AgentFinalResponse(
                    final_answer=f"Error: Failed to get response from LLM - {str(e.original_error)}"
                )
                yield error_response
                return
            
            # Parse response (could be AgentPlan, AgentStep, AgentFinalResponse, or legacy AgentResponse)
            agent_response = self.response_parser.parse(llm_response_text)
            
            # Add to history
            self.conversation_history.append(
                Message(role="assistant", content=llm_response_text)
            )
            
            # Handle different response types
            if isinstance(agent_response, AgentPlan):
                logger.info(f"Agent created plan with {len(agent_response.plan)} steps")
                yield agent_response
                # Continue to next iteration - agent should follow up with AgentStep or AgentFinalResponse
                continue
            
            elif isinstance(agent_response, AgentStep):
                logger.info(f"Executing {len(agent_response.tool_calls)} tool call(s)")
                yield agent_response
                
                # Execute tools and add results to conversation
                tool_results = self._execute_tool_calls(agent_response.tool_calls)
                results_text = self._format_tool_results(tool_results)
                
                self.conversation_history.append(
                    Message(role="user", content=results_text)
                )
                
                # Yield tool results info
                yield {
                    "type": "tool_results",
                    "results": tool_results
                }
                
                continue
            
            elif isinstance(agent_response, AgentFinalResponse):
                logger.success("Agent produced final answer")
                yield agent_response
                return
            
            # Legacy AgentResponse handling
            elif isinstance(agent_response, AgentResponse):
                # Handle tool calls
                if agent_response.has_tool_calls:
                    logger.info(f"Executing {len(agent_response.tool_calls)} tool call(s)")
                    
                    # Convert to AgentStep for consistency
                    step = AgentStep(
                        thought=agent_response.thought.content if agent_response.thought and hasattr(agent_response.thought, 'content') else None,
                        tool_calls=agent_response.tool_calls
                    )
                    yield step
                    
                    tool_results = self._execute_tool_calls(agent_response.tool_calls)
                    results_text = self._format_tool_results(tool_results)
                    
                    self.conversation_history.append(
                        Message(role="user", content=results_text)
                    )
                    
                    # Yield tool results info
                    yield {
                        "type": "tool_results",
                        "results": tool_results
                    }
                    
                    continue
                
                # Return final answer
                if agent_response.is_final:
                    logger.success("Agent produced final answer")
                    # Convert to AgentFinalResponse
                    thought = None
                    if agent_response.thought:
                        thought = agent_response.thought.content if hasattr(agent_response.thought, 'content') else str(agent_response.thought)
                    final_response = AgentFinalResponse(
                        thought=thought,
                        final_answer=agent_response.final_answer
                    )
                    yield final_response
                    return
        
        logger.warning("Agent reached maximum iterations without final answer")
        raise MaxIterationsError(max_iterations=self.max_iterations)
    
    def run(self, user_input: str) -> str:
        """
        Run the agent on a user input and return the final answer.
        
        This is the main entry point for agent execution. The agent will:
        1. Add the user input to conversation history
        2. Call the LLM to get a response
        3. Execute any requested tools
        4. Repeat until a final answer is produced or max iterations reached
        
        Args:
            user_input: The user's question or request
            
        Returns:
            The final answer string from the agent
            
        Raises:
            MaxIterationsError: If max iterations reached without final answer
            
        Example:
            ```python
            result = agent.run("What is the weather?")
            print(result)  # "The weather is sunny."
            ```
        """
        final_answer = None
        for step in self.run_stream(user_input):
            # Skip intermediate steps and stream events
            if isinstance(step, AgentFinalResponse):
                final_answer = step.final_answer
                break
            elif isinstance(step, dict) and step.get("type") == "stream_end":
                # For streaming mode, wait for final response after stream ends
                continue
        
        # If we got here without a final answer, something went wrong
        if final_answer is None:
            raise MaxIterationsError(max_iterations=self.max_iterations)
        
        return final_answer
    
    def reset(self) -> None:
        """
        Clear conversation history.
        
        Use this to start a fresh conversation with the agent.
        """
        self.conversation_history = []
        logger.info("Agent conversation history reset")
    
    def get_conversation_history(self) -> List[Message]:
        """
        Get the current conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self.conversation_history.copy()
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the custom instructions portion of the system prompt.
        The schemas will be automatically re-generated and appended.
        
        Args:
            prompt: New custom instructions
        """
        self.custom_instructions = prompt
        self.system_prompt = build_system_prompt(prompt)
        logger.info("System prompt updated")
    
    def __repr__(self) -> str:
        return (
            f"Agent(tools={len(self.tool_registry)}, "
            f"history={len(self.conversation_history)}, "
            f"max_iterations={self.max_iterations})"
        )
