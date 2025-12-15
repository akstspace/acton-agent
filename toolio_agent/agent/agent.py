"""
Main Agent implementation for the AI Agent Framework.

This module contains the core Agent class that orchestrates LLM interactions,
tool execution, and conversation management.
"""

from typing import Generator, List, Optional, Union

from loguru import logger

from .client import LLMClient
from .exceptions import LLMCallError, MaxIterationsError, ToolExecutionError
from .models import (
    AgentFinalResponse,
    AgentPlan,
    AgentResponse,
    AgentStep,
    Message,
    ToolCall,
    ToolResult,
)
from .parser import ResponseParser
from .prompts import build_system_prompt
from .retry import RetryConfig
from .tools import Tool, ToolRegistry


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
        stream: bool = False,
    ):
        """
        Create an Agent configured to orchestrate LLM calls, tool execution, and conversation state.
        
        Parameters:
            llm_client (LLMClient): The LLM client used for generating responses.
            system_prompt (Optional[str]): Optional custom instructions that are embedded into the agent's system prompt.
            max_iterations (int): Maximum number of reasoning iterations the agent will perform before raising an error.
            retry_config (Optional[RetryConfig]): Retry behavior for LLM and tool calls; a default RetryConfig is created when omitted.
            stream (bool): If true, enables streaming responses from the LLM (tokens yielded as they arrive).
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
        Register a Tool with the agent, making it available for use in tool calls.
        
        Parameters:
            tool (Tool): The tool instance to register with the agent's tool registry.
        """
        self.tool_registry.register(tool)

    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a registered tool from the agent's tool registry.
        
        Parameters:
            tool_name (str): Name of the tool to remove.
        
        Raises:
            ToolNotFoundError: If no tool with the given name is registered.
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
        Compose the sequence of messages to send to the LLM.
        
        Creates a system message that contains the agent's system prompt followed by the tool registry formatted for inclusion in prompts, then appends the current conversation history in order.
        
        Returns:
            messages (List[Message]): Ordered list of Message objects starting with the system message and followed by the conversation history.
        """
        messages = [
            Message(
                role="system",
                content=f"{self.system_prompt}\n\n{self.tool_registry.format_for_prompt()}",
            )
        ]
        messages.extend(self.conversation_history)
        return messages

    def _execute_single_tool(self, tool: Tool, parameters: dict) -> str:
        """
        Execute the given tool with configured retry behavior and return its result.
        
        Parameters:
            tool (Tool): The tool to invoke.
            parameters (dict): Parameters to pass to the tool's `execute` method.
        
        Returns:
            result_text (str): The tool's execution result.
        
        Raises:
            ToolExecutionError: If the tool fails after the configured retry attempts; wraps the original exception.
        """

        def _execute():
            """
            Invoke the current tool with the provided parameters and return its execution result.
            
            Returns:
                The value returned by the tool's `execute` method.
            """
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
        Execute a sequence of tool calls and produce corresponding ToolResult entries.
        
        For each ToolCall in the input list, the agent attempts to locate and invoke the named tool, producing a ToolResult that records the tool_call id, tool name, returned text, and any error.
        
        Parameters:
            tool_calls (List[ToolCall]): Tool calls to execute in order.
        
        Returns:
            List[ToolResult]: A list of ToolResult objects in the same order as the input ToolCall list. If a tool is not registered the corresponding ToolResult contains an error message "Tool '<name>' not found". If a tool's execution output begins with "Error", that text is recorded in the `error` field and the `result` is set to an empty string. If execution raises a ToolExecutionError, the exception string is recorded in the `error` field and the `result` is an empty string.
        """
        results = []

        for tool_call in tool_calls:
            tool = self.tool_registry.get(tool_call.tool_name)

            if tool is None:
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    result="",
                    error=f"Tool '{tool_call.tool_name}' not found",
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
                        error=error,
                    )

                    if result.success:
                        logger.success(
                            f"Tool {tool_call.tool_name} executed successfully"
                        )
                    else:
                        logger.warning(
                            f"Tool {tool_call.tool_name} returned error: {error}"
                        )

                except ToolExecutionError as e:
                    logger.error(f"Tool {tool_call.tool_name} execution failed: {e}")
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        result="",
                        error=str(e),
                    )

            results.append(result)

        return results

    def _format_tool_results(self, results: List[ToolResult]) -> str:
        """
        Create a human-readable string representation of multiple tool execution results for appending to conversation history.
        
        Parameters:
            results (List[ToolResult]): Tool execution outcomes to format; each entry's `tool_name`, `tool_call_id`, and either `result` (on success) or `error` (on failure) are included.
        
        Returns:
            str: Multi-line string that lists each tool's name and call ID, followed by either "Success: <result>" or "Error: <error>" for that call.
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
        Invoke the configured LLM client with retry handling.
        
        Parameters:
            messages (List[Message]): The message sequence to send to the LLM.
        
        Returns:
            str: The LLM's full response text.
        
        Raises:
            LLMCallError: If the LLM call fails after the configured number of retry attempts.
        """

        def _call():
            """
            Invoke the configured LLM client with the assembled messages and return its response.
            
            Returns:
                The response returned by the LLM client.
            """
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

    def _call_llm_with_retry_stream(
        self, messages: List[Message]
    ) -> Generator[str, None, str]:
        """
        Stream tokens from the LLM for the provided message sequence.
        
        Yields token chunks as they arrive from the LLM. The generator's final return value (accessible via StopIteration.value) is the complete accumulated response string.
        
        Returns:
            The complete accumulated response string produced by the LLM.
        
        Raises:
            AttributeError: If the LLM client does not implement `call_stream()`.
            LLMCallError: If the streaming call fails after retry handling.
        """

        def _call_stream():
            """
            Stream tokens from the LLM client and yield each received chunk.
            
            Yields:
                str: Each chunk of text produced by the LLM as it becomes available.
            
            Returns:
                final_text (str): The full concatenated text of all yielded chunks when the stream completes.
            
            Raises:
                AttributeError: If the configured LLM client does not implement a `call_stream` method.
            """
            logger.debug("Calling LLM (streaming)...")
            # Check if client has call_stream method
            if not hasattr(self.llm_client, "call_stream"):
                raise AttributeError(
                    "LLM client does not support streaming. Use stream=False or use a client with call_stream() method."
                )

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

    def run_stream(
        self, user_input: str
    ) -> Generator[Union[AgentPlan, AgentStep, AgentFinalResponse, dict], None, None]:
        """
        Stream the agent's execution for a single user input, yielding plans, steps, intermediate tool results, streaming tokens, and the final response.
        
        Parameters:
            user_input (str): The user's question or request to process.
        
        Yields:
            AgentPlan: A multi-step plan the agent intends to follow.
            AgentStep: A single step containing one or more tool calls to execute.
            AgentFinalResponse: The final answer produced by the agent.
            dict: Event or progress objects, including:
                - {"type": "stream_start" | "stream_end"} for streaming boundaries,
                - {"type": "token", "content": <str>} for streaming tokens,
                - {"type": "tool_results", "results": <List[ToolResult]>} for executed tool outputs,
                - error dictionaries when LLM or tool execution fails.
        
        Raises:
            MaxIterationsError: If the agent exhausts max_iterations without producing a final response.
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
                yield {"type": "tool_results", "results": tool_results}

                continue

            elif isinstance(agent_response, AgentFinalResponse):
                logger.success("Agent produced final answer")
                yield agent_response
                return

            # Legacy AgentResponse handling
            elif isinstance(agent_response, AgentResponse):
                # Handle tool calls
                if agent_response.has_tool_calls:
                    logger.info(
                        f"Executing {len(agent_response.tool_calls)} tool call(s)"
                    )

                    # Convert to AgentStep for consistency
                    step = AgentStep(
                        thought=agent_response.thought.content
                        if agent_response.thought
                        and hasattr(agent_response.thought, "content")
                        else None,
                        tool_calls=agent_response.tool_calls,
                    )
                    yield step

                    tool_results = self._execute_tool_calls(agent_response.tool_calls)
                    results_text = self._format_tool_results(tool_results)

                    self.conversation_history.append(
                        Message(role="user", content=results_text)
                    )

                    # Yield tool results info
                    yield {"type": "tool_results", "results": tool_results}

                    continue

                # Return final answer
                if agent_response.is_final:
                    logger.success("Agent produced final answer")
                    # Convert to AgentFinalResponse
                    thought = None
                    if agent_response.thought:
                        thought = (
                            agent_response.thought.content
                            if hasattr(agent_response.thought, "content")
                            else str(agent_response.thought)
                        )
                    final_response = AgentFinalResponse(
                        thought=thought, final_answer=agent_response.final_answer
                    )
                    yield final_response
                    return

        logger.warning("Agent reached maximum iterations without final answer")
        raise MaxIterationsError(max_iterations=self.max_iterations)

    def run(self, user_input: str) -> str:
        """
        Run the agent on user input and produce the conversation's final answer.
        
        Parameters:
            user_input (str): The user's prompt or request.
        
        Returns:
            str: The agent's final answer.
        
        Raises:
            MaxIterationsError: If no final answer is produced within the configured max_iterations.
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
        Clear the agent's conversation history.
        """
        self.conversation_history = []
        logger.info("Agent conversation history reset")

    def get_conversation_history(self) -> List[Message]:
        """
        Return a copy of the agent's conversation history.
        
        Returns:
            List[Message]: A copy of the conversation history as a list of Message objects.
        """
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the agent's custom instruction portion and rebuild the full system prompt.
        
        Parameters:
            prompt (str): New custom instructions to embed into the agent's system prompt.
        """
        self.custom_instructions = prompt
        self.system_prompt = build_system_prompt(prompt)
        logger.info("System prompt updated")

    def __repr__(self) -> str:
        """
        Return a compact string summarizing the agent's tool count, conversation history length, and max iterations.
        
        Returns:
            str: A representation like "Agent(tools=<n>, history=<m>, max_iterations=<k>)" where `<n>` is the number of registered tools, `<m>` is the number of messages in conversation history, and `<k>` is the configured maximum iterations.
        """
        return (
            f"Agent(tools={len(self.tool_registry)}, "
            f"history={len(self.conversation_history)}, "
            f"max_iterations={self.max_iterations})"
        )