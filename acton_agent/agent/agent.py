"""
Main Agent implementation for the AI Agent Framework.

This module contains the core Agent class that orchestrates LLM interactions,
tool execution, and conversation management.
"""

from datetime import datetime
from typing import Generator, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from .client import LLMClient
from .exceptions import LLMCallError, MaxIterationsError, ToolExecutionError
from .models import (
    AgentFinalResponse,
    AgentFinalResponseEvent,
    AgentPlan,
    AgentPlanEvent,
    AgentStep,
    AgentStepEvent,
    AgentStreamEnd,
    AgentStreamStart,
    AgentToken,
    AgentToolResultsEvent,
    Message,
    StreamingEvent,
    ToolCall,
    ToolResult,
)
from .parser import ResponseParser
from .prompts import build_system_prompt, get_default_format_instructions
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
        final_answer_format_instructions: Optional[str] = None,
        timezone: str = "UTC",
    ):
        """
        Initialize an Agent for orchestrating LLM interactions, tool execution, retries, and conversation state.

        Parameters:
            llm_client (LLMClient): LLM client used for all model calls.
            system_prompt (Optional[str]): Custom system instructions to include in the agent's system prompt.
            max_iterations (int): Maximum reasoning iterations before the agent raises a MaxIterationsError.
            retry_config (Optional[RetryConfig]): Retry configuration for LLM and tool calls; a default is created when omitted.
            stream (bool): Enable streaming LLM responses (tokens yielded as they arrive) when True.
            final_answer_format_instructions (Optional[str]): Instructions controlling final-answer formatting; defaults to the module's standard format when omitted.
            timezone (str): Timezone name used when inserting the current date/time into system messages (e.g., "UTC", "America/New_York"); defaults to "UTC".
        """
        self.llm_client = llm_client
        self.custom_instructions = system_prompt  # Store custom instructions separately
        self.final_answer_format_instructions = (
            final_answer_format_instructions or get_default_format_instructions()
        )
        self.timezone = timezone
        self.system_prompt = build_system_prompt(
            system_prompt, self.final_answer_format_instructions
        )
        self.max_iterations = max_iterations
        self.retry_config = retry_config or RetryConfig()
        self.stream = stream

        self.tool_registry = ToolRegistry()
        self.conversation_history: List[Message] = []
        self.response_parser = ResponseParser()

        logger.success("Agent initialized successfully")

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the agent's tool registry so it can be invoked in future tool calls.

        Parameters:
            tool (Tool): The tool instance to add to the agent's registry.
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
        Build the ordered message list to send to the LLM.

        The first message is a system message containing the agent's system prompt, the current date and time in the agent's configured timezone (falls back to UTC on error), and the tool registry formatted for inclusion in prompts. The remaining messages are the current conversation history in chronological order.

        Returns:
            messages (List[Message]): Ordered list of Message objects starting with the system message followed by the conversation history.
        """
        # Get current date and time in the specified timezone
        try:
            tz = ZoneInfo(self.timezone)
            current_datetime = datetime.now(tz)
            datetime_str = current_datetime.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except Exception as e:
            logger.warning(
                f"Failed to get timezone '{self.timezone}': {e}. Falling back to UTC."
            )
            current_datetime = datetime.now(ZoneInfo("UTC"))
            datetime_str = current_datetime.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")

        messages = [
            Message(
                role="system",
                content=f"{self.system_prompt}\n\nCurrent Date and Time: {datetime_str}\n\n{self.tool_registry.format_for_prompt()}",
            )
        ]
        messages.extend(self.conversation_history)
        return messages

    def _execute_single_tool(self, tool: Tool, parameters: dict) -> str:
        """
        Execute a Tool using the agent's configured retry policy and return the tool's result.

        Parameters:
            tool (Tool): Tool to invoke.
            parameters (dict): Arguments to pass to the tool's `execute` method.

        Returns:
            result_text (str): The text returned by the tool's execution.

        Raises:
            ToolExecutionError: If the tool fails after the configured retry attempts; wraps the original exception.
        """

        def _execute():
            """
            Invoke the current tool with the provided parameters and return its execution result.

            Returns:
                The tool's execution result string.
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
        Format multiple tool execution results into a readable multi-line string suitable for appending to conversation history.

        Each entry includes the tool name, call ID, and either "Success: <result>" or "Error: <error>".

        Parameters:
            results (List[ToolResult]): Sequence of tool results to format.

        Returns:
            str: Multi-line string summarizing each tool call and its outcome.
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
        Stream token chunks from the configured LLM for the given message sequence.

        Yields each text chunk as it becomes available; when the generator completes, its return value (accessible as StopIteration.value) is the full concatenated response.

        Returns:
            final_text (str): The complete response text produced by the LLM.

        Raises:
            AttributeError: If the configured LLM client does not implement `call_stream`.
            LLMCallError: If the streaming call fails.
        """

        def _call_stream():
            """
            Yield token chunks produced by the LLM client's streaming interface and return the concatenated final text.

            Yields:
                str: Each chunk of text produced by the LLM as it becomes available.

            Returns:
                final_text (str): Concatenation of all yielded chunks when the stream completes.

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

    def run_stream(self, user_input: str) -> Generator[StreamingEvent, None, None]:
        """
        Stream the agent's execution for a single user input, yielding structured streaming events.

        Parameters:
            user_input (str): The user's question or request to process.

        Yields:
            StreamingEvent: One of the structured streaming event models:
                - AgentStreamStart: Emitted when LLM streaming starts.
                - AgentToken: Individual tokens from the LLM stream.
                - AgentStreamEnd: Emitted when LLM streaming ends.
                - AgentToolResultsEvent: Tool execution results.
                - AgentPlanEvent: A complete agent plan.
                - AgentStepEvent: A complete agent step with tool calls.
                - AgentFinalResponseEvent: The final answer from the agent.

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
                    yield AgentStreamStart()
                    for chunk in self._call_llm_with_retry_stream(messages):
                        llm_response_text += chunk
                        yield AgentToken(content=chunk)
                    yield AgentStreamEnd()
                else:
                    # Non-streaming mode
                    llm_response_text = self._call_llm_with_retry(messages)
            except LLMCallError as e:
                logger.error(f"LLM call failed: {e}")
                error_response = AgentFinalResponse(
                    final_answer=f"Error: Failed to get response from LLM - {str(e.original_error)}"
                )
                yield AgentFinalResponseEvent(response=error_response)
                return

            # Parse response (could be AgentPlan, AgentStep, or AgentFinalResponse)
            agent_response = self.response_parser.parse(llm_response_text)

            # Add to history
            self.conversation_history.append(
                Message(role="assistant", content=llm_response_text)
            )

            # Handle different response types
            if isinstance(agent_response, AgentPlan):
                logger.info(f"Agent created plan with {len(agent_response.plan)} steps")
                yield AgentPlanEvent(plan=agent_response)
                # Continue to next iteration - agent should follow up with AgentStep or AgentFinalResponse
                continue

            elif isinstance(agent_response, AgentStep):
                logger.info(f"Executing {len(agent_response.tool_calls)} tool call(s)")
                yield AgentStepEvent(step=agent_response)

                # Execute tools and add results to conversation
                tool_results = self._execute_tool_calls(agent_response.tool_calls)
                results_text = self._format_tool_results(tool_results)

                self.conversation_history.append(
                    Message(role="user", content=results_text)
                )

                # Yield tool results info
                yield AgentToolResultsEvent(results=tool_results)

                continue

            elif isinstance(agent_response, AgentFinalResponse):
                logger.success("Agent produced final answer")
                yield AgentFinalResponseEvent(response=agent_response)
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
        for event in self.run_stream(user_input):
            # Skip intermediate steps and stream events, only capture final response
            if isinstance(event, AgentFinalResponseEvent):
                final_answer = event.response.final_answer
                break

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
        Retrieve a shallow copy of the agent's conversation history.

        Returns:
            List[Message]: A shallow copy of the conversation history as a list of Message objects in chronological order.
        """
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the agent's custom instructions and rebuild the system prompt using the current final-answer format instructions.
        """
        self.custom_instructions = prompt
        self.system_prompt = build_system_prompt(
            prompt, self.final_answer_format_instructions
        )
        logger.info("System prompt updated")

    def set_final_answer_format(self, format_instructions: str) -> None:
        """
        Update the formatting instructions for final answers and rebuild the system prompt.

        Parameters:
            format_instructions (str): New formatting instructions for final answers.
        """
        self.final_answer_format_instructions = format_instructions
        self.system_prompt = build_system_prompt(
            self.custom_instructions, format_instructions
        )
        logger.info("Final answer format instructions updated")

    def set_timezone(self, timezone: str) -> None:
        """
        Update the agent's timezone used when rendering current date/time in system messages.

        Parameters:
            timezone (str): IANA timezone name (e.g., "UTC", "America/New_York", "Europe/London").

        Raises:
            ValueError: If the provided timezone name is not valid.
        """
        # Validate timezone
        try:
            ZoneInfo(timezone)
            self.timezone = timezone
            logger.info(f"Timezone updated to {timezone}")
        except Exception as e:
            logger.error(f"Invalid timezone '{timezone}': {e}")
            raise ValueError(f"Invalid timezone: {timezone}")

    def __repr__(self) -> str:
        """
        Compactly summarize the agent's registered tools, conversation history length, and configured maximum iterations.

        Returns:
            str: Representation in the form `Agent(tools=<n>, history=<m>, max_iterations=<k>)` where `<n>` is the number of registered tools, `<m>` is the number of messages in the conversation history, and `<k>` is the configured maximum iterations.
        """
        return (
            f"Agent(tools={len(self.tool_registry)}, "
            f"history={len(self.conversation_history)}, "
            f"max_iterations={self.max_iterations})"
        )
