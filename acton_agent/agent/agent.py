"""
Main Agent implementation for the AI Agent Framework.

This module contains the core Agent class that orchestrates LLM interactions,
tool execution, and conversation management.
"""

import uuid
from collections.abc import Generator
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

from ..client import LLMClient
from ..memory import AgentMemory, SimpleAgentMemory
from ..parsers import ResponseParser
from ..tools import Tool, ToolCall, ToolRegistry, ToolResult, ToolSet
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
    AgentToolExecutionEvent,
    AgentToolResultsEvent,
    Message,
    StreamingEvent,
)
from .prompts import build_system_prompt, get_default_format_instructions
from .retry import RetryConfig


class Agent:
    """
    LLM agent with tool execution capabilities.

    Features:
    - Extensible tool system
    - Automatic retries with tenacity
    - Structured conversation history
    - Comprehensive error handling
    - Loguru logging throughout

    Note: This is an experimental framework. The API may change without notice.

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
        system_prompt: str | None = None,
        max_iterations: int = 10,
        retry_config: RetryConfig | None = None,
        stream: bool = False,
        final_answer_format_instructions: str | None = None,
        timezone: str = "UTC",
        memory: AgentMemory | None = None,
    ):
        """
        Create a new Agent configured to coordinate LLM calls, tool execution, retries, and conversation memory.
        
        Parameters:
            llm_client: LLM client used for all model calls.
            system_prompt: Optional custom system instructions injected into the agent's system prompt.
            max_iterations: Maximum number of reasoning iterations before raising a MaxIterationsError.
            retry_config: Retry policy for LLM and tool calls; a default RetryConfig is created when omitted.
            stream: If True, enable streaming LLM responses (tokens delivered as they arrive).
            final_answer_format_instructions: Optional instructions that control final-answer formatting; defaults to the module's standard format when omitted.
            timezone: Timezone name used when inserting the current date/time into system messages (e.g., "UTC", "America/New_York"); defaults to "UTC".
            memory: Optional memory manager; when None memory management is disabled. If omitted, a default SimpleAgentMemory instance is used.
        """
        self.llm_client = llm_client
        self.custom_instructions = system_prompt  # Store custom instructions separately
        self.final_answer_format_instructions = final_answer_format_instructions or get_default_format_instructions()
        self.timezone = timezone
        self.system_prompt = build_system_prompt(system_prompt, self.final_answer_format_instructions)
        self.max_iterations = max_iterations
        self.retry_config = retry_config or RetryConfig()
        self.stream = stream
        # Use SimpleAgentMemory by default if no custom memory provided
        self.memory: AgentMemory | None = memory if memory is not None else SimpleAgentMemory()

        self.tool_registry = ToolRegistry()
        self.conversation_history: list[Message] = []
        self.response_parser = ResponseParser()

        logger.success("Agent initialized successfully")

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool so the agent can invoke it in future tool calls.

        Parameters:
            tool (Tool): Tool instance to add to the agent's registry.
        """
        self.tool_registry.register(tool)

    def register_toolset(self, toolset: "ToolSet") -> None:
        """
        Register a ToolSet with the agent's ToolRegistry.

        Registers every tool from the provided ToolSet so they are available for future tool calls and adds the set's description to agent prompts for context.

        Parameters:
            toolset (ToolSet): Collection of related tools (and an optional shared description) to register.
        """

        self.tool_registry.register_toolset(toolset)

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool by name from the agent's tool registry.

        Raises:
            ToolNotFoundError: If no tool with the given name is registered.
        """
        self.tool_registry.unregister(tool_name)

    def list_tools(self) -> list[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool names
        """
        return self.tool_registry.list_tool_names()

    def _build_messages(self) -> list[Message]:
        """
        Build the ordered message list to send to the LLM.

        The first message is a system message containing the agent's system prompt, the current date and time in the agent's configured timezone (falls back to UTC on error), and the tool registry formatted for inclusion in prompts. The remaining messages are the current conversation history in chronological order.

        Automatically manages conversation history using the configured memory instance (if any).

        Returns:
            messages (List[Message]): Ordered list of Message objects starting with the system message followed by the managed conversation history.
        """
        # Apply memory management if memory is configured
        if self.memory is not None:
            managed_history = self.memory.manage_history(self.conversation_history)
        else:
            managed_history = self.conversation_history

        # Get current date and time in the specified timezone
        try:
            tz = ZoneInfo(self.timezone)
            current_datetime = datetime.now(tz)
            datetime_str = current_datetime.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except Exception as e:
            logger.warning(f"Failed to get timezone '{self.timezone}': {e}. Falling back to UTC.")
            current_datetime = datetime.now(ZoneInfo("UTC"))
            datetime_str = current_datetime.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")

        messages = [
            Message(
                role="system",
                content=f"{self.system_prompt}\n\nCurrent Date and Time: {datetime_str}\n\n{self.tool_registry.format_for_prompt()}",
            )
        ]
        messages.extend(managed_history)
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
                str: The tool's execution result string.
            """
            logger.debug(f"Executing tool: {tool.name} with parameters: {parameters}")
            toolset_params = self.tool_registry.get_toolset_params(tool.name)
            result = tool.execute(parameters, toolset_params)
            logger.debug(f"Tool {tool.name} execution completed")
            return result

        try:
            # Wrap with retry logic
            wrapped_func = self.retry_config.wrap_function(_execute)
            return wrapped_func()
        except Exception as e:
            logger.error(f"Tool {tool.name} failed after {self.retry_config.max_attempts} attempts: {e}")
            raise ToolExecutionError(tool.name, e) from e

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """
        Execute a sequence of ToolCall requests and return their ToolResult entries in order.

        Each ToolCall is resolved against the agent's registry and invoked with the call's parameters. If a tool name is not registered, the corresponding ToolResult contains an error "Tool '<name>' not found". If a tool's execution output begins with the literal text "Error", that text is recorded in the ToolResult's `error` field and the `result` is set to an empty string. If execution raises a ToolExecutionError, the exception message is recorded in the `error` field and the `result` is an empty string.

        Parameters:
            tool_calls (List[ToolCall]): Ordered tool calls to execute.

        Returns:
            List[ToolResult]: ToolResult objects corresponding to each input ToolCall, in the same order.
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
                        logger.success(f"Tool {tool_call.tool_name} executed successfully")
                    else:
                        logger.warning(f"Tool {tool_call.tool_name} returned error: {error}")

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

    def _execute_tool_calls_stream(
        self, tool_calls: list[ToolCall], step_id: str
    ) -> Generator[AgentToolExecutionEvent, None, list[ToolResult]]:
        """
        Stream execution of a sequence of tool calls and emit progress events for each call.

        Executes each ToolCall in order, yielding AgentToolExecutionEvent items with status "started", "completed", or "failed" for that step. Each emitted event includes the provided step_id, the tool call id, the tool name, and—when available—the resulting ToolResult. Execution continues through all provided calls and the final return value is the ordered list of ToolResult objects corresponding to the input calls.

        Parameters:
            tool_calls (List[ToolCall]): Ordered tool call requests to execute.
            step_id (str): Identifier included on each emitted AgentToolExecutionEvent to correlate events with a higher-level agent step.

        Yields:
            AgentToolExecutionEvent: Progress events for each tool call indicating start, completion, or failure. Completed/failed events include the associated ToolResult when available.

        Returns:
            List[ToolResult]: List of ToolResult objects in the same order as `tool_calls`, containing results or error details for each call.
        """
        results = []

        for tool_call in tool_calls:
            # Emit started event
            yield AgentToolExecutionEvent(
                step_id=step_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                status="started",
            )

            tool = self.tool_registry.get(tool_call.tool_name)

            if tool is None:
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    result="",
                    error=f"Tool '{tool_call.tool_name}' not found",
                )
                logger.error(f"Tool not found: {tool_call.tool_name}")

                # Emit failed event
                yield AgentToolExecutionEvent(
                    step_id=step_id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    status="failed",
                    result=result,
                )
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
                        logger.success(f"Tool {tool_call.tool_name} executed successfully")
                        # Emit completed event
                        yield AgentToolExecutionEvent(
                            step_id=step_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            status="completed",
                            result=result,
                        )
                    else:
                        logger.warning(f"Tool {tool_call.tool_name} returned error: {error}")
                        # Emit failed event
                        yield AgentToolExecutionEvent(
                            step_id=step_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            status="failed",
                            result=result,
                        )

                except ToolExecutionError as e:
                    logger.error(f"Tool {tool_call.tool_name} execution failed: {e}")
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        result="",
                        error=str(e),
                    )

                    # Emit failed event
                    yield AgentToolExecutionEvent(
                        step_id=step_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        status="failed",
                        result=result,
                    )

            results.append(result)

        return results

    def _format_tool_results(self, results: list[ToolResult]) -> str:
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

    def _call_llm_with_retry(self, messages: list[Message]) -> str:
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
            logger.error(f"LLM call failed after {self.retry_config.max_attempts} attempts: {e}")
            raise LLMCallError(e, self.retry_config.max_attempts) from e

    def _call_llm_with_retry_stream(self, messages: list[Message]) -> Generator[str, None, str]:
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
            raise LLMCallError(e, self.retry_config.max_attempts) from e

    def run_stream(self, user_input: str) -> Generator[StreamingEvent, None, None]:
        """
        Stream the agent's processing of a single user input as a sequence of structured streaming events.

        Yields streaming events representing LLM activity, agent planning/steps, tool execution progress, aggregated tool results, and the final agent response:
        - AgentStreamStart: emitted when LLM streaming begins for the step.
        - AgentToken: individual token/chunk produced by the LLM stream.
        - AgentStreamEnd: emitted when LLM streaming ends for the step.
        - AgentPlanEvent: a complete agent plan describing future steps.
        - AgentStepEvent: an agent step that contains tool calls to execute.
        - AgentToolExecutionEvent: progress events for individual tool executions (started, completed, failed).
        - AgentToolResultsEvent: aggregated results from executed tools for the step.
        - AgentFinalResponseEvent: the final answer produced by the agent.

        Parameters:
            user_input (str): The user's question or request to process.

        Raises:
            MaxIterationsError: If the agent exhausts the configured maximum iterations without producing a final response.
        """
        logger.info(f"Agent starting run with input: {user_input[:100]}...")
        self.conversation_history.append(Message(role="user", content=user_input))

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")

            step_id = str(uuid.uuid4())

            # Build messages
            messages = self._build_messages()

            # Get LLM response with retry
            try:
                if self.stream:
                    # Streaming mode - yield tokens and accumulate response
                    llm_response_text = ""
                    yield AgentStreamStart(step_id=step_id)
                    for chunk in self._call_llm_with_retry_stream(messages):
                        llm_response_text += chunk
                        yield AgentToken(step_id=step_id, content=chunk)
                    yield AgentStreamEnd(step_id=step_id)
                else:
                    # Non-streaming mode
                    llm_response_text = self._call_llm_with_retry(messages)
            except LLMCallError as e:
                logger.error(f"LLM call failed: {e}")
                error_response = AgentFinalResponse(
                    final_answer=f"Error: Failed to get response from LLM - {e.original_error!s}"
                )
                yield AgentFinalResponseEvent(step_id=step_id, response=error_response)
                return

            # Parse response (could be AgentPlan, AgentStep, or AgentFinalResponse)
            agent_response = self.response_parser.parse(llm_response_text)

            # Add to history
            self.conversation_history.append(Message(role="assistant", content=llm_response_text))

            # Handle different response types
            if isinstance(agent_response, AgentPlan):
                logger.info(f"Agent created plan with {len(agent_response.plan)} steps")
                yield AgentPlanEvent(step_id=step_id, plan=agent_response)
                # Continue to next iteration - agent should follow up with AgentStep or AgentFinalResponse
                continue

            elif isinstance(agent_response, AgentStep):
                logger.info(f"Executing {len(agent_response.tool_calls)} tool call(s)")
                yield AgentStepEvent(step_id=step_id, step=agent_response)

                tool_results = []
                for event in self._execute_tool_calls_stream(agent_response.tool_calls, step_id):
                    yield event
                    if event.status in ["completed", "failed"] and event.result:
                        tool_results.append(event.result)

                results_text = self._format_tool_results(tool_results)

                self.conversation_history.append(Message(role="user", content=results_text))

                yield AgentToolResultsEvent(step_id=step_id, results=tool_results)

                continue

            elif isinstance(agent_response, AgentFinalResponse):
                logger.success("Agent produced final answer")
                yield AgentFinalResponseEvent(step_id=step_id, response=agent_response)
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

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message with the given role and content to the agent's conversation history.

        Parameters:
            role (str): The message role (e.g., 'user', 'assistant', 'system').
            content (str): The message text to append.
        """
        message = Message(role=role, content=content)
        self.conversation_history.append(message)
        logger.info(f"Added {role} message to conversation history")

    def get_conversation_history(self) -> list[Message]:
        """
        Get a shallow copy of the agent's conversation history in chronological order.

        Returns:
            List[Message]: A list of Message objects representing the conversation history from oldest to newest.
        """
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the agent's custom instructions and rebuild the system prompt using the current final-answer format instructions.
        """
        self.custom_instructions = prompt
        self.system_prompt = build_system_prompt(prompt, self.final_answer_format_instructions)
        logger.info("System prompt updated")

    def set_final_answer_format(self, format_instructions: str) -> None:
        """
        Update the formatting instructions for final answers and rebuild the system prompt.

        Parameters:
            format_instructions (str): New formatting instructions for final answers.
        """
        self.final_answer_format_instructions = format_instructions
        self.system_prompt = build_system_prompt(self.custom_instructions, format_instructions)
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
            raise ValueError(f"Invalid timezone: {timezone}") from e

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