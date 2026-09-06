# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import diskcache
from dotenv import load_dotenv
from ollama import Client, ListResponse
from pydantic import BaseModel, Field, create_model

from . import errors
from .llm_tool import Tool
from .ollama import OllamaWrapper
from .pydantic_output_parser import MinimalPydanticOutputParser
from .remote_ollama import RemoteOllama
from .token_usage import TokenUsage
from .utils import setup_logging

# Load environment variables from .env.local file
load_dotenv(".env.local")


class NoCache:
    def get(self, key):
        return None

    def set(self, key, value):
        pass


class ModelError(Exception):
    pass


class LLMInterface:
    """
    A unified interface for interacting with various Language Learning Models (LLMs).

    This class provides a consistent way to interact with different LLM providers
    (e.g., Ollama, OpenAI, Anthropic) while handling caching, logging, tool execution,
    and structured output generation. It supports both raw text generation and
    schema-validated outputs using Pydantic models.

    Attributes:
        model_name (str): Name of the LLM model to use.
        log_dir (str): Path in which to store the logs from the LLMInterface.
        client: The underlying LLM client instance (e.g., Ollama Client, OpenAI client).
        support_json_mode (bool): Whether the model supports JSON output mode.
        support_structured_outputs (bool): Whether the model supports structured outputs.
        support_system_prompt (bool): Whether the model supports system prompts.
        requires_thinking (bool): Whether the model requires a 'thinking' field in the output to work.
        disk_cache: Cache instance for storing and retrieving model responses.
        token_usage: TokenUsage instance for tracking token usage across requests.
        timeout (float): Timeout in seconds for API requests.
        max_retries (int): Maximum number of retries for API requests.
        retry_delay (float): Delay in seconds between retries for API requests.
        max_tool_rounds (int): Maximum number of tool-call round-trips allowed in a single
            chat() call before a final answer is forced. Defaults to 5.

    Example:
        >>> llm = LLMInterface(
        ...     model_name="llama2",
        ...     log_dir="logs",
        ...     support_json_mode=True
        ... )
        >>> response = llm.chat([
        ...     {"role": "user", "content": "What is the capital of France?"}
        ... ])
        >>> print(response)
        'The capital of France is Paris.'

        >>> # Using with Pydantic models for structured output
        >>> from pydantic import BaseModel
        >>> class CityInfo(BaseModel):
        ...     city: str
        ...     country: str
        ...     population: int
        >>> result = llm.generate_pydantic(
        ...     "Describe Paris",
        ...     output_schema=CityInfo,
        ...     system="You are a helpful assistant."
        ... )
    """

    def __init__(
        self,
        model_name: str = "llama2",
        log_dir: str = "logs",
        client: Optional[Any] = None,
        host: Optional[str] = None,
        support_json_mode: bool = True,
        support_structured_outputs: bool = False,
        support_system_prompt: bool = True,
        requires_thinking: bool = False,
        use_cache: bool = True,
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_tool_rounds: int = 5,
    ):
        self.model_name = model_name
        self.client = client if client else OllamaWrapper(host=host, timeout=timeout)
        self.support_json_mode = support_json_mode
        self.support_structured_outputs = support_structured_outputs
        self.support_system_prompt = support_system_prompt
        self.requires_thinking = requires_thinking
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tool_rounds = max_tool_rounds

        self.logger = setup_logging(
            logs_dir=log_dir, logs_prefix="llm_interface", logger_name=__name__
        )

        # Initialize disk cache for caching responses
        self.disk_cache = (
            diskcache.Cache(
                directory=".response_cache", eviction_policy="least-recently-used"
            )
            if use_cache
            else NoCache()
        )

        # Initialize token usage tracker
        self.token_usage = TokenUsage()

    def list(self) -> ListResponse:
        """List available models."""
        return self.client.list()

    def _inject_thinking_if_needed(
        self, output_object: Type[BaseModel]
    ) -> Type[BaseModel]:
        """
        Injects a 'thinking' field into the BaseModel if it doesn't exist.

        Args:
            output_object: The original Pydantic model class

        Returns:
            A new Pydantic model class with the thinking field added as the first field,
            or the original if it already has the thinking field.
        """
        # Check if thinking field already exists
        if "thinking" in output_object.model_fields:
            return output_object

        # Create ordered field definitions with thinking first
        field_definitions = {
            "thinking": (str, Field(..., description="Model's thinking process")),
        }

        # Add all existing fields from the original model
        for field_name, field in output_object.model_fields.items():
            field_definitions[field_name] = (
                field.annotation,
                field.default if field.default != ... else ...,
            )

        # Create new model with ordered fields
        new_model = create_model(
            f"{output_object.__name__}WithThinking", **field_definitions
        )

        return new_model

    def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], tools: List[Tool]
    ) -> List[Dict[str, Any]]:
        """Execute every tool call made in a single assistant turn, as one unit.

        Builds one assistant message carrying the full ``tool_calls`` list (each
        entry ``{"id", "type": "function", "function": {"name", "arguments"}}``),
        followed by one ``{"role": "tool", ...}`` message per call, in the same
        order the calls were made. This keeps a multi-tool-call turn intact for
        providers (like Anthropic) that require all `tool_result` blocks to be
        returned together.

        A tool call is never silently dropped: if the tool raises, is unknown, or
        its arguments fail to parse as JSON, a tool message is still emitted with
        content ``"Error: <message>"`` and an extra ``"is_error": True`` key, so
        the conversation continues and the model gets a chance to recover.

        Args:
            tool_calls (List[Dict[str, Any]]): The tool calls from the assistant's response.
            tools (List[Tool]): The tools available for execution.

        Returns:
            List[Dict[str, Any]]: ``[assistant_message, tool_message, ...]`` - one
            assistant message followed by one tool message per call, in order.
        """
        tool_map = {t.name: t for t in tools} if tools else {}

        assistant_tool_calls: List[Dict[str, Any]] = []
        tool_messages: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get(
                "name"
            )
            call_id = tool_call.get("id", "")
            arguments = tool_call.get("arguments")
            if arguments is None:
                arguments = tool_call.get("function", {}).get("arguments", {})

            parse_error = None
            if isinstance(arguments, str):
                # Parse JSON string if needed
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as e:
                    parse_error = f"Failed to parse tool arguments: {e}"
                    self.logger.error(parse_error)

            assistant_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,  # Ollama requires this as a Dict but OpenAI requires it as a string
                    },
                }
            )

            if parse_error is not None:
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": f"Error: {parse_error}",
                        "is_error": True,
                    }
                )
                continue

            if tool_name not in tool_map:
                error_message = f"Tool '{tool_name}' not found."
                self.logger.error(error_message)
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": f"Error: {error_message}",
                        "is_error": True,
                    }
                )
                continue

            try:
                result = tool_map[tool_name].execute(**arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": str(result),
                    }
                )
            except Exception as e:
                error_message = f"Tool execution failed: {e}"
                self.logger.error(error_message)
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": f"Error: {error_message}",
                        "is_error": True,
                    }
                )

        assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": assistant_tool_calls,
        }

        return [assistant_message] + tool_messages

    def _create_prompt_hash(
        self,
        model_name: str,
        message_content: str,
        tool_content: str,
        temperature: Optional[float] = None,
        cache_salt: Optional[str] = None,
    ) -> str:
        """Create a hash of the prompt for caching.

        Args:
            model_name (str): Name of the model being used.
            message_content (str): Concatenated content of all messages.
            tool_content (str): Concatenated name/description of all tools.
            temperature (Optional[float]): Sampling temperature, if any.
            cache_salt (Optional[str]): Optional extra value mixed into the hash so
                callers whose tools read external state (e.g. files on disk) can
                invalidate the response cache without changing the conversation.
        """
        return self._generate_hash(
            model_name
            + (f"-{temperature}" if temperature is not None else "")
            + message_content
            + tool_content
            + (f"-{cache_salt}" if cache_salt else "")
        )

    def _chat_once(
        self,
        current_messages: List[Dict[str, Any]],
        converted_tools: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        token_usage: TokenUsage,
    ) -> Dict[str, Any]:
        """Make a single request to the underlying client.

        Retries on timeout/connection errors (per ``self.max_retries`` /
        ``self.retry_delay``) and updates ``token_usage`` from the response.
        """
        retry_count = 0
        response: Dict[str, Any] = {}
        while retry_count <= self.max_retries:
            response = self.client.chat(
                model=self.model_name,
                tools=converted_tools,
                messages=current_messages,
                **kwargs,
            )

            self.logger.info("Received chat response: %s", response)

            # Check for timeout error
            if (
                "error" in response
                and "error_type" in response
                and (
                    response.get("error_type") == errors.TIMEOUT
                    or response.get("error_type") == errors.CONNECTION
                )
            ):
                retry_count += 1
                if retry_count <= self.max_retries:
                    self.logger.warning(
                        "Request error (%s). Retrying (%d/%d) after %.1f seconds...",
                        response["error"],
                        retry_count,
                        self.max_retries,
                        self.retry_delay * retry_count,
                    )
                    time.sleep(self.retry_delay * retry_count)
                    continue
                else:
                    self.logger.error(
                        "Request failed out after %d retries.", self.max_retries
                    )
                    break
            else:
                # Not a timeout error, proceed normally
                break

        # Special handling for Ollama client
        if isinstance(self.client, Client):
            token_usage.update(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )
        # Extract token usage information from response if available
        elif response.get("usage"):
            usage_data = response["usage"]
            token_usage.update(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cached_tokens=usage_data.get("cached_tokens", 0),
                reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            )

        return response

    def _cached_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        token_usage: Optional[TokenUsage] = None,
        allow_json_mode: bool = True,
        max_tool_rounds: Optional[int] = None,
        cache_salt: Optional[str] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Execute a chat conversation with caching and optional tool execution.

        This method handles caching of chat responses and supports structured outputs,
        JSON mode, and tool execution. It will retrieve cached responses when available
        or make new API calls when needed.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys
            tools (Optional[List[Tool]], optional): List of Tool objects that can be called by the LLM. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature for response generation. Defaults to None.
            response_schema (Optional[Type[BaseModel]], optional): Pydantic model for structured output. Defaults to None.
            token_usage (Optional[TokenUsage]): Object to track token usage. If None, uses self.token_usage.
            allow_json_mode (bool): Whether to allow JSON mode for the response. Defaults to True.
            max_tool_rounds (Optional[int]): Per-call override for the number of tool-call
                round-trips allowed before a final answer is forced. Defaults to
                ``self.max_tool_rounds`` when None.
            cache_salt (Optional[str]): Optional value mixed into the cache key so callers
                whose tools read external state can invalidate the response cache.
            transcript (Optional[List[Dict[str, Any]]]): When a list is given, it is
                filled with the full conversation as sent on the last request (the
                initial messages plus every tool call and tool result), excluding the
                final assistant answer, so a caller can continue the conversation
                without repeating the tool work. Left empty on a cache hit.

        Returns:
            str: The content of the chat response message

        Raises:
            ModelError: If the model returns an error or refuses the request

        Note:
            - Caching is based on a hash of the model name, messages, tools, temperature,
              and (if provided) cache_salt
            - Supports up to ``max_tool_rounds`` sequential tool-call rounds per conversation;
              if the model still wants to call tools after the limit is reached, one final
              request is made with ``tool_choice="none"`` to force a textual answer
            - Compatible with both OpenAI and Ollama clients
        """
        # Use provided token_usage or default to self.token_usage
        token_usage = token_usage or self.token_usage
        effective_max_tool_rounds = (
            max_tool_rounds if max_tool_rounds is not None else self.max_tool_rounds
        )

        # Concatenate all messages to use as the cache key
        message_content = "".join(
            [
                (
                    msg["role"]
                    + msg["content"]
                    + (str(msg["images"]) if "images" in msg else "")
                )
                for msg in messages
            ]
        )
        tool_content = ""
        if tools:
            tool_content = "".join([f"{t.name}{t.description}" for t in tools])
        prompt_hash = self._create_prompt_hash(
            model_name=self.model_name,
            message_content=message_content,
            tool_content=tool_content,
            temperature=temperature,
            cache_salt=cache_salt,
        )

        self.logger.info("Chatting with messages: %s", messages)

        # Check if prompt response is in cache
        response = self.disk_cache.get(prompt_hash)

        if response is None:
            kwargs: Dict[str, Any] = {}
            current_messages = messages.copy()

            # some models can generate structured outputs
            if allow_json_mode:
                if self.support_structured_outputs and response_schema:
                    if isinstance(self.client, Client):
                        # For Ollama, we need to pass the schema directly
                        kwargs["format"] = response_schema.model_json_schema()
                    else:
                        # For OpenAI, we need to pass the schema as a pydantic object
                        kwargs["response_schema"] = response_schema
                elif self.support_json_mode:
                    kwargs["format"] = "json"

            # ollama expects temperature to be passed as an option
            options = {}
            if temperature is not None:
                options["temperature"] = temperature
                kwargs["options"] = options

            converted_tools = [tool.to_dict() for tool in tools] if tools else []

            # Ollama's native client (and the SSH-tunneled RemoteOllama) have a
            # fixed keyword signature and raise on unknown kwargs, so only pass
            # `tool_choice` to clients that can actually accept it.
            client_accepts_tool_choice = not isinstance(
                self.client, Client
            ) and not isinstance(self.client, RemoteOllama)

            num_tool_rounds = 0
            while True:
                num_tool_rounds += 1

                response = self._chat_once(
                    current_messages, converted_tools, kwargs, token_usage
                )

                # Check if the response contains tool calls
                tool_calls = response.get("message", {}).get("tool_calls", [])
                if not tool_calls:
                    break

                self.logger.info("Received tool calls: %s", tool_calls)
                # Execute all tool calls from this turn as one unit and add
                # the results to the conversation.
                tool_messages = self._execute_tool_calls(tool_calls, tools or [])
                current_messages.extend(tool_messages)

                if num_tool_rounds >= effective_max_tool_rounds:
                    # We've used up the allowed tool-call rounds and the model
                    # still wants to call tools. Rather than silently return an
                    # empty response, ask explicitly for a final answer and
                    # force the model to stop calling tools.
                    self.logger.warning(
                        "Reached max_tool_rounds (%d) with pending tool calls; "
                        "forcing a final answer.",
                        effective_max_tool_rounds,
                    )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The tool-call limit has been reached. You must "
                                "provide a final answer now without calling any "
                                "more tools."
                            ),
                        }
                    )
                    self.logger.info("Chatting with messages: %s", current_messages)

                    final_kwargs = dict(kwargs)
                    if client_accepts_tool_choice:
                        final_kwargs["tool_choice"] = "none"

                    response = self._chat_once(
                        current_messages, converted_tools, final_kwargs, token_usage
                    )
                    break

                self.logger.info("Chatting with messages: %s", current_messages)

            if transcript is not None:
                transcript.clear()
                transcript.extend(current_messages)

            # Cache the response with hashed prompt as key
            try:
                self.disk_cache.set(prompt_hash, response)
            except Exception as e:
                self.logger.error("Error caching response: %s", e)

        if "error" in response:
            self.logger.error("Error in chat response: %s", response["error"])
            raise ModelError("Generic error: " + response["error"])
        elif "refusal" in response:
            self.logger.error("Model refused the request: %s", response["refusal"])
            raise ModelError("Model refusal: " + response["refusal"])

        return response["message"]["content"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        token_usage: Optional[TokenUsage] = None,
        allow_json_mode: bool = True,
        max_tool_rounds: Optional[int] = None,
        cache_salt: Optional[str] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Sends a chat request to the LLM and returns the response.

        This method handles both regular chat responses and structured responses based on the provided schema.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries containing the conversation history.
                Each message should have 'role' and 'content' keys.
            tools (Optional[List[Tool]], optional): List of tools/functions available to the LLM. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature for response generation.
                Higher values make output more random, lower values more deterministic. Defaults to None.
            response_schema (Optional[Type[BaseModel]], optional): Pydantic model defining the expected response structure.
                If provided, the response will be parsed according to this schema. Defaults to None.
            token_usage (Optional[TokenUsage]): Object to track token usage. If None, uses self.token_usage.
            allow_json_mode (bool): Whether to allow JSON mode for the response. Defaults to True. Set False for a straight chat experience.
            max_tool_rounds (Optional[int]): Per-call override for the number of tool-call round-trips
                allowed before a final answer is forced. Defaults to ``self.max_tool_rounds`` when None.
            cache_salt (Optional[str]): Optional value mixed into the cache key, useful when tools read
                external state (e.g. files) so a stale cached response can be invalidated on demand.

        Returns:
            str: The LLM's response text, stripped of leading/trailing whitespace if string,
                or the structured response if a schema was provided.

        Raises:
            ModelError: If the model returns an error or refuses the request
        """
        # Use provided token_usage or default to self.token_usage
        token_usage = token_usage or self.token_usage

        response = self._cached_chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            response_schema=response_schema,
            token_usage=token_usage,
            allow_json_mode=allow_json_mode,
            max_tool_rounds=max_tool_rounds,
            cache_salt=cache_salt,
            transcript=transcript,
        )
        self.logger.info(
            "Received chat response: %s...",
            response[:850] if isinstance(response, str) else response,
        )
        return response.strip() if isinstance(response, str) else response

    def _strip_text_from_json_response(self, response: str) -> str:
        pattern = r"^[^{\[]*([{\[].*[}\]])[^}\]]*$"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1)
        else:
            return response  # Return original response if no JSON block is found

    def generate_full_prompt(
        self, prompt_template: str, system: str = "", **kwargs
    ) -> str:
        """
        Generate a full prompt with input variables filled in.

        Args:
            prompt_template (str): The prompt template with placeholders for variables.
            system (str): The system prompt to use for generation.
            **kwargs: Keyword arguments to fill in the prompt template.

        Returns:
            str: The formatted prompt
        """
        formatted_prompt = prompt_template.format(**kwargs)
        return formatted_prompt

    def generate_pydantic(
        self,
        prompt_template: str,
        output_schema: Type[BaseModel],
        system: str = "",
        tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        debug_saver: Optional[Callable[[str, Dict[str, Any], BaseModel], None]] = None,
        extra_validation: Optional[Callable[[BaseModel], Optional[str]]] = None,
        temperature: Optional[float] = None,
        token_usage: Optional[TokenUsage] = None,
        images: Optional[List[str]] = None,
        max_tool_rounds: Optional[int] = None,
        cache_salt: Optional[str] = None,
        **kwargs,
    ) -> Optional[BaseModel]:
        """
        Generates a Pydantic model instance based on a specified prompt template and output schema.

        This function uses a prompt template with variable placeholders to generate a full prompt. It utilizes
        this prompt in combination with a specified system prompt to interact with a chat-based interface,
        aiming to produce a structured output conforming to a given Pydantic schema. The function attempts up to
        three iterations to obtain a valid response, applying parsing, validation, and optional extra validation
        functions. If all iterations fail, None is returned.

        Args:
            prompt_template (str): The template containing placeholders for formatting the prompt.
            output_schema (Type[BaseModel]): A Pydantic model that defines the expected schema of the output data.
            system (str): An optional system prompt used during the generation process.
            logger (Optional[logging.Logger]): An optional logger for recording the generated prompt and events.
            debug_saver (Optional[Callable[[str, Dict[str, Any], BaseModel], None]]): An optional callback for saving debugging information,
                which receives the prompt and the response.
            extra_validation (Optional[Callable[[BaseModel], str]]): An optional function for additional validation of
                the generated output. It should return an error message if validation fails, otherwise None.
            token_usage (Optional[TokenUsage]): Object to track token usage. If None, uses self.token_usage.
            images (Optional[List[str]]): List of image paths to encode and send to the model.
            max_tool_rounds (Optional[int]): Per-call override for the number of tool-call round-trips
                allowed before a final answer is forced. Passed through to ``chat()``.
            cache_salt (Optional[str]): Optional value mixed into the cache key. Passed through to ``chat()``.
            **kwargs: Additional keyword arguments for populating the prompt template.

        Returns:
            Optional[BaseModel]: An instance of the specified Pydantic model with generated data if successful,
            or None if all attempts at generation fail or the response is invalid.
        """
        # Use provided token_usage or default to self.token_usage
        token_usage = token_usage or self.token_usage

        parser = MinimalPydanticOutputParser(pydantic_object=output_schema)

        formatted_prompt = self.generate_full_prompt(
            prompt_template=prompt_template, system=system, **kwargs
        )

        self.logger.info("Generated prompt: %s", formatted_prompt)
        if logger:
            logger.info("Generated prompt: %s", formatted_prompt)

        messages = []
        if self.support_system_prompt:
            messages.append({"role": "system", "content": system})
        message = {"role": "user", "content": formatted_prompt}
        if images:
            message["images"] = images
        messages.append(message)

        new_output_schema = output_schema
        if self.requires_thinking:
            new_output_schema = self._inject_thinking_if_needed(output_schema)

        response = None
        iteration = 0
        while iteration < 3:
            iteration += 1

            # A retry continues the conversation that produced the bad answer,
            # tool calls and results included, so the model keeps what it read.
            transcript: List[Dict[str, Any]] = []
            try:
                raw_response = self.chat(
                    messages=messages,
                    temperature=temperature,
                    response_schema=new_output_schema,
                    tools=tools,
                    token_usage=token_usage,
                    max_tool_rounds=max_tool_rounds,
                    cache_salt=cache_salt,
                    transcript=transcript,
                )
            except ModelError as e:
                raw_response = None
                messages = (transcript or messages) + [
                    {"role": "assistant", "content": str(e)},
                    {
                        "role": "user",
                        "content": "Try again while avoiding the previous error.",
                    },
                ]
                continue

            if self.support_structured_outputs:
                # If the model supports structured outputs, we should get a Pydantic object directly
                # or a string that can be parsed directly
                try:
                    if raw_response is None:
                        response = None
                        error_message = "The model refused the request"
                    elif isinstance(raw_response, BaseModel):
                        response = raw_response
                        error_message = None
                    else:
                        response = output_schema.model_validate_json(raw_response)
                        error_message = None
                except Exception as e:
                    self.logger.error("Error parsing structured response: %s", e)
                    error_message = str(e)
                    response = None
            else:
                if not self.support_json_mode:
                    raw_response = self._strip_text_from_json_response(raw_response)
                error_message, response = self._parse_response(raw_response, parser)

            if response is None:
                messages = (transcript or messages) + [
                    {"role": "assistant", "content": raw_response or ""},
                    {
                        "role": "user",
                        "content": f"Try again. Your previous response was invalid and led to this error message: {error_message}",
                    },
                ]
                continue

            if extra_validation:
                extra_error_message = extra_validation(response)
                if extra_error_message:
                    if isinstance(response, BaseModel):
                        # the response was a pydantic object, so we need to dump it to a string
                        raw_response = response.model_dump_json()
                    elif not isinstance(raw_response, str):
                        raise ValueError(
                            "The response should be a string if the model does not support structured outputs."
                        )
                    messages = (transcript or messages) + [
                        {"role": "assistant", "content": raw_response},
                        {
                            "role": "user",
                            "content": f"Try again. Your previous response was invalid and led to this error message: {extra_error_message}",
                        },
                    ]
                    continue
            break

        if debug_saver is not None:
            assert isinstance(response, BaseModel)
            debug_saver(formatted_prompt, kwargs, response)

        if output_schema != new_output_schema and response is not None:
            dump = response.model_dump()
            dump.pop("thinking", None)  # Remove the thinking field from the response
            response = output_schema.model_validate(dump)

        return response

    def _generate_hash(self, prompt: str) -> str:
        hash_object = hashlib.sha256(prompt.encode())
        return hash_object.hexdigest()

    def _parse_response(
        self, response: str, parser: MinimalPydanticOutputParser
    ) -> Tuple[str, Dict[str, Any]]:
        self.logger.info("Parsing JSON response: %s", response)
        error_message = None
        try:
            response = parser.parse(response)
        except Exception as e:
            self.logger.error("Error parsing response: %s", e)
            error_message = str(e)
            response = None
        return error_message, response

    @staticmethod
    def get_format_instructions(pydantic_object: Type[BaseModel]) -> str:
        """
        Generate format instructions for a Pydantic model's JSON output.

        This function creates a string of instructions on how to format JSON output
        based on the schema of a given Pydantic model. It's compatible with both
        Pydantic v1 and v2.

        Args:
            pydantic_object (Type[BaseModel]): The Pydantic model class to generate instructions for.

        Returns:
            str: A string containing the format instructions.

        Note:
            This function is adapted from the LangChain framework.
            Original source: https://github.com/langchain-ai/langchain
            License: MIT (https://github.com/langchain-ai/langchain/blob/master/LICENSE)
        """
        _PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```
"""
        schema = pydantic_object.model_json_schema().copy()

        schema.pop("title", None)
        schema.pop("type", None)

        schema_str = json.dumps(schema, ensure_ascii=False)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)
