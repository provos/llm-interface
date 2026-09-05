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
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from anthropic import Anthropic, APIConnectionError, APIError, APITimeoutError
from ollama import ListResponse

from . import errors
from .utils import encode_image_to_base64


def translate_tools_for_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translate a list of tools from Ollama/API format to Anthropic format.

    Args:
        tools (List[Tool]): List of tool objects from the Ollama/API.

    Returns:
        List[Dict[str, Any]]: Translated tools ready for Anthropic API consumption.
    """
    anthropic_tools = []

    for tool in tools:
        # Extract the function from the tool
        function = tool["function"]

        # Assuming Tool objects have keys 'name', 'description', and 'parameters' which is a dict
        input_schema = {
            "type": "object",
            "properties": function["parameters"]["properties"],
            "required": function["parameters"]["required"],
        }
        translated_tool = {
            "name": function["name"],
            "description": function["description"],
            "input_schema": input_schema,
        }

        # Strict tool use: mirror OpenAI's `strict` flag onto Anthropic's schema.
        # Anthropic expects `strict` alongside the tool definition and
        # `additionalProperties: False` inside the input_schema itself.
        if function.get("strict"):
            translated_tool["strict"] = True
            input_schema["additionalProperties"] = False

        anthropic_tools.append(translated_tool)

    return anthropic_tools


def translate_messages_for_anthropic(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Translate messages from Ollama/API format to Anthropic format.

    An assistant message carrying N tool_calls becomes ONE assistant message
    whose content is an optional text block (only when the message has
    non-empty content) followed by N `tool_use` blocks. Consecutive `tool` role
    messages are merged into a single `user` message containing one
    `tool_result` block per call - Anthropic requires every tool_result for a
    turn to be returned together. Plain user/assistant/system-free
    conversations pass through unchanged, so this function is safe to call
    unconditionally.

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries in Ollama format

    Returns:
        List[Dict[str, Any]]: Translated messages in Anthropic format
    """
    translated_messages: List[Dict[str, Any]] = []
    # Reference to the content list of the most recently appended tool_result
    # group, so consecutive tool messages get merged into one user message.
    current_tool_result_group: Optional[List[Dict[str, Any]]] = None

    for msg in messages:
        if "images" in msg and msg["images"]:
            content = [{"type": "text", "text": msg["content"]}]
            for image in msg["images"]:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "data": encode_image_to_base64(image),
                            "media_type": "image/jpeg",
                        },
                    }
                )
            translated_messages.append({"role": "user", "content": content})
            current_tool_result_group = None

        elif msg["role"] == "user":
            # Regular user messages pass through unchanged
            translated_messages.append({"role": "user", "content": msg["content"]})
            current_tool_result_group = None

        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            # Convert every tool call made in this turn into one tool_use block
            # on a single assistant message.
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})

            for tool_call in msg["tool_calls"]:
                function = tool_call["function"]
                tool_input = function["arguments"]
                if isinstance(tool_input, str):
                    tool_input = json.loads(tool_input)
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": function["name"],
                        "input": tool_input,
                    }
                )

            translated_messages.append({"role": "assistant", "content": content})
            current_tool_result_group = None

        elif msg["role"] == "tool":
            # Convert tool response to Anthropic's tool_result format, merging
            # consecutive tool messages into one user message.
            tool_result: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg["content"],
            }
            if msg.get("is_error"):
                tool_result["is_error"] = True

            if current_tool_result_group is not None:
                current_tool_result_group.append(tool_result)
            else:
                current_tool_result_group = [tool_result]
                translated_messages.append(
                    {"role": "user", "content": current_tool_result_group}
                )

        elif msg["role"] == "assistant":
            # Assistant messages with plain string content pass through.
            translated_messages.append(msg)
            current_tool_result_group = None
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")

    return translated_messages


def convert_anthropic_models_to_ollama_response(
    models_data: Dict[str, Any],
) -> ListResponse:
    """
    Converts Anthropic model list API response to Ollama format.

    Args:
        models_data: The response from Anthropic's models API endpoint.

    Returns:
        An instance of ollama's ListResponse.
    """
    ollama_models = []
    for model_data in models_data["data"]:
        # Convert creation time from ISO format to datetime
        created_at = datetime.fromisoformat(
            model_data["created_at"].replace("Z", "+00:00")
        )

        model = {
            "model": model_data["id"],
            "modified_at": created_at,
            "digest": "unknown",
            "size": 0,
            "details": {
                "parent_model": "",
                "format": "unknown",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": "unknown",
                "quantization_level": "unknown",
                "display_name": model_data["display_name"],
            },
        }
        ollama_models.append(ListResponse.Model(**model))

    return ListResponse(models=ollama_models)


class AnthropicWrapper:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        prompt_caching: bool = True,
        thinking: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
    ):
        """
        Args:
            api_key (str): Anthropic API key.
            max_tokens (int): Default max_tokens for requests. Requests always
                stream, so large values are safe.
            timeout (float): Request timeout in seconds.
            prompt_caching (bool): When True (the default), every request carries
                top-level `cache_control={"type": "ephemeral"}`, which auto-caches
                the last cacheable block - useful for a growing tool-call
                transcript.
            thinking (Optional[Dict[str, Any]]): Extended thinking configuration,
                e.g. {"type": "adaptive"}. When set, `temperature` is never sent
                (current models reject it together with thinking) and
                `budget_tokens` is never used.
            effort (Optional[str]): One of low|medium|high|xhigh|max. Forwarded as
                `output_config.effort`.
        """
        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.prompt_caching = prompt_caching
        self.thinking = thinking
        self.effort = effort

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation using the Anthropic API.

        Args:
            messages (list[Mapping[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
            tools (Optional[List[Dict[str, Any]]]): Tool definitions in Ollama/OpenAI format.
            **kwargs: Additional arguments, including:
                - model (str): The Anthropic model to use.
                - max_tokens (int): Overrides the instance's max_tokens.
                - options (dict): May contain "temperature".
                - response_schema (Type[BaseModel]): When provided, it is passed as
                  `output_format` so the streamed final message carries a validated
                  `parsed_output`.
                - tool_choice (str | dict): "none" -> {"type": "none"},
                  "auto" -> {"type": "auto"}, or a dict passed through as-is.

        Returns:
            A dictionary containing the Anthropic response formatted to match Ollama's expected output.
        """
        # Extract the system message from the messages and prepare it as a separate argument
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )

        # Filter out the system message to prevent duplication if it's not needed in the messages parameter
        filtered_messages = [msg for msg in messages if msg["role"] != "system"]

        # Always translate: this normalizes tool_calls/tool-result/image
        # messages into Anthropic's format, and is a no-op for a plain
        # user/assistant conversation.
        filtered_messages = translate_messages_for_anthropic(filtered_messages)

        # Common parameters
        params: Dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": filtered_messages,
            "model": kwargs.get("model", "claude-3-5-sonnet-20240620"),
        }

        # Only include system parameter if it has a value
        if system_message is not None:
            params["system"] = system_message

        # Extended thinking / effort.
        if self.thinking is not None:
            params["thinking"] = self.thinking
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}

        # Conditionally add temperature if it exists in kwargs. Current models
        # reject `temperature` when adaptive thinking is configured, so only
        # send it when the caller explicitly asked for it AND thinking is off.
        if (
            self.thinking is None
            and "options" in kwargs
            and "temperature" in kwargs["options"]
        ):
            params["temperature"] = kwargs["options"]["temperature"]

        if tools:
            # Translate tools into Anthropic format
            anthropic_tools = translate_tools_for_anthropic(tools)
            params["tools"] = anthropic_tools

        # tool_choice: "none"/"auto" strings map to Anthropic's dict form; a
        # dict is passed through untouched.
        tool_choice = kwargs.get("tool_choice")
        if tool_choice == "none":
            params["tool_choice"] = {"type": "none"}
        elif tool_choice == "auto":
            params["tool_choice"] = {"type": "auto"}
        elif isinstance(tool_choice, dict):
            params["tool_choice"] = tool_choice

        response_schema = kwargs.get("response_schema")

        # Every request streams: `messages.stream` accepts `output_format`
        # (structured output, validated into `parsed_output`) together with
        # `cache_control`, and streaming avoids the SDK's timeout guard on
        # large `max_tokens` values.
        if response_schema is not None:
            params["output_format"] = response_schema
        if self.prompt_caching:
            # Auto-caches the last cacheable block - what a growing
            # tool-call transcript wants.
            params["cache_control"] = {"type": "ephemeral"}

        try:
            with self.client.messages.stream(**params) as stream:
                response = stream.get_final_message()

            # Extract usage information
            usage = response.usage
            usage_info = {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "cached_tokens": usage.cache_read_input_tokens or 0,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }

            if response.stop_reason == "refusal":
                stop_details = response.stop_details
                explanation = (
                    stop_details.explanation if stop_details is not None else None
                )
                return {
                    "refusal": explanation or "refused",
                    "content": None,
                    "done": False,
                }

            if response.stop_reason == "max_tokens":
                return {
                    "error": "Response exceeded the maximum allowed length.",
                    "error_type": errors.LENGTH,
                    "content": None,
                    "done": False,
                    "usage": None,
                }

            # Handle tool calls if present. This shape is returned whether or
            # not `response_schema` was requested (parsed_output is unused
            # here and stays None on the response in that case).
            tool_use_blocks = [
                block for block in response.content if block.type == "tool_use"
            ]
            if tool_use_blocks:
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_block.id,
                                "name": tool_block.name,
                                "arguments": tool_block.input,  # Anthropic uses 'input' instead of 'arguments'
                            }
                            for tool_block in tool_use_blocks
                        ],
                    },
                    "usage": usage_info,
                    "done": response.stop_reason == "end_turn",
                }

            if response_schema is not None:
                # `output_format` makes the final message a ParsedMessage with
                # `parsed_output`; `generate_pydantic` accepts a BaseModel here.
                return {
                    "message": {"content": getattr(response, "parsed_output", None)},
                    "usage": usage_info,
                    "done": response.stop_reason == "end_turn",
                }

            # Extract content blocks as text and simulate Ollama-like response
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )

            return {
                "message": {"content": content},
                "usage": usage_info,
                "done": response.stop_reason == "end_turn",
            }

        except APITimeoutError as e:
            error_message = f"Anthropic API timeout: {str(e)}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            }
        except APIConnectionError as e:
            error_message = f"Anthropic API connection error: {str(e)}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.CONNECTION,
                "content": None,
                "done": False,
                "usage": None,
            }
        except APIError as e:
            error_message = f"Anthropic API error: {str(e)}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.PROVIDER_SPECIFIC,
                "content": None,
                "done": False,
                "usage": None,
            }
        except Exception as e:
            # The streaming accumulator validates structured output as blocks
            # complete and raises a pydantic ValidationError when a text block
            # is empty or truncated (for example when thinking exhausted
            # max_tokens). Surface it as a model error so generate_pydantic
            # can retry instead of crashing the caller.
            error_message = f"Anthropic response could not be parsed: {str(e)}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.PROVIDER_SPECIFIC,
                "content": None,
                "done": False,
                "usage": None,
            }

    def list(self) -> ListResponse:
        """
        Returns a list of available Anthropic models in Ollama format.
        Uses the Anthropic API to get the current list of models.
        """
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

        response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
        response.raise_for_status()

        return convert_anthropic_models_to_ollama_response(response.json())
