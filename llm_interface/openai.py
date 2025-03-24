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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ollama import ListResponse
from openai import (
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
)

from . import errors


def translate_tools_for_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = messages.copy()
    for message in messages:
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if (
                    "function" not in tool_call
                    or "arguments" not in tool_call["function"]
                ):
                    continue
                if isinstance(tool_call["function"]["arguments"], dict):
                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"]
                    )

    return messages


def transform_messages_with_images(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Transform messages to include base64-encoded images in OpenAI's required format.

    Args:
        messages (List[Dict[str, Any]]): Messages that may contain image paths

    Returns:
        List[Dict[str, Any]]: Transformed messages compatible with OpenAI's image API
    """
    from .utils import encode_image_to_base64

    transformed_messages = []

    for message in messages:
        new_message = {"role": message["role"]}

        if "images" in message and message["images"]:
            # Message contains images, transform to OpenAI format
            content_items = []

            # Add text content if it exists
            if "content" in message and message["content"]:
                content_items.append({"type": "text", "text": message["content"]})

            # Add image content
            for image_path in message["images"]:
                b64_image = encode_image_to_base64(image_path)
                # Determine image type from file extension
                image_type = image_path.split(".")[-1].lower()
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{b64_image}"
                        },
                    }
                )

            new_message["content"] = content_items
        else:
            # Regular message without images
            new_message["content"] = message.get("content", "")

        transformed_messages.append(new_message)

    return transformed_messages


def convert_openai_models_to_ollama_response(openai_models_data) -> ListResponse:
    """
    Converts the output of openai.client.list() into ollama's ListResponse format.

    Args:
        openai_models_data: The output of openai.client.list(), which is a SyncPage[Model] object.

    Returns:
        An instance of ollama's ListResponse.
    """
    ollama_models = []
    for openai_model in openai_models_data.data:
        # Convert creation time to datetime with UTC timezone
        modified_at = datetime.fromtimestamp(openai_model.created, tz=timezone.utc)

        # Create model details matching Ollama's format
        details = {
            "parent_model": "",
            "format": "unknown",
            "family": "openai",
            "families": ["openai"],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        }

        model = {
            "model": openai_model.id,
            "modified_at": modified_at,
            "digest": "unknown",
            "size": 0,
            "details": details,
        }
        ollama_models.append(ListResponse.Model(**model))

    return ListResponse(models=ollama_models)


class OpenAIWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096, timeout: float = 600.0):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_tokens = max_tokens

    def list(self) -> ListResponse:
        return convert_openai_models_to_ollama_response(self.client.models.list())

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation using the OpenAI API, with optional structured output.

        This method interfaces with the OpenAI API to provide conversational capabilities.
        It supports structured outputs using Pydantic models to ensure responses adhere
        to specific JSON schemas if a `response_schema` is provided.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries representing the conversation
                                             history, where each dictionary contains 'role' (e.g., 'system',
                                             'user', 'assistant') and 'content' (the message text).
            tools (Optional[List[Dict[str, Any]]]): List of tools to be used in the conversation.
            **kwargs: Additional parameters that can include:
                - model (str): The OpenAI model to be used. Defaults to 'gpt-3.5-turbo' if not specified.
                - max_tokens (int): Maximum number of tokens for the API call. Defaults to the instance's max_tokens.
                - temperature (float): The temperature setting for the response generation.
                - response_schema (Type[BaseModel]): An optional Pydantic model for structured output.

        Returns:
            Dict[str, Any]: If a `response_schema` is specified, returns a dictionary containing parsed content
                            according to the schema. If the model refuses the request, it returns a refusal message.
                            Without a `response_schema`, it returns the OpenAI response formatted with the message content.

        Raises:
            Exception: Propagates any exceptions raised during the API interaction.

        Additional support for image-based messages with the following format:
        {'role': 'user', 'content': 'text prompt', 'images': ['/path/to/image1.jpg', '/path/to/image2.jpg']}
        """
        # Check if any messages contain images
        has_images = any("images" in message for message in messages)

        if has_images:
            # Transform messages to include base64-encoded images
            messages = transform_messages_with_images(messages)
        else:
            # Standard message format transformation for tool calls
            messages = translate_tools_for_openai(messages)

        api_params = {
            "model": kwargs.get("model", "gpt-3.5-turbo"),
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools:
            # Convert tools to OpenAI function format which at the moment is identical to the tool format
            api_params["tools"] = tools

        if "options" in kwargs:
            if "temperature" in kwargs["options"]:
                api_params["temperature"] = kwargs["options"]["temperature"]

        logging.debug("API parameters: %s", api_params)

        try:
            if "response_schema" in kwargs:
                response = self.client.beta.chat.completions.parse(
                    response_format=kwargs.get("response_schema"),
                    **api_params,
                )
                message = response.choices[0].message

                # Check for refusal
                if "refusal" in message:
                    return {"refusal": message.refusal, "content": None, "done": False}

                content = message.parsed
            else:
                if "format" in kwargs and kwargs["format"] == "json":
                    api_params["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**api_params)
                message = response.choices[0].message
                content = message.content

            # Log the usage details
            usage = response.usage
            logging.info(
                "Usage details - Prompt tokens: %d, Completion tokens: %d, Total tokens: %d, Cached tokens: %d, Reasoning tokens: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                (
                    usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details
                    else 0
                ),
                (
                    usage.completion_tokens_details.reasoning_tokens
                    if usage.completion_tokens_details
                    else 0
                ),
            )

            return_message = {
                "message": {"content": content},
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "done": response.choices[0].finish_reason == "stop",
            }
            if usage.prompt_tokens_details:
                return_message["usage"][
                    "cached_tokens"
                ] = usage.prompt_tokens_details.cached_tokens
            # Check for tool calls
            if message.tool_calls:
                return_message["message"]["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    for tool_call in message.tool_calls
                ]

            return return_message
        except LengthFinishReasonError:
            # Handle the length error
            return {
                "error": "Response exceeded the maximum allowed length.",
                "error_type": errors.LENGTH,
                "content": None,
                "done": False,
                "usage": None,
            }
        except ContentFilterFinishReasonError:
            # Handle the content filter error
            return {
                "error": "Content was rejected by the content filter.",
                "error_type": errors.CONTENT_FILTER,
                "content": None,
                "done": False,
                "usage": None,
            }
        except APITimeoutError:
            # Handle the timeout error
            return {
                "error": "The request timed out.",
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            }
        except Exception as e:
            raise e
