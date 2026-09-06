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
"""OpenAI Responses API wrapper.

The Responses API is where OpenAI's reasoning models expose reasoning together
with function tools (chat completions only allows tools with
``reasoning_effort="none"``). Requests are stateless (``store=False``): the whole
transcript is translated into input items on every round, and the items the
model produced in a tool-calling turn (reasoning, function calls) are replayed
verbatim on the next round so the model keeps its reasoning across tool calls.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ollama import ListResponse
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
    pydantic_function_tool,
)

from . import errors
from .openai import convert_openai_models_to_ollama_response
from .utils import encode_image_to_base64

# Key on an assistant message under which LLMInterface keeps the raw output items
# of a tool-calling turn so this wrapper can replay them.
PROVIDER_ITEMS = "provider_items"


def translate_tools_for_responses(
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Chat-completions style tools (``{"type": "function", "function": {...}}``)
    to the flat Responses API shape."""
    translated = []
    for tool in tools or []:
        function = tool.get("function", tool)
        translated.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
                "strict": bool(function.get("strict", False)),
            }
        )
    return translated


def translate_messages_for_responses(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Ollama/API style messages to (instructions, input items).

    System messages become the ``instructions`` string; an assistant turn with
    tool calls becomes its replayed provider items when it has them, otherwise
    one ``function_call`` item per call; tool messages become
    ``function_call_output`` items.
    """
    instruction_parts: List[str] = []
    items: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            if msg.get("content"):
                instruction_parts.append(msg["content"])
        elif role == "user":
            if msg.get("images"):
                content: List[Dict[str, Any]] = [
                    {"type": "input_text", "text": msg.get("content", "")}
                ]
                for image in msg["images"]:
                    image_type = image.split(".")[-1].lower()
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{image_type};base64,"
                            + encode_image_to_base64(image),
                        }
                    )
                items.append({"role": "user", "content": content})
            else:
                items.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant" and msg.get("tool_calls"):
            if msg.get(PROVIDER_ITEMS):
                items.extend(msg[PROVIDER_ITEMS])
                continue
            if msg.get("content"):
                items.append({"role": "assistant", "content": msg["content"]})
            for tool_call in msg["tool_calls"]:
                function = tool_call.get("function", tool_call)
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function["name"],
                        "arguments": arguments,
                    }
                )
        elif role == "tool":
            if not msg.get("tool_call_id"):
                raise ValueError(
                    "A tool message must carry the tool_call_id it answers; "
                    f"got {msg!r}"
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg["tool_call_id"],
                    "output": str(msg.get("content", "")),
                }
            )
        elif role == "assistant":
            items.append({"role": "assistant", "content": msg.get("content", "")})
        else:
            raise ValueError(f"Unsupported message role: {role!r}")

    instructions = "\n\n".join(instruction_parts) or None
    return instructions, items


def _json_schema_text_format(schema: Any) -> Dict[str, Any]:
    strict_schema = pydantic_function_tool(schema)["function"]["parameters"]
    return {
        "format": {
            "type": "json_schema",
            "name": schema.__name__,
            "schema": strict_schema,
            "strict": True,
        }
    }


class OpenAIResponsesWrapper:
    """Client for the OpenAI Responses API with the same ``chat()`` contract as
    the other wrappers."""

    # LLMInterface keeps the raw output items of tool-calling turns on the
    # transcript for clients that declare this
    keeps_provider_items = True

    def __init__(
        self,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Args:
            api_key (str): OpenAI API key.
            max_tokens (int): Default ``max_output_tokens`` for requests.
            timeout (float): Request timeout in seconds.
            reasoning_effort (Optional[str]): ``reasoning.effort`` for every
                request ("none", "low", "medium", "high", ...). None leaves the
                model default.
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def list(self) -> ListResponse:
        return convert_openai_models_to_ollama_response(self.client.models.list())

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        One request to the Responses API.

        Args:
            messages: Conversation in the shared message format (system, user,
                assistant with optional tool_calls, tool).
            tools: Tools in the ``{"type": "function", "function": {...}}`` shape.
            **kwargs: ``model``, ``max_tokens``, ``options`` (temperature),
                ``response_schema`` (Pydantic model: strict json_schema output),
                ``format`` ("json"), ``tool_choice``.

        Returns:
            The shared response dict: ``message`` (content, tool_calls,
            provider_items), ``usage``, ``done``; or ``error``/``refusal`` entries.
        """
        instructions, input_items = translate_messages_for_responses(messages)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", "gpt-5"),
            "input": input_items,
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            # stateless: the transcript is replayed in full on every round
            "store": False,
        }
        if instructions:
            params["instructions"] = instructions
        if tools:
            params["tools"] = translate_tools_for_responses(tools)
        if kwargs.get("tool_choice") is not None:
            params["tool_choice"] = kwargs["tool_choice"]
        if "options" in kwargs and "temperature" in kwargs["options"]:
            params["temperature"] = kwargs["options"]["temperature"]
        if self.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.reasoning_effort}
        if self.reasoning_effort != "none":
            # reasoning items can only be replayed with their encrypted content
            params["include"] = ["reasoning.encrypted_content"]
        if "response_schema" in kwargs:
            params["text"] = _json_schema_text_format(kwargs["response_schema"])
        elif kwargs.get("format") == "json":
            params["text"] = {"format": {"type": "json_object"}}

        logging.debug("Responses API parameters: %s", params)

        try:
            response = self.client.responses.create(**params)
        except APITimeoutError:
            return {
                "error": "Request timed out.",
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            }
        except RateLimitError as e:
            return {
                "error": f"Rate limited: {e}",
                "error_type": errors.RATE_LIMIT,
                "content": None,
                "done": False,
                "usage": None,
            }
        except APIConnectionError as e:
            return {
                "error": f"Connection error: {e}",
                "error_type": errors.CONNECTION,
                "content": None,
                "done": False,
                "usage": None,
            }

        return self._translate_response(response)

    @staticmethod
    def _translate_response(response: Any) -> Dict[str, Any]:
        output = list(getattr(response, "output", None) or [])
        tool_calls: List[Dict[str, Any]] = []
        texts: List[str] = []
        refusal: Optional[str] = None
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append(
                    {"id": item.call_id, "name": item.name, "arguments": item.arguments}
                )
            elif item_type == "message":
                for part in getattr(item, "content", None) or []:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        texts.append(part.text)
                    elif part_type == "refusal":
                        refusal = part.refusal

        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        usage_info = {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_tokens": _usage_int(getattr(input_details, "cached_tokens", 0)),
            "reasoning_tokens": _usage_int(
                getattr(output_details, "reasoning_tokens", 0)
            ),
        }

        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            if reason == "max_output_tokens":
                return {
                    "error": "Response exceeded the maximum allowed length.",
                    "error_type": errors.LENGTH,
                    "content": None,
                    "done": False,
                    "usage": usage_info,
                }
            if reason == "content_filter":
                return {
                    "error": "Content was rejected by the content filter.",
                    "error_type": errors.CONTENT_FILTER,
                    "content": None,
                    "done": False,
                    "usage": usage_info,
                }

        if refusal is not None and not tool_calls:
            return {
                "refusal": refusal,
                "content": None,
                "done": False,
                "usage": usage_info,
            }

        message: Dict[str, Any] = {"content": "".join(texts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
            # everything the model emitted this turn (reasoning, calls, text),
            # replayed verbatim on the next request
            message[PROVIDER_ITEMS] = [
                item.model_dump(exclude_none=True) for item in output
            ]

        return {
            "message": message,
            "usage": usage_info,
            "done": status == "completed" and not tool_calls,
        }


def _usage_int(value: Any) -> int:
    return value if isinstance(value, int) else 0
