# Copyright 2025 Niels Provos
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from ollama import ListResponse
from pydantic import BaseModel


def translate_tools_for_gemini(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates tools from Ollama format to Gemini format.
    """
    gemini_tools = []
    for tool in tools:
        function = tool["function"]
        gemini_tool = {
            "function_declarations": [
                {
                    "name": function["name"],
                    "description": function["description"],
                    "parameters": {
                        "type": "OBJECT",
                        "properties": function["parameters"]["properties"],
                        "required": function["parameters"]["required"],
                    },
                }
            ]
        }
        gemini_tools.append(gemini_tool)
    return gemini_tools


def translate_messages_for_gemini(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Translates messages from Ollama format to Gemini format.
    Handles text, tool calls, and images.
    """
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Skip system messages as they're handled separately
            continue
        if msg["role"] == "tool":
            # Convert tool responses to proper Gemini functionResponse format
            content = {
                "functionResponse": {
                    "name": msg["name"],
                    "response": {
                        "name": msg["name"],
                        "content": json.loads(msg["content"]),
                    },
                }
            }
            gemini_messages.append({"role": "user", "parts": [content]})
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            # Convert assistant tool calls to proper Gemini functionCall format
            tool_call = msg["tool_calls"][0]["function"]
            content = {
                "functionCall": {
                    "name": tool_call["name"],
                    "args": (
                        json.loads(tool_call["arguments"])
                        if isinstance(tool_call["arguments"], str)
                        else tool_call["arguments"]
                    ),
                }
            }
            gemini_messages.append({"role": "model", "parts": [content]})
        else:
            parts = []
            # Add text content if it exists
            if "content" in msg and msg["content"]:
                parts.append({"text": msg["content"]})
            # Check if message contains images
            if "images" in msg and msg["images"]:
                # Add each image as a separate part
                for image_path in msg["images"]:
                    image = types.Part.from_bytes(
                        data=Path(image_path).read_bytes(), mime_type="image/jpeg"
                    )
                    parts.append(image)
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": parts})

    return gemini_messages


def convert_gemini_models_to_ollama_response(models_data) -> ListResponse:
    """
    Converts Gemini model data to Ollama's ListResponse format.
    """
    ollama_models = []
    for model in models_data:
        # Use current time as modified_at since Gemini doesn't provide this
        modified_at = datetime.now(timezone.utc)

        model_data = {
            "model": model.name,
            "modified_at": modified_at,
            "digest": "unknown",
            "size": 0,
            "details": {
                "parent_model": "",
                "format": "unknown",
                "family": "gemini",
                "families": ["gemini"],
                "parameter_size": "unknown",
                "quantization_level": "unknown",
            },
        }
        ollama_models.append(ListResponse.Model(**model_data))

    return ListResponse(models=ollama_models)


class GeminiWrapper:
    def __init__(self, api_key: str, max_tokens: int = 4096, timeout: float = 600.0):
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout)
        )
        self.max_tokens = max_tokens

    def list(self) -> ListResponse:
        """List available models in Ollama format."""
        models = self.client.list_models()
        return convert_gemini_models_to_ollama_response(models)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation using the Gemini API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional parameters including model name, temperature, etc.
        """
        try:
            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Convert messages to Gemini format
            contents = translate_messages_for_gemini(messages)
            if system_message:
                # Add system message as first content
                contents.insert(
                    0,
                    {"role": "user", "parts": [{"text": f"System: {system_message}"}]},
                )

            # Prepare configuration
            config = {}
            if "max_tokens" in kwargs:
                config["max_output_tokens"] = kwargs["max_tokens"]
            elif self.max_tokens:
                config["max_output_tokens"] = self.max_tokens

            if "options" in kwargs and "temperature" in kwargs["options"]:
                config["temperature"] = kwargs["options"]["temperature"]

            # Handle tools if provided
            if tools:
                gemini_tools = translate_tools_for_gemini(tools)
                config["tools"] = gemini_tools
                # AUTO means the model can decide when to call the function
                # ANY means it will always call a function
                config["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

            # Handle structured output if response_schema is provided
            if "response_schema" in kwargs:
                schema = kwargs["response_schema"]
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    if tools:
                        raise ValueError(
                            "The Gemini API does not support structured output with function calling/tools."
                        )
                    config.update(
                        {
                            "response_mime_type": "application/json",
                            "response_schema": schema,
                        }
                    )

            response = self.client.models.generate_content(
                model=kwargs.get("model", "gemini-2.0-flash"),
                contents=contents,
                config=config,
            )

            # Create response object
            result = {}

            # Add token usage to the response object
            if hasattr(response, "usage_metadata"):
                result["usage"] = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            # Check for function calls in the response
            if hasattr(response, "candidates") and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if (
                        hasattr(part, "function_call")
                        and part.function_call is not None
                    ):
                        result["message"] = {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": getattr(part.function_call, "id", ""),
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(part.function_call.args),
                                }
                            ],
                        }
                        return result

            # Return regular message response
            result["message"] = {"content": response.text}
            return result

        except Exception as e:
            logging.error(f"Error in Gemini chat: {str(e)}")
            raise e
