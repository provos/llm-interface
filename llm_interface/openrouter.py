# Copyright 2025 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

import requests
from ollama import ListResponse

from . import errors
from .utils import encode_image_to_base64

# Base URL for OpenRouter API
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def translate_tools_for_openrouter(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Translates the tools format for OpenRouter API.
    In OpenRouter, the format is similar to OpenAI, so we can reuse the same logic.
    """
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
        elif "images" in message and message["images"]:
            content = [{"type": "text", "text": message.get("content", "")}]
            images = message.pop("images")
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,"
                        + encode_image_to_base64(image),
                    }
                )
            message["content"] = content

    return messages


def convert_openrouter_models_to_ollama_response(
    openrouter_models_data,
) -> ListResponse:
    """
    Converts OpenRouter models data into Ollama's ListResponse format.

    Args:
        openrouter_models_data: The response from OpenRouter's models endpoint.

    Returns:
        An instance of Ollama's ListResponse.
    """
    ollama_models = []
    for model in openrouter_models_data:
        # Extract model ID, handling both formats OpenRouter might return
        model_id = model.get("id", model.get("name", "unknown"))

        # Use creation timestamp if available, otherwise current time
        if "created" in model and model["created"]:
            # Convert Unix timestamp to datetime
            modified_at = datetime.fromtimestamp(model["created"], timezone.utc)
        else:
            modified_at = datetime.now(timezone.utc)

        # Extract architecture details
        architecture = model.get("architecture", {})

        # Create model details matching the required Pydantic structure
        # Only include the fields defined in ModelDetails
        details = {
            "parent_model": "",  # OpenRouter doesn't provide parent model info
            "format": architecture.get("tokenizer", "unknown"),
            "family": "openrouter",
            "families": ["openrouter"],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        }

        # Create the model entry
        ollama_model = {
            "model": model_id,
            "modified_at": modified_at,
            "digest": "openrouter-" + model_id,  # Create a unique digest
            "size": 0,
            "details": details,
        }
        ollama_models.append(ListResponse.Model(**ollama_model))

    return ListResponse(models=ollama_models)


class OpenRouterWrapper:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 4096,
        site_url: str = None,
        site_name: str = None,
        timeout: float = 600.0,
    ):
        """
        Initialize an OpenRouterWrapper client.

        Args:
            api_key (str): The OpenRouter API key
            max_tokens (int, optional): Maximum number of tokens for responses. Defaults to 4096.
            site_url (str, optional): Site URL for rankings on openrouter.ai. Defaults to None.
            site_name (str, optional): Site name for rankings on openrouter.ai. Defaults to None.
            timeout (float, optional): Timeout in seconds for API requests. Defaults to 600.0.
        """
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.timeout = timeout

        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Add optional headers for site identification
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        self.session.headers.update(headers)

    def list(self) -> ListResponse:
        """
        List available models from OpenRouter.

        Returns:
            ListResponse: Ollama-compatible response containing model information.
        """
        try:
            response = self.session.get(
                f"{OPENROUTER_API_BASE}/models", timeout=self.timeout
            )
            response.raise_for_status()

            # OpenRouter's models endpoint returns either a "data" array or an array directly
            data = response.json()
            models_data = data.get("data", data) if isinstance(data, dict) else data

            return convert_openrouter_models_to_ollama_response(models_data)
        except Exception as e:
            logging.error("Error fetching OpenRouter models: %s", e)
            # Return empty list on error
            return ListResponse(models=[])

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat request to OpenRouter API.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries with 'role' and 'content' keys.
            tools (Optional[List[Dict[str, Any]]], optional): List of tools available to the model.
            **kwargs: Additional parameters for the API call.

        Returns:
            Dict[str, Any]: The response from OpenRouter API formatted to be compatible with Ollama.
        """
        # Translate tools format if needed
        messages = translate_tools_for_openrouter(messages)

        # Build request parameters
        api_params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add model if provided
        if "model" in kwargs:
            api_params["model"] = kwargs["model"]

        # Configure more specific provider preferences for tool calling
        provider_prefs = {
            "require_parameters": True,  # Only use providers that support all parameters we request
        }

        # If tools are provided, add them and set specific provider preferences
        if tools:
            api_params["tools"] = tools
            # Explicitly add tool-specific provider preferences
            provider_prefs["data_collection"] = (
                "allow"  # Ensure we don't restrict models needed for tool calling
            )

            # Add log of tools being used
            logging.info(
                "Using tools: %s", [t.get("function", {}).get("name") for t in tools]
            )

        # Handle temperature and other options
        if "options" in kwargs:
            if "temperature" in kwargs["options"]:
                api_params["temperature"] = kwargs["options"]["temperature"]

        # Handle JSON format
        if "format" in kwargs and kwargs["format"] == "json":
            api_params["response_format"] = {"type": "json_object"}

        # Handle structured output (Pydantic schema)
        if "response_schema" in kwargs:
            schema = kwargs["response_schema"].model_json_schema()

            # Explicitly set additionalProperties to false as required by OpenAI schema validation
            schema["additionalProperties"] = False

            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                },
            }

        # Set the provider preferences
        api_params["provider"] = provider_prefs

        logging.debug("OpenRouter API request: %s", json.dumps(api_params, indent=2))

        # Initialize usage data with zeros
        usage_data = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            # Make the API call
            response = self.session.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                json=api_params,
                timeout=self.timeout,
            )

            # Log the raw response for debugging
            logging.debug("OpenRouter API response status: %s", response.status_code)
            response_text = response.text
            logging.debug("OpenRouter API response body: %s", response_text[:1000])

            response.raise_for_status()
            response_data = response.json()

            # Extract usage information if available
            if "usage" in response_data:
                usage = response_data["usage"]
                usage_data = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

                logging.info(
                    "Usage details - Prompt tokens: %d, Completion tokens: %d, Total tokens: %d",
                    usage_data["prompt_tokens"],
                    usage_data["completion_tokens"],
                    usage_data["total_tokens"],
                )

            # Format response to be compatible with Ollama/LLMInterface
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                content = None
                tool_calls = None

                if "message" in choice:
                    content = choice["message"].get("content")
                    tool_calls = choice["message"].get("tool_calls")
                elif "delta" in choice:
                    content = choice["delta"].get("content")
                    tool_calls = choice["delta"].get("tool_calls")

                return_message = {
                    "message": {"content": content},
                    "usage": usage_data,
                }

                # Add tool calls if present
                if tool_calls:
                    logging.info("Received tool calls from OpenRouter: %s", tool_calls)
                    return_message["message"]["tool_calls"] = tool_calls

                return return_message
            else:
                # Handle case where no choices are returned
                logging.warning("No choices returned in OpenRouter response")
                if "error" in response_data:
                    error_detail = response_data.get("error", {})
                    logging.error("OpenRouter API error: %s", error_detail)
                    return {
                        "error": f"OpenRouter API error: {error_detail}",
                        "error_type": errors.PROVIDER_SPECIFIC,
                        "message": {"content": ""},
                        "usage": usage_data,
                    }

                return {
                    "message": {"content": None},
                    "error": "No response choices returned from OpenRouter",
                    "error_type": errors.PROVIDER_SPECIFIC,
                    "usage": usage_data,
                }
        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP errors (like 4xx, 5xx)
            error_message = f"HTTP error occurred: {http_err}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.HTTP,
                "content": None,
                "done": False,
                "usage": usage_data,
            }
        except requests.exceptions.ConnectionError:
            error_message = "Connection error: Failed to connect to OpenRouter API"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.CONNECTION,
                "content": None,
                "done": False,
                "usage": usage_data,
            }
        except requests.exceptions.Timeout:
            error_message = "Request timeout: OpenRouter API request timed out"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": usage_data,
            }
        except requests.exceptions.RequestException as e:
            error_message = f"OpenRouter API error: {str(e)}"
            logging.error(error_message)
            return {
                "error": error_message,
                "error_type": errors.PROVIDER_SPECIFIC,
                "content": None,
                "done": False,
                "usage": usage_data,
            }
