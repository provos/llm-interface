import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import requests
from ollama import ListResponse

from llm_interface.openrouter import (
    OpenRouterWrapper,
    convert_openrouter_models_to_ollama_response,
    translate_tools_for_openrouter,
)


class TestOpenRouterWrapper(unittest.TestCase):
    def setUp(self):
        # Mock the requests Session
        self.mock_session = Mock()

        # Patch the requests.Session to use our mock
        patcher = patch(
            "llm_interface.openrouter.requests.Session", return_value=self.mock_session
        )
        self.addCleanup(patcher.stop)
        self.mock_requests_session = patcher.start()

        # Initialize wrapper with test API key
        self.api_key = "test_api_key"
        self.openrouter_wrapper = OpenRouterWrapper(
            api_key=self.api_key, site_url="https://test.com", site_name="Test App"
        )

        # Verify headers were set correctly
        self.mock_session.headers.update.assert_called_once()
        headers = self.mock_session.headers.update.call_args[0][0]
        self.assertEqual(headers["Authorization"], f"Bearer {self.api_key}")
        self.assertEqual(headers["HTTP-Referer"], "https://test.com")
        self.assertEqual(headers["X-Title"], "Test App")

    def test_list_models(self):
        # Sample models data from OpenRouter
        models_data = [
            {
                "id": "anthropic/claude-2",
                "name": "Claude 2",
                "created": 1689979576,
                "architecture": {"tokenizer": "claude"},
            },
            {
                "id": "openai/gpt-4",
                "name": "GPT-4",
                "created": 1687882411,
                "architecture": {"tokenizer": "gpt"},
            },
        ]

        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = {"data": models_data}
        mock_response.raise_for_status = Mock()
        self.mock_session.get.return_value = mock_response

        # Call the list method
        response = self.openrouter_wrapper.list()

        # Verify the API call
        self.mock_session.get.assert_called_once_with(
            "https://openrouter.ai/api/v1/models",
            timeout=600.0,
        )

        # Verify the response format matches Ollama's format
        self.assertIsInstance(response, ListResponse)
        self.assertEqual(len(response.models), 2)

        # Verify first model data
        model = response.models[0]
        self.assertEqual(model.model, "anthropic/claude-2")
        self.assertEqual(
            model.modified_at, datetime.fromtimestamp(1689979576, tz=timezone.utc)
        )
        self.assertEqual(model.digest, "openrouter-anthropic/claude-2")

        # Verify model details
        self.assertEqual(model.details.family, "openrouter")
        self.assertEqual(model.details.families, ["openrouter"])
        self.assertEqual(model.details.parameter_size, "unknown")
        self.assertEqual(model.details.quantization_level, "unknown")

    def test_list_models_error(self):
        # Configure the mock response to raise an exception
        self.mock_session.get.side_effect = requests.exceptions.ConnectionError(
            "Connection error"
        )

        # Call the list method - should return empty models list
        response = self.openrouter_wrapper.list()

        # Verify the response is empty
        self.assertEqual(len(response.models), 0)

    def test_chat_basic(self):
        # Sample chat response
        chat_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you today?",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = chat_response
        mock_response.text = json.dumps(chat_response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Call the chat method
        messages = [{"role": "user", "content": "Hello!"}]
        response = self.openrouter_wrapper.chat(messages=messages)

        # Verify the API call
        self.mock_session.post.assert_called_once()
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/chat/completions")

        # Check request data
        request_data = kwargs["json"]
        self.assertEqual(request_data["messages"], messages)
        self.assertEqual(request_data["max_tokens"], 4096)

        # Verify the response
        self.assertEqual(
            response["message"]["content"], "Hello, how can I help you today?"
        )
        self.assertNotIn("tool_calls", response["message"])

    def test_chat_with_tools(self):
        # Sample chat response with tool calls
        chat_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"New York","unit":"celsius"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
        }

        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = chat_response
        mock_response.text = json.dumps(chat_response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        # Call the chat method with tools
        messages = [{"role": "user", "content": "What's the weather in New York?"}]
        response = self.openrouter_wrapper.chat(messages=messages, tools=tools)

        # Verify the API call included tools
        self.mock_session.post.assert_called_once()
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(kwargs["json"]["tools"], tools)

        # Verify the response includes tool calls
        self.assertIsNone(response["message"]["content"])
        self.assertIn("tool_calls", response["message"])
        tool_call = response["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(
            tool_call["function"]["arguments"],
            '{"location":"New York","unit":"celsius"}',
        )

    def test_chat_with_model_parameter(self):
        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Response from specified model",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.text = '{"choices":[{"message":{"content":"Response from specified model","role":"assistant"},"finish_reason":"stop"}]}'
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Call with specific model
        messages = [{"role": "user", "content": "Test message"}]
        self.openrouter_wrapper.chat(messages=messages, model="anthropic/claude-3-opus")

        # Verify model was in the request
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(kwargs["json"]["model"], "anthropic/claude-3-opus")

    def test_chat_with_options(self):
        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Response with temperature",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.text = '{"choices":[{"message":{"content":"Response with temperature","role":"assistant"},"finish_reason":"stop"}]}'
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Call with options
        messages = [{"role": "user", "content": "Test message"}]
        options = {"temperature": 0.7}
        self.openrouter_wrapper.chat(messages=messages, options=options)

        # Verify temperature was in the request
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(kwargs["json"]["temperature"], 0.7)

    def test_chat_with_json_format(self):
        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": '{"key": "value"}', "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.text = '{"choices":[{"message":{"content":"{\\"key\\": \\"value\\"}","role":"assistant"},"finish_reason":"stop"}]}'
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Call with JSON format
        messages = [{"role": "user", "content": "Return JSON"}]
        self.openrouter_wrapper.chat(messages=messages, format="json")

        # Verify response_format was in the request
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(kwargs["json"]["response_format"]["type"], "json_object")

    def test_chat_with_response_schema(self):
        # Mock schema
        class MockSchema:
            @classmethod
            def model_json_schema(cls):
                return {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                }

        # Configure the mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "John", "age": 30}',
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.text = '{"choices":[{"message":{"content":"{\\"name\\": \\"John\\", \\"age\\": 30}","role":"assistant"},"finish_reason":"stop"}]}'
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        self.mock_session.post.return_value = mock_response

        # Call with response schema
        messages = [{"role": "user", "content": "Get user data"}]
        self.openrouter_wrapper.chat(messages=messages, response_schema=MockSchema)

        # Verify json_schema was in the request
        args, kwargs = self.mock_session.post.call_args
        self.assertEqual(kwargs["json"]["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["json"]["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            kwargs["json"]["response_format"]["json_schema"]["name"],
            "structured_response",
        )
        self.assertFalse(
            kwargs["json"]["response_format"]["json_schema"]["schema"][
                "additionalProperties"
            ]
        )

    def test_http_error_handling(self):
        # Configure the mock response to raise an HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error"
        )
        mock_response.text = '{"error":{"message":"Model not found"}}'
        mock_response.json.return_value = {"error": {"message": "Model not found"}}
        mock_response.status_code = 404
        self.mock_session.post.return_value = mock_response

        # Call the chat method
        messages = [{"role": "user", "content": "Hello"}]
        response = self.openrouter_wrapper.chat(messages=messages)

        # Verify error handling
        self.assertIn("error", response)
        self.assertEqual(response["error"], "HTTP error occurred: 404 Client Error")
        self.assertIsNone(response["content"])
        self.assertFalse(response["done"])

    def test_connection_error_handling(self):
        # Configure the mock to raise a connection error
        self.mock_session.post.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        # Call the chat method
        messages = [{"role": "user", "content": "Hello"}]
        response = self.openrouter_wrapper.chat(messages=messages)

        # Verify error handling
        self.assertIn("error", response)
        self.assertEqual(
            response["error"], "Connection error: Failed to connect to OpenRouter API"
        )
        self.assertIsNone(response["content"])
        self.assertFalse(response["done"])

    def test_timeout_error_handling(self):
        # Configure the mock to raise a timeout
        self.mock_session.post.side_effect = requests.exceptions.Timeout(
            "Request timed out"
        )

        # Call the chat method
        messages = [{"role": "user", "content": "Hello"}]
        response = self.openrouter_wrapper.chat(messages=messages)

        # Verify error handling
        self.assertIn("error", response)
        self.assertEqual(
            response["error"], "Request timeout: OpenRouter API request timed out"
        )
        self.assertIsNone(response["content"])
        self.assertFalse(response["done"])

    def test_translate_tools_for_openrouter(self):
        # Test with dict arguments
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "weather", "location": "New York"},
                        }
                    }
                ],
            }
        ]

        translated = translate_tools_for_openrouter(messages)

        # Verify dict was converted to JSON string
        self.assertEqual(
            json.loads(translated[0]["tool_calls"][0]["function"]["arguments"]),
            {"query": "weather", "location": "New York"},
        )

    def test_convert_openrouter_models_to_ollama_response(self):
        # Sample OpenRouter models data
        models_data = [
            {"id": "mistral/mistral-7b", "created": 1650000000},
            {"id": "anthropic/claude-3-haiku", "created": None},
        ]

        # Convert to Ollama format
        result = convert_openrouter_models_to_ollama_response(models_data)

        # Verify result structure
        self.assertIsInstance(result, ListResponse)
        self.assertEqual(len(result.models), 2)
        self.assertEqual(result.models[0].model, "mistral/mistral-7b")
        self.assertEqual(
            result.models[0].modified_at,
            datetime.fromtimestamp(1650000000, tz=timezone.utc),
        )
        # Second model should use current time since created is None
        self.assertIsNotNone(result.models[1].modified_at)


if __name__ == "__main__":
    unittest.main()
